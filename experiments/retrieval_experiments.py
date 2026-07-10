"""
检索增强实验：Query Rewriting & Multi-Query
用途：对比原始 query / 改写后 query / Multi-Query 的检索效果

实验目录：data_output/experiments/

用法：
  python experiments/retrieval_experiments.py --experiment rewrite
  python experiments/retrieval_experiments.py --experiment multi_query
  python experiments/retrieval_experiments.py --experiment all
"""
import argparse
import json
import os
import sys
import time
import warnings
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ====== 配置 ======
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = "E:/models/Qwen2.5-1.5B"
EMBEDDING_MODEL = "BAAI/bge-large-zh"
CHROMA_DIR = os.path.join(PROJECT_ROOT, 'output', 'chroma_db_ov75')
CHROMA_DIR_HTML = os.path.join(PROJECT_ROOT, 'output', 'chroma_db_html_cs500')
CHROMA_DIR_CSV = os.path.join(PROJECT_ROOT, 'output', 'chroma_db_csv')
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, 'data_output', 'experiments')
TOP_K = 8  # 默认 top-k，可改

# ====== Qwen2-0.5B 模型 ======

_qwen_model = None
_qwen_tokenizer = None


def load_qwen():
    """加载 Qwen2-0.5B 量化模型"""
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is not None:
        return _qwen_tokenizer, _qwen_model

    print("正在加载 Qwen2-0.5B-Instruct...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  ✓ 模型加载完成，耗时 {time.perf_counter() - t0:.1f}s")
    _qwen_model = model
    _qwen_tokenizer = tokenizer
    return tokenizer, model


def qwen_generate(prompt: str, max_new_tokens: int = 256, temperature: float = 0.3):
    """调用 Qwen2-0.5B 生成"""
    tokenizer, model = load_qwen()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
        )
    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
    )
    return response.strip()


# ====== Query Rewriting ======

REWRITE_PROMPT = """You are a query rewriter. Rewrite the following question to be more precise, self-contained, and search-friendly for a RAG retrieval system.

Rules:
1. Expand abbreviations and acronyms (e.g., "BERT" → "BERT (Bidirectional Encoder Representations from Transformers)")
2. Add context from related concepts if the question is vague
3. Keep the original intent
4. Output ONLY the rewritten question, nothing else.

Original question: {question}
Rewritten question:"""


def rewrite_query(question: str) -> str:
    """使用 Qwen2-0.5B 改写查询"""
    prompt = REWRITE_PROMPT.format(question=question)
    result = qwen_generate(prompt, max_new_tokens=128, temperature=0.3)
    # 清理可能的额外输出
    result = result.split('\n')[0].strip().strip('"').strip("'")
    return result if len(result) > 5 else question  # 失败时返回原文


# ====== Multi-Query Generation ======

MULTI_QUERY_PROMPT = """Generate 3 different variations of the following question. Each variation should rephrase the question using different wording or perspective, but keep the same intent.

Output EXACTLY 3 lines, each line is one variation. No numbering, no extra text.

Original question: {question}
Variations:"""


def generate_multi_queries(question: str) -> list:
    """使用 Qwen2-0.5B 生成 3 个查询变体"""
    prompt = MULTI_QUERY_PROMPT.format(question=question)
    result = qwen_generate(prompt, max_new_tokens=256, temperature=0.7)
    lines = [l.strip().lstrip('-').lstrip('0123456789.').strip() for l in result.split('\n') if l.strip()]
    # 过滤太短的
    lines = [l for l in lines if len(l) > 10]
    if len(lines) < 2:
        return [question, question, question]
    return lines[:3]


# ====== 检索 ======

def load_chroma_db(path: str, embeddings):
    """加载一个 Chroma 数据库"""
    db = Chroma(persist_directory=path, embedding_function=embeddings)
    return db


def retrieve_and_check(db, query: str, expected_sources: list, k: int = TOP_K):
    """检索 top-k，检查 HitRate 和 Recall"""
    results = db.similarity_search_with_score(query, k=k)
    sources = [doc.metadata.get('source', 'unknown') for doc, _ in results]

    expected_set = set(expected_sources)
    hit = any(s in expected_set for s in sources[:k])
    hits = sum(1 for s in sources[:k] if s in expected_set)
    recall = hits / k
    return {
        'hit': hit,
        'recall': recall,
        'hits': hits,
        'sources': sources[:k],
    }


def retrieve_multi_query_dbs(dbs, query: str, expected_sources: list, k: int = TOP_K):
    """在多个数据库中检索，合并结果"""
    all_docs = []
    for db in dbs:
        results = db.similarity_search_with_score(query, k=k)
        for doc, score in results:
            all_docs.append({
                'content': doc.page_content[:100],
                'source': doc.metadata.get('source', 'unknown'),
                'score': float(score),
            })
    # 按分数排序，去重取 top-k
    seen = set()
    merged = []
    for doc in sorted(all_docs, key=lambda x: x['score']):
        if doc['source'] not in seen:
            merged.append(doc)
            seen.add(doc['source'])
        if len(merged) >= k:
            break

    sources = [d['source'] for d in merged]
    expected_set = set(expected_sources)
    hit = any(s in expected_set for s in sources[:k])
    hits = sum(1 for s in sources[:k] if s in expected_set)
    recall = hits / k
    return {
        'hit': hit,
        'recall': recall,
        'hits': hits,
        'sources': sources[:k],
    }


# ====== 加载问题集 ======

def load_all_questions():
    """加载所有问题集"""
    all_questions = []
    paths = [
        (os.path.join(PROJECT_ROOT, 'data_output', 'eval_questions.json'), 'chroma_db_ov75'),
        (os.path.join(PROJECT_ROOT, 'data_output', 'eval_questions_html.json'), 'chroma_db_html_cs500'),
        (os.path.join(PROJECT_ROOT, 'data_output', 'eval_questions_csv.json'), 'chroma_db_csv'),
    ]

    for path, db_name in paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for q in data['questions']:
            q['db'] = db_name
            all_questions.append(q)

    return all_questions


# ====== 主实验 ======

def run_rewrite_experiment(questions: list):
    """Query Rewriting 实验"""
    print("\n" + "=" * 70)
    print("实验 1: Query Rewriting — HitRate 对比")
    print("=" * 70)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    dbs = {
        'chroma_db_ov75': load_chroma_db(CHROMA_DIR, embeddings),
        'chroma_db_html_cs500': load_chroma_db(CHROMA_DIR_HTML, embeddings),
        'chroma_db_csv': load_chroma_db(CHROMA_DIR_CSV, embeddings),
    }

    results = []
    rewrite_times = []

    for i, q in enumerate(questions):
        db = dbs.get(q.get('db', 'chroma_db_ov75'))
        if db is None:
            continue

        expected_sources = [q['source']]

        # 原始查询
        t0 = time.perf_counter()
        orig = retrieve_and_check(db, q['question'], expected_sources)
        orig_time = time.perf_counter() - t0

        # 改写查询
        t0 = time.perf_counter()
        rewritten = rewrite_query(q['question'])
        rewrite_time = time.perf_counter() - t0
        rewrite_times.append(rewrite_time * 1000)

        t0 = time.perf_counter()
        rw_result = retrieve_and_check(db, rewritten, expected_sources)
        rw_retrieval_time = time.perf_counter() - t0

        results.append({
            'id': q['id'],
            'question': q['question'],
            'rewritten': rewritten,
            'source': q['source'],
            'type': q.get('type', 'unknown'),
            'orig_hit': orig['hit'],
            'orig_recall': orig['recall'],
            'rw_hit': rw_result['hit'],
            'rw_recall': rw_result['recall'],
            'rewrite_time_ms': round(rewrite_time * 1000, 1),
        })

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(questions)} 题")

    # 统计
    orig_hit_rate = sum(r['orig_hit'] for r in results) / len(results)
    rw_hit_rate = sum(r['rw_hit'] for r in results) / len(results)
    avg_rw_time = sum(rewrite_times) / len(rewrite_times) if rewrite_times else 0

    print(f"\n{'='*70}")
    print(f"原始 HitRate: {orig_hit_rate:.4f}")
    print(f"改写后 HitRate: {rw_hit_rate:.4f}")
    print(f"提升: {(rw_hit_rate - orig_hit_rate)*100:+.1f}%")
    print(f"平均改写耗时: {avg_rw_time:.1f}ms")

    # 按类型分析
    for qtype in ['factoid', 'reasoning', 'comparison']:
        typed = [r for r in results if r['type'] == qtype]
        if not typed:
            continue
        orig_h = sum(r['orig_hit'] for r in typed) / len(typed)
        rw_h = sum(r['rw_hit'] for r in typed) / len(typed)
        print(f"  {qtype}: {orig_h:.4f} → {rw_h:.4f} ({(rw_h-orig_h)*100:+.1f}%)")

    # 保存
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    output_path = os.path.join(EXPERIMENT_DIR, 'rewrite_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'query_rewriting',
            'model': 'Qwen2-0.5B-Instruct (FP16)',
            'total_questions': len(results),
            'original_hit_rate': orig_hit_rate,
            'rewritten_hit_rate': rw_hit_rate,
            'improvement': rw_hit_rate - orig_hit_rate,
            'avg_rewrite_time_ms': avg_rw_time,
            'by_type': {
                qtype: {
                    'original': sum(r['orig_hit'] for r in typed) / len(typed),
                    'rewritten': sum(r['rw_hit'] for r in typed) / len(typed),
                }
                for qtype in ['factoid', 'reasoning', 'comparison']
                if (typed := [r for r in results if r['type'] == qtype])
            },
            'details': results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 结果已保存: {output_path}")
    return results


def run_multi_query_experiment(questions: list):
    """Multi-Query ×3 实验"""
    print("\n" + "=" * 70)
    print("实验 2: Multi-Query ×3 — Recall/Precision 对比")
    print("=" * 70)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    dbs = {
        'chroma_db_ov75': load_chroma_db(CHROMA_DIR, embeddings),
        'chroma_db_html_cs500': load_chroma_db(CHROMA_DIR_HTML, embeddings),
        'chroma_db_csv': load_chroma_db(CHROMA_DIR_CSV, embeddings),
    }

    results = []
    gen_times = []
    total_orig_times = []
    total_mq_times = []

    for i, q in enumerate(questions):
        db = dbs.get(q.get('db', 'chroma_db_ov75'))
        if db is None:
            continue

        expected_sources = [q['source']]

        # 原始单查询 (recall@5, 与 Multi-Query 公平对比)
        t0 = time.perf_counter()
        orig = retrieve_and_check(db, q['question'], expected_sources, k=5)
        orig_time = time.perf_counter() - t0
        total_orig_times.append(orig_time * 1000)

        # 生成 3 个变体
        t0 = time.perf_counter()
        variations = generate_multi_queries(q['question'])
        gen_time = time.perf_counter() - t0
        gen_times.append(gen_time * 1000)

        # 用原始查询 + 3 个变体分别检索，合并去重
        t0 = time.perf_counter()
        all_queries = [q['question']] + variations
        all_docs = []
        for query in all_queries:
            docs = db.similarity_search_with_score(query, k=8)
            for doc, score in docs:
                all_docs.append({
                    'source': doc.metadata.get('source', 'unknown'),
                    'score': float(score),
                })
        # 按 source 去重（保留最低分数 = 最相似），取 top-5
        best_by_source = {}
        for doc in all_docs:
            s = doc['source']
            if s not in best_by_source or doc['score'] < best_by_source[s]['score']:
                best_by_source[s] = doc
        sorted_docs = sorted(best_by_source.values(), key=lambda x: x['score'])
        top_docs = sorted_docs[:5]
        mq_sources = [d['source'] for d in top_docs]

        expected_set = set(expected_sources)
        mq_hit = any(s in expected_set for s in mq_sources[:5])
        mq_hits = sum(1 for s in mq_sources[:5] if s in expected_set)
        mq_recall = mq_hits / 5
        mq_time = time.perf_counter() - t0
        total_mq_times.append(mq_time * 1000)

        results.append({
            'id': q['id'],
            'question': q['question'],
            'variations': variations,
            'source': q['source'],
            'type': q.get('type', 'unknown'),
            'orig_hit': orig['hit'],
            'orig_recall': orig['recall'],
            'mq_hit': mq_hit,
            'mq_recall': mq_recall,
            'gen_time_ms': round(gen_time * 1000, 1),
            'orig_retrieval_time_ms': round(orig_time * 1000, 1),
            'mq_total_time_ms': round(mq_time * 1000, 1),
        })

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(questions)} 题")

    # 统计
    orig_recall_avg = sum(r['orig_recall'] for r in results) / len(results)
    mq_recall_avg = sum(r['mq_recall'] for r in results) / len(results)
    avg_gen_time = sum(gen_times) / len(gen_times) if gen_times else 0
    avg_mq_time = sum(total_mq_times) / len(total_mq_times) if total_mq_times else 0
    avg_orig_time = sum(total_orig_times) / len(total_orig_times) if total_orig_times else 0

    print(f"\n{'='*70}")
    print(f"原始 Recall@5: {orig_recall_avg:.4f}")
    print(f"Multi-Query Recall@5: {mq_recall_avg:.4f}")
    print(f"提升: {(mq_recall_avg - orig_recall_avg)*100:+.1f}%")
    print(f"Multi-Query 生成耗时: {avg_gen_time:.0f}ms")
    print(f"原始检索耗时: {avg_orig_time:.0f}ms")
    print(f"Multi-Query 检索耗时: {avg_mq_time:.0f}ms")
    print(f"Multi-Query 总延迟: {avg_gen_time + avg_mq_time:.0f}ms")

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    output_path = os.path.join(EXPERIMENT_DIR, 'multi_query_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'multi_query_x3',
            'model': 'Qwen2-0.5B-Instruct (FP16)',
            'total_questions': len(results),
            'original_recall_at_5': orig_recall_avg,
            'multi_query_recall_at_5': mq_recall_avg,
            'improvement': mq_recall_avg - orig_recall_avg,
            'avg_generation_time_ms': avg_gen_time,
            'avg_retrieval_time_ms': avg_mq_time,
            'total_latency_ms': avg_gen_time + avg_mq_time,
            'details': results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 结果已保存: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='检索增强实验')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['rewrite', 'multi_query', 'all'],
                        help='实验类型')
    parser.add_argument('--top_k', type=int, default=8,
                        help='检索 top-k (默认: 8)')
    args = parser.parse_args()

    # 使用局部变量避免 global 声明问题
    top_k = args.top_k

    # 将 top_k 传给各函数需要的位置...
    # 所有 retrieve_and_check / retrieve_multi_query_dbs 调用使用 top_k

    # 加载问题集
    questions = load_all_questions()
    print(f"加载了 {len(questions)} 道测试题")

    if args.experiment in ['rewrite', 'all']:
        rewrite_results = run_rewrite_experiment(questions)

    if args.experiment in ['multi_query', 'all']:
        mq_results = run_multi_query_experiment(questions)

    if args.experiment == 'all':
        print("\n" + "=" * 70)
        print("📊 实验总结")
        print("=" * 70)
        # 读取两个结果文件汇总
        for exp_name in ['rewrite', 'multi_query']:
            path = os.path.join(EXPERIMENT_DIR, f'{exp_name}_results.json')
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if exp_name == 'rewrite':
                    imp = f"{data['improvement']*100:+.1f}%"
                    print(f"  Query Rewriting: HitRate {data['original_hit_rate']:.4f} → {data['rewritten_hit_rate']:.4f} ({imp}), 模型耗时 {data['avg_rewrite_time_ms']:.0f}ms")
                else:
                    imp = f"{data['improvement']*100:+.1f}%"
                    print(f"  Multi-Query ×3: Recall@5 {data['original_recall_at_5']:.4f} → {data['multi_query_recall_at_5']:.4f} ({imp}), 总延迟 {data['total_latency_ms']:.0f}ms")

    print("\n✅ 实验完成！结果目录:", EXPERIMENT_DIR)


if __name__ == '__main__':
    main()
