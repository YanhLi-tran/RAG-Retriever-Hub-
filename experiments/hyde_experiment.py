"""
HyDE (Hypothetical Document Embeddings) 实验
流程：用户 query → LLM 生成假设文档 → embedding 假设文档 → 检索 → 对比 HitRate
"""
import json
import os
import sys
import time
import warnings
from dotenv import load_dotenv
from openai import OpenAI

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1'),
)
MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL = "BAAI/bge-large-zh"
TOP_K = 8

HYDE_PROMPT = """You are a helpful AI assistant. Given a question, write a short paragraph (3-5 sentences) that would be an ideal answer to the question. Write as if it's a factual passage from a textbook or research paper.

Question: {question}
Passage:"""


def generate_hyde_passage(question: str) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
                temperature=0.3,
                max_tokens=256,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return question


def main():
    print("=" * 70)
    print("HyDE (Hypothetical Document Embeddings) 实验")
    print("=" * 70)

    all_questions = []
    paths = [
        ('data_output/eval_questions.json', 'chroma_db_ov75'),
        ('data_output/eval_questions_html.json', 'chroma_db_html_cs500'),
    ]
    for rel, db in paths:
        fp = os.path.join(PROJECT_ROOT, rel)
        if not os.path.exists(fp):
            continue
        with open(fp, 'r', encoding='utf-8') as f:
            for q in json.load(f)['questions']:
                q['db'] = db
                all_questions.append(q)

    print(f"加载了 {len(all_questions)} 道测试题")
    qid_map = {q['id']: q for q in all_questions}

    print("加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    print("加载 Chroma 数据库...")
    dbs = {
        'chroma_db_ov75': Chroma(persist_directory='output/chroma_db_ov75', embedding_function=embeddings),
        'chroma_db_html_cs500': Chroma(persist_directory='output/chroma_db_html_cs500', embedding_function=embeddings),
    }

    # 阶段 1: 生成 HyDE 文档
    print("\n阶段 1: 生成假设文档 (DeepSeek API)...")
    hyde_results = []
    gen_times = []
    for i, q in enumerate(all_questions):
        t0 = time.perf_counter()
        passage = generate_hyde_passage(q['question'])
        elapsed = time.perf_counter() - t0
        gen_times.append(elapsed * 1000)
        hyde_results.append({
            'id': q['id'],
            'question': q['question'],
            'hypothetical_doc': passage,
            'gen_time_ms': round(elapsed * 1000, 1),
            'source': q['source'],
            'db': q['db'],
        })
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(all_questions)} 题")

    avg_gen = sum(gen_times) / len(gen_times)
    print(f"  平均生成耗时: {avg_gen:.0f}ms")

    # 阶段 2: 检索对比
    print("\n阶段 2: 检索对比 (query vs HyDE)...")
    results = []
    for r in hyde_results:
        qinfo = qid_map.get(r['id'])
        if not qinfo:
            continue
        db = dbs.get(qinfo['db'])
        if not db:
            continue
        expected = [qinfo['source']]

        # 原始 query 检索
        orig_res = db.similarity_search_with_score(r['question'], k=TOP_K)
        orig_hit = any(d.metadata.get('source', '') in expected for d, _ in orig_res)
        orig_srcs = [d.metadata.get('source', '') for d, _ in orig_res]
        orig_recall = sum(1 for s in orig_srcs if s in expected) / TOP_K

        # HyDE 检索
        hyde_res = db.similarity_search_with_score(r['hypothetical_doc'], k=TOP_K)
        hyde_hit = any(d.metadata.get('source', '') in expected for d, _ in hyde_res)
        hyde_srcs = [d.metadata.get('source', '') for d, _ in hyde_res]
        hyde_recall = sum(1 for s in hyde_srcs if s in expected) / TOP_K

        results.append({
            'id': r['id'],
            'orig_hit': orig_hit,
            'orig_recall': orig_recall,
            'hyde_hit': hyde_hit,
            'hyde_recall': hyde_recall,
            'gen_time_ms': r['gen_time_ms'],
        })

    orig_hit_rate = sum(r['orig_hit'] for r in results) / len(results)
    hyde_hit_rate = sum(r['hyde_hit'] for r in results) / len(results)
    orig_recall_avg = sum(r['orig_recall'] for r in results) / len(results)
    hyde_recall_avg = sum(r['hyde_recall'] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"原始 HitRate:      {orig_hit_rate:.4f}")
    print(f"HyDE   HitRate:    {hyde_hit_rate:.4f} ({(hyde_hit_rate-orig_hit_rate)*100:+.1f}%)")
    print(f"原始 Recall@8:     {orig_recall_avg:.4f}")
    print(f"HyDE   Recall@8:   {hyde_recall_avg:.4f} ({(hyde_recall_avg-orig_recall_avg)*100:+.1f}%)")
    print(f"平均生成耗时: {avg_gen:.0f}ms")
    print(f"{'='*60}")

    os.makedirs('data_output/experiments', exist_ok=True)
    out = {
        'experiment': 'hyde',
        'model': 'DeepSeek Chat',
        'total': len(results),
        'original_hit_rate': orig_hit_rate,
        'hyde_hit_rate': hyde_hit_rate,
        'hit_rate_improvement': hyde_hit_rate - orig_hit_rate,
        'original_recall_at_8': orig_recall_avg,
        'hyde_recall_at_8': hyde_recall_avg,
        'recall_improvement': hyde_recall_avg - orig_recall_avg,
        'avg_generation_time_ms': avg_gen,
    }
    with open('data_output/experiments/hyde_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: data_output/experiments/hyde_results.json")


if __name__ == '__main__':
    main()
