"""
RAG 系统评估脚本
功能：
  1. 加载测试问题集
  2. 对每个问题进行向量检索（top-10）
  3. 计算 Recall@10
  4. 将 top-10 chunks 传入 DeepSeek 生成答案（带来源溯源）
  5. 使用 Ragas 评估生成质量

用法：
  python scripts/evaluate_rag.py --max_questions 10
  python scripts/evaluate_rag.py --model deepseek-chat --top_k 10
  python scripts/evaluate_rag.py --resume data_output/eval_results/results.json
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# 加载 .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ====== Ragas 兼容补丁 ======
# ragas 0.4.3 仍依赖 langchain_community.chat_models.vertexai，但当前环境缺少此模块
import types
from unittest.mock import MagicMock
_vtx_mod = types.ModuleType('langchain_community.chat_models.vertexai')
_vtx_mod.ChatVertexAI = MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = _vtx_mod
# ===========================

import numpy as np
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

sys.stdout.reconfigure(encoding='utf-8')

# ====== 配置 ======
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(PROJECT_ROOT, 'output', 'chroma_db')
QUESTIONS_PATH = os.path.join(PROJECT_ROOT, 'data_output', 'eval_questions.json')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'data_output', 'eval_results')

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

EMBEDDING_MODEL = "BAAI/bge-large-zh"


def parse_args():
    parser = argparse.ArgumentParser(description='RAG 系统评估')
    parser.add_argument('--questions', type=str, default=QUESTIONS_PATH,
                        help='测试问题集路径')
    parser.add_argument('--model', type=str, default=DEEPSEEK_MODEL,
                        help='DeepSeek 模型名称')
    parser.add_argument('--top_k', type=int, default=10,
                        help='检索返回的 chunk 数 (默认: 10)')
    parser.add_argument('--chroma_dir', type=str, default=None,
                        help='Chroma 数据库目录 (默认: output/chroma_db)')
    parser.add_argument('--min_chunk_size', type=int, default=30,
                        help='跳过少于该字符的短 chunk (默认: 30)')
    parser.add_argument('--max_questions', type=int, default=None,
                        help='最大测试问题数 (默认: 全部)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从已有结果文件继续')
    parser.add_argument('--tag', type=str, default=None,
                        help='实验标签，附加到结果目录名 (默认: 根据参数自动生成)')
    parser.add_argument('--use_bm25', action='store_true',
                        help='启用 BM25 多路召回与向量检索融合 (默认: 仅向量检索)')
    parser.add_argument('--rrf_weight', type=float, default=0.5,
                        help='RRF 融合中向量检索的权重 (0=仅BM25, 0.5=均衡, 1=仅向量, 默认: 0.5)')
    parser.add_argument('--rrf_k', type=int, default=60,
                        help='RRF 融合常数 (默认: 60)')
    parser.add_argument('--use_reranker', action='store_true',
                        help='启用 Cross-encoder Reranker 重排 (默认: 关闭)')
    parser.add_argument('--reranker_model', type=str,
                        default='cross-encoder/ms-marco-MiniLM-L-6-v2',
                        help='Reranker 模型名称 (默认: ms-marco-MiniLM-L-6-v2)')
    parser.add_argument('--reranker_candidates', type=int, default=15,
                        help='Reranker 的候选数 (默认: 15)')
    return parser.parse_args()


# ====== 初始化 ======

def init_embeddings():
    """初始化 Embedding 模型"""
    import os
    print("正在加载 Embedding 模型...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu', 'local_files_only': True},
        encode_kwargs={'normalize_embeddings': True}
    )


def init_chroma(embeddings, chroma_dir=None):
    """加载 Chroma 数据库"""
    if chroma_dir is None:
        chroma_dir = CHROMA_DIR
    print(f"正在加载 Chroma 数据库: {chroma_dir}")
    db = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
    )
    count = db._collection.count()
    print(f"  ✓ 总文档数: {count}")
    return db


def init_deepseek():
    """初始化 DeepSeek 客户端"""
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    return client


# ====== 检索（向量） ======

def retrieve_top_k(db, query: str, k: int = 10, min_chunk_size: int = 30):
    """检索 top-k chunks，返回 (docs, elapsed_seconds)
    跳过过短的 chunks（通常是页码、格式残留等噪声）"""
    t0 = time.perf_counter()
    results = db.similarity_search_with_score(query, k=k * 3)
    docs = []
    for doc, score in results:
        content = doc.page_content.strip()
        if len(content) < min_chunk_size:
            continue
        docs.append({
            'content': content,
            'source': doc.metadata.get('source', 'unknown'),
            'chunk_index': doc.metadata.get('chunk_index', -1),
            'score': float(score),
        })
        if len(docs) >= k:
            break
    elapsed = time.perf_counter() - t0
    return docs, elapsed


# ====== 检索（混合：向量 + BM25 多路召回） ======

STOP_WORDS = {
    'a', 'an', 'the', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they',
    'this', 'that', 'these', 'those',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'and', 'or', 'but', 'not', 'no',
    'do', 'does', 'did', 'have', 'has', 'had', 'can', 'could', 'will', 'would',
    'shall', 'should', 'may', 'might', 'must',
    'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
    'about', 'into', 'over', 'after', 'before', 'between', 'under',
    'up', 'out', 'off', 'down', 'just', 'also', 'very', 'too',
    'get', 'got', 'tell', 'told', 'let', 'use', 'used', 'using',
}


def _tokenize(text: str) -> list:
    """简单分词：按非字母数字字符切分，转小写，过滤停用词"""
    import re
    return [w for w in re.findall(r'[a-zA-Z0-9]+', text.lower()) if w not in STOP_WORDS and len(w) > 1]


def build_bm25_index(db):
    """从 Chroma 集合加载所有文档，构建 BM25 索引
    
    返回: (bm25, corpus_data)
        bm25: BM25Okapi 实例
        corpus_data: [{content, source, chunk_index}] 列表，与 BM25 索引顺序一致
    """
    from rank_bm25 import BM25Okapi

    print("正在构建 BM25 索引...")
    t0 = time.perf_counter()

    collection = db._collection
    all_data = collection.get(include=['documents', 'metadatas'])
    
    tokenized_corpus = []
    corpus_data = []
    for i, doc_text in enumerate(all_data['documents']):
        if doc_text is None:
            continue
        content = doc_text.strip()
        if len(content) < 10:
            continue
        tokenized_corpus.append(_tokenize(content))
        meta = all_data['metadatas'][i] or {}
        corpus_data.append({
            'content': content,
            'source': meta.get('source', 'unknown'),
            'chunk_index': meta.get('chunk_index', -1),
        })

    bm25 = BM25Okapi(tokenized_corpus)
    elapsed = time.perf_counter() - t0
    print(f"  ✓ BM25 索引构建完成: {len(corpus_data)} 文档, 耗时 {elapsed*1000:.0f}ms")
    return bm25, corpus_data


def retrieve_hybrid(db, bm25, corpus_data, query: str, k: int = 10,
                    min_chunk_size: int = 30, rrf_weight: float = 0.5,
                    rrf_k: int = 60):
    """混合检索：向量检索 + BM25 检索 → RRF 融合
    
    rrf_weight: 向量检索的权重 (0=仅BM25, 0.5=均衡, 1=仅向量)
    rrf_k: RRF 常数
    
    返回格式同 retrieve_top_k: (docs, elapsed_seconds)
    """
    t0 = time.perf_counter()

    # 1. 向量检索（取 k*2 个候选）
    vector_results = db.similarity_search_with_score(query, k=k * 2)
    vector_docs = []
    for doc, score in vector_results:
        content = doc.page_content.strip()
        if len(content) < min_chunk_size:
            continue
        vector_docs.append({
            'content': content,
            'source': doc.metadata.get('source', 'unknown'),
            'chunk_index': doc.metadata.get('chunk_index', -1),
            'score_dense': float(score),
        })

    # 2. BM25 检索（取 k*2 个候选）
    tokenized_query = _tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    # 按 BM25 分数排序，取 top-k*2 的索引
    top_bm25_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:k * 2]

    bm25_docs = []
    for idx in top_bm25_indices:
        if len(corpus_data[idx]['content']) < min_chunk_size:
            continue
        bm25_docs.append({
            'content': corpus_data[idx]['content'],
            'source': corpus_data[idx]['source'],
            'chunk_index': corpus_data[idx]['chunk_index'],
            'score_sparse': float(bm25_scores[idx]),
        })

    # 3. RRF 融合
    w_vec = rrf_weight      # 向量权重
    w_bm25 = 1.0 - w_vec    # BM25 权重

    # 为每个文档计算 RRF 分数（通过内容匹配去重）
    doc_map = {}  # key: (content, source, chunk_index)
    
    for rank, doc in enumerate(vector_docs):
        key = (doc['content'], doc['source'], doc['chunk_index'])
        if key not in doc_map:
            doc_map[key] = {'ranks': [], 'dense_score': doc['score_dense']}
        doc_map[key]['ranks'].append(('dense', rank))

    for rank, doc in enumerate(bm25_docs):
        key = (doc['content'], doc['source'], doc['chunk_index'])
        if key not in doc_map:
            doc_map[key] = {'ranks': [], 'sparse_score': doc['score_sparse']}
        doc_map[key]['ranks'].append(('sparse', rank))

    # 计算每个文档的 RRF 分数（带权重）
    for key, val in doc_map.items():
        rrf_score = 0.0
        for rtype, rank in val['ranks']:
            if rtype == 'dense':
                rrf_score += w_vec / (rrf_k + rank + 1)
            else:
                rrf_score += w_bm25 / (rrf_k + rank + 1)
        val['rrf_score'] = rrf_score

    # 按 RRF 分数排序
    sorted_keys = sorted(doc_map.keys(), key=lambda k: doc_map[k]['rrf_score'], reverse=True)

    # 4. 组装返回结果
    docs = []
    for key in sorted_keys[:k]:
        content, source, chunk_idx = key
        docs.append({
            'content': content,
            'source': source,
            'chunk_index': chunk_idx,
            'rrf_score': doc_map[key]['rrf_score'],
        })

    elapsed = time.perf_counter() - t0
    return docs, elapsed


# ====== 检索（Reranker 重排） ======

def init_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """初始化 Cross-encoder Reranker 模型"""
    from sentence_transformers import CrossEncoder
    print(f"正在加载 Reranker 模型: {model_name}...")
    t0 = time.perf_counter()
    model = CrossEncoder(model_name, max_length=512)
    elapsed = time.perf_counter() - t0
    print(f"  ✓ Reranker 加载完成, 耗时 {elapsed*1000:.0f}ms")
    return model


def rerank(model, query: str, candidates: list, k: int = 8,
           min_chunk_size: int = 30) -> tuple:
    """对候选 chunks 进行 Cross-encoder 重排
    
    返回: (docs, elapsed_seconds)
    """
    t0 = time.perf_counter()

    # 过滤过短 chunk
    filtered = [c for c in candidates if len(c['content'].strip()) >= min_chunk_size]
    if not filtered:
        filtered = candidates[:k]

    # 准备 query + document 对
    pairs = [(query, doc['content'][:500]) for doc in filtered]

    # Cross-encoder 评分
    scores = model.predict(pairs, show_progress_bar=False)

    # 按 score 排序（Cross-encoder 返回的是相关性分数，越高越好）
    scored = list(zip(filtered, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    # 取 top-k
    docs = []
    for doc, score in scored[:k]:
        docs.append({
            'content': doc['content'],
            'source': doc.get('source', 'unknown'),
            'chunk_index': doc.get('chunk_index', -1),
            'rerank_score': float(score),
        })

    elapsed = time.perf_counter() - t0
    return docs, elapsed


def compute_recall_at_k(retrieved_docs: list, expected_source: str, k: int):
    """
    计算真正的 Recall@k
    分子: top-k 中来自正确来源文档的 chunk 数量
    分母: k (即期望的 top-k 个结果全部来自正确文档)
    等价于 Precision@k / 文档级别的检索纯度

    同时也返回 HitRate@k (正确文档是否出现在 top-k 中)
    """
    hits = sum(1 for d in retrieved_docs[:k] if d['source'] == expected_source)
    recall = hits / k
    hit_rate = 1.0 if hits > 0 else 0.0
    return recall, hit_rate


# ====== 生成 ======

SYSTEM_PROMPT = """You are a helpful research assistant. Your task is to answer questions based on the provided context documents.

Instructions:
1. Answer ONLY using the information in the provided context documents.
2. Cite the source document for each piece of information using the format [来源: document_name.md].
3. If the context partially answers the question, provide what you can find and note any gaps.
4. Only say "I cannot find enough information" if NONE of the documents contain any relevant information.
5. Be concise and accurate.
6. Keep your answer in English."""


def build_prompt(question: str, docs: list) -> str:
    """构建包含上下文和问题的 prompt"""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(
            f"[文档 {i}] 来源: {doc['source']}\n"
            f"内容: {doc['content']}\n"
        )
    context_text = "\n---\n".join(context_parts)

    prompt = f"""Please answer the following question using the provided context documents.

Context documents:
{context_text}

Question: {question}

Answer (with source citations in [来源: filename.md] format):"""
    return prompt


def call_llm(client, model: str, prompt: str, max_retries: int = 3):
    """调用 DeepSeek API"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  ⚠ API 调用失败 ({e}), {wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"  ✗ API 调用失败: {e}")
                return "[ERROR: API call failed]"


# ====== Ragas 评估 ======

def run_ragas_eval(results: list, embeddings):
    """使用 Ragas 计算评估指标"""
    from datasets import Dataset
    from langchain_openai import ChatOpenAI

    data = {
        'user_input': [],
        'response': [],
        'retrieved_contexts': [],
        'reference': [],
    }

    for r in results:
        if r.get('generated_answer', '').startswith('[ERROR'):
            continue
        data['user_input'].append(r['question'])
        data['response'].append(r['generated_answer'])
        data['retrieved_contexts'].append([d['content'] for d in r['retrieved_docs'][:10]])
        data['reference'].append(r['expected_answer'])

    if len(data['user_input']) == 0:
        print("  ⚠ 没有可评估的结果")
        return {}

    dataset = Dataset.from_dict(data)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]

    print(f"\n正在运行 Ragas 评估 ({len(data['user_input'])} 个有效样本)...")
    print(f"  (使用 DeepSeek 作为评估 LLM)")

    evaluator_llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0,
        n=1,  # DeepSeek only supports n=1
    )

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=embeddings,
        )

        print("\n=== Ragas 评估结果 ===")
        df = result.to_pandas()
        scores = {}
        for col in df.columns:
            if col in ['user_input', 'response', 'retrieved_contexts', 'reference']:
                continue
            mean_val = df[col].mean()
            print(f"  {col}: {mean_val:.4f}")
            scores[col] = float(mean_val)

        return scores
    except Exception as e:
        print(f"  ⚠ Ragas 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ====== 主流程 ======

def main():
    args = parse_args()

    # 加载问题集
    with open(args.questions, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data['questions']
    if args.max_questions:
        questions = questions[:args.max_questions]

    print(f"加载了 {len(questions)} 个测试问题")
    print(f"检索 top-k: {args.top_k}")
    print(f"最小 chunk 长度: {args.min_chunk_size}")
    print(f"LLM 模型: {args.model}")
    print(f"检索方式: {'向量 + BM25 混合' if args.use_bm25 else '纯向量'}"
          f"{f' (RRF权重: 向量={args.rrf_weight}, BM25={1-args.rrf_weight:.1f})' if args.use_bm25 else ''}"
          f"{' + Reranker' if args.use_reranker else ''}")
    print("=" * 70)

    # 加载已有结果（断点续跑）
    existing_results = {}
    if args.resume and os.path.exists(args.resume):
        with open(args.resume, 'r', encoding='utf-8') as f:
            existing_results_list = json.load(f)
        for r in existing_results_list:
            existing_results[r['id']] = r
        print(f"从 {args.resume} 加载了 {len(existing_results)} 个已有结果")

    # 初始化
    embeddings = init_embeddings()
    db = init_chroma(embeddings, args.chroma_dir)
    client = init_deepseek()
    model_name = args.model

    # BM25 混合检索初始化
    bm25 = None
    corpus_data = None
    if args.use_bm25:
        bm25, corpus_data = build_bm25_index(db)
        print(f"  ✓ BM25 多路召回已启用")

    # Reranker 初始化
    reranker_model = None
    if args.use_reranker:
        reranker_model = init_reranker(args.reranker_model)
        print(f"  ✓ Reranker 重排已启用")

    # 自动产出隔离：根据评估参数生成结果目录
    eval_parts = []
    if args.top_k != 10:
        eval_parts.append(f'tk{args.top_k}')
    if args.chroma_dir:
        db_name = os.path.basename(args.chroma_dir)
        if db_name != 'chroma_db':
            eval_parts.append(db_name.replace('chroma_db_', ''))
    if args.use_bm25:
        eval_parts.append('bm25')
        if args.rrf_weight != 0.5:
            eval_parts.append(f'rrf{args.rrf_weight}')
    if args.use_reranker:
        eval_parts.append('reranker')
        if args.reranker_model != 'cross-encoder/ms-marco-MiniLM-L-6-v2':
            model_short = args.reranker_model.split('/')[-1]
            eval_parts.append(model_short)
    if args.tag:
        eval_parts.append(args.tag)

    eval_results_dir = RESULTS_DIR
    if eval_parts:
        eval_results_dir = RESULTS_DIR + f'_{"_".join(eval_parts)}'

    os.makedirs(eval_results_dir, exist_ok=True)
    print(f"评估结果目录: {eval_results_dir}")

    # 逐问题评估
    all_results = list(existing_results.values())
    recall_at_10_scores = []
    retrieval_times = []  # 每题的检索耗时（秒）

    for i, q in enumerate(questions):
        qid = q['id']
        if qid in existing_results:
            print(f"[{i+1}/{len(questions)}] 问题 {qid}: 跳过（已有结果）")
            recall_at_10_scores.append(existing_results[qid].get('recall_at_k', 
                                          existing_results[qid].get('recall_at_10', 0)))
            continue

        print(f"[{i+1}/{len(questions)}] 问题 {qid}: {q['question'][:60]}...")

        # 1. 检索
        candidate_k = args.reranker_candidates if args.use_reranker else args.top_k

        if bm25 is not None:
            docs, retrieval_time = retrieve_hybrid(
                db, bm25, corpus_data, q['question'],
                k=candidate_k, min_chunk_size=args.min_chunk_size,
                rrf_weight=args.rrf_weight, rrf_k=args.rrf_k,
            )
        else:
            docs, retrieval_time = retrieve_top_k(db, q['question'], k=candidate_k,
                                           min_chunk_size=args.min_chunk_size)

        # 1b. Reranker 重排（若启用）
        if args.use_reranker and len(docs) > args.top_k:
            docs, rerank_time = rerank(
                reranker_model, q['question'], docs,
                k=args.top_k, min_chunk_size=args.min_chunk_size,
            )
            retrieval_time += rerank_time

        retrieval_times.append(retrieval_time)

        # 2. 计算 Recall@10 和 HitRate@10
        recall, hit_rate = compute_recall_at_k(docs, q['source'], k=args.top_k)
        recall_at_10_scores.append(recall)

        # 3. 生成答案
        prompt = build_prompt(q['question'], docs)
        answer = call_llm(client, model_name, prompt)

        # 4. 保存结果
        result = {
            'id': qid,
            'question': q['question'],
            'expected_answer': q['answer'],
            'expected_source': q['source'],
            'generated_answer': answer,
            'recall_at_k': recall,
            'hit_rate_at_k': hit_rate,
            'retrieval_time_s': round(retrieval_time, 4),
            'retrieved_docs': docs[:args.top_k],
            'retrieved_sources': [d['source'] for d in docs[:args.top_k]],
            'type': q.get('type', 'unknown'),
        }
        all_results.append(result)

        print(f"  Recall@k: {recall:.2f} (top-{args.top_k} 中 {int(recall * args.top_k)}/{args.top_k} 来自正确文档)")
        print(f"  HitRate@k: {hit_rate:.2f} (正确文档是否出现)")
        print(f"  ⏱ 检索耗时: {retrieval_time*1000:.1f}ms")
        answer_preview = answer[:80].replace('\n', ' ') if answer else '(空)'
        print(f"  回答: {answer_preview}...")

        # 每 10 个问题保存一次中间结果
        if (i + 1) % 10 == 0:
            mid_path = os.path.join(eval_results_dir, f'intermediate_{i+1}.json')
            with open(mid_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"  💾 中间结果已保存: {mid_path}")

        time.sleep(1)  # API 限流

    # ====== 汇总 ======

    # 保存完整结果
    results_path = os.path.join(eval_results_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 完整结果已保存: {results_path}")

    # ====== 检索耗时统计 ======
    retrieval_times_ms = [t * 1000 for t in retrieval_times]
    total_retrieval_s = sum(retrieval_times)
    avg_retrieval_ms = np.mean(retrieval_times_ms) if retrieval_times_ms else 0
    max_retrieval_ms = np.max(retrieval_times_ms) if retrieval_times_ms else 0
    min_retrieval_ms = np.min(retrieval_times_ms) if retrieval_times_ms else 0

    print("\n" + "=" * 70)
    print("⏱ 检索耗时统计")
    print("=" * 70)
    print(f"  总检索耗时: {total_retrieval_s:.2f}s")
    print(f"  平均每次检索: {avg_retrieval_ms:.1f}ms")
    print(f"  最快: {min_retrieval_ms:.1f}ms")
    print(f"  最慢: {max_retrieval_ms:.1f}ms")

    # Recall@k 统计
    recall_scores = [r['recall_at_k'] for r in all_results]
    hit_rate_scores = [r.get('hit_rate_at_k', 0) for r in all_results]
    avg_recall = np.mean(recall_scores)
    avg_hit_rate = np.mean(hit_rate_scores)

    # 按文档类型统计 Recall
    print("\n" + "=" * 70)
    print(f"📊 Recall@{args.top_k} / HitRate@{args.top_k} 总体统计")
    print("=" * 70)
    print(f"  平均 Recall@{args.top_k} (top-{args.top_k} 纯度): {avg_recall:.4f}")
    print(f"  平均 HitRate@{args.top_k} (文档命中率): {avg_hit_rate:.4f}")
    print(f"  中位数 Recall@{args.top_k}: {np.median(recall_scores):.4f}")

    # 按文档统计
    doc_recalls = {}
    for r in all_results:
        src = r['expected_source']
        if src not in doc_recalls:
            doc_recalls[src] = []
        doc_recalls[src].append(r['recall_at_k'])

    print(f"\n按文档统计 Recall@{args.top_k}:")
    for src, scores in sorted(doc_recalls.items()):
        print(f"  {src}: {np.mean(scores):.4f} ({len(scores)} 题)")

    # 按类型统计
    type_recalls = {}
    for r in all_results:
        t = r['type']
        if t not in type_recalls:
            type_recalls[t] = []
        type_recalls[t].append(r['recall_at_k'])

    print(f"\n按问题类型统计 Recall@{args.top_k}:")
    for t, scores in sorted(type_recalls.items()):
        print(f"  {t}: {np.mean(scores):.4f} ({len(scores)} 题)")

    # ====== Ragas 评估 ======
    print("\n" + "=" * 70)
    print("🧪 Ragas 评估")
    print("=" * 70)
    try:
        ragas_scores = run_ragas_eval(all_results, embeddings)
    except Exception as e:
        print(f"  ⚠ Ragas 评估失败: {e}")
        ragas_scores = {}

    # ====== 最终报告 ======
    report = {
        'config': {
            'embedding_model': EMBEDDING_MODEL,
            'llm_model': args.model,
            'top_k': args.top_k,
            'total_questions': len(all_results),
            'retrieval': 'vector+bm25' if args.use_bm25 else 'vector_only',
            'rrf_weight': args.rrf_weight if args.use_bm25 else None,
            'reranker': args.reranker_model if args.use_reranker else None,
            'reranker_candidates': args.reranker_candidates if args.use_reranker else None,
        },
        f'recall_at_{args.top_k}': {
            'average': float(avg_recall),
            'median': float(np.median(recall_scores)),
            f'hit_rate_at_{args.top_k}': float(avg_hit_rate),
            'note': f'recall@{args.top_k} = top-{args.top_k}中来自正确文档的chunk比例; hit_rate = 正确文档是否出现在top-{args.top_k}中',
            'by_source': {src: float(np.mean(scores)) for src, scores in doc_recalls.items()},
            'by_type': {t: float(np.mean(scores)) for t, scores in type_recalls.items()},
        },
        'retrieval_timing': {
            'total_s': round(total_retrieval_s, 3),
            'avg_ms': round(avg_retrieval_ms, 2),
            'min_ms': round(min_retrieval_ms, 2),
            'max_ms': round(max_retrieval_ms, 2),
            'per_question_ms': {r['id']: round(r['retrieval_time_s'] * 1000, 2) for r in all_results},
        },
        'ragas': {k: float(v) for k, v in ragas_scores.items()},
    }

    report_path = os.path.join(eval_results_dir, 'report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n💾 评估报告已保存: {report_path}")
    print("\n✅ 评估完成！")


if __name__ == '__main__':
    main()
