"""
RAG 交互式问答 Demo
用途：在终端中输入问题，实时展示检索结果和 LLM 生成答案（带来源溯源）

用法：
  python scripts/demo_rag.py
  python scripts/demo_rag.py --chroma_dir output/chroma_db_combined --top_k 5 --use_bm25
"""
import argparse
import os
import re
import sys
import time
import types
import warnings
from unittest.mock import MagicMock

from dotenv import load_dotenv
from openai import OpenAI

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Ragas 兼容补丁
_vtx_mod = types.ModuleType('langchain_community.chat_models.vertexai')
_vtx_mod.ChatVertexAI = MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = _vtx_mod

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

EMBEDDING_MODEL = "BAAI/bge-large-zh"
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYSTEM_PROMPT = """You are a helpful research assistant. Answer the question based ONLY on the provided context documents.

Rules:
1. Cite the source for each claim using [来源: filename.md]
2. If the context doesn't contain enough information, say so honestly
3. Be concise and accurate
4. Keep your answer in English"""


def parse_args():
    parser = argparse.ArgumentParser(description='RAG 交互式问答 Demo')
    parser.add_argument('--chroma_dir', type=str, default='output/chroma_db_combined',
                        help='Chroma 数据库目录')
    parser.add_argument('--top_k', type=int, default=5,
                        help='检索返回的 chunk 数')
    parser.add_argument('--use_bm25', action='store_true', default=True,
                        help='启用 BM25 多路召回')
    parser.add_argument('--no_bm25', action='store_true',
                        help='关闭 BM25')
    return parser.parse_args()


def _tokenize(text: str) -> list:
    return re.findall(r'[a-zA-Z0-9]+', text.lower())


def build_bm25(db):
    """构建 BM25 索引"""
    print("  构建 BM25 索引...")
    collection = db._collection
    all_data = collection.get(include=['documents', 'metadatas'])
    tok_docs = []
    corpus = []
    for i, doc_text in enumerate(all_data['documents']):
        if doc_text is None:
            continue
        content = doc_text.strip()
        if len(content) < 10:
            continue
        tok_docs.append(_tokenize(content))
        meta = all_data['metadatas'][i] or {}
        corpus.append({
            'content': content,
            'source': meta.get('source', 'unknown'),
        })
    return BM25Okapi(tok_docs), corpus


def retrieve_top_k(db, bm25, corpus_data, query: str, k: int, use_bm25: bool):
    """检索并返回带来源的 chunks"""
    t0 = time.perf_counter()

    # 向量检索
    vec_results = db.similarity_search_with_score(query, k=k * 3)
    vec_docs = []
    for doc, score in vec_results:
        content = doc.page_content.strip()
        if len(content) < 30:
            continue
        vec_docs.append({
            'content': content,
            'source': doc.metadata.get('source', 'unknown'),
            'score': float(score),
        })

    if use_bm25 and bm25 is not None:
        # BM25 检索
        bm25_scores = bm25.get_scores(_tokenize(query))
        top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k * 2]
        bm25_docs = []
        for idx in top_idx:
            if len(corpus_data[idx]['content']) < 30:
                continue
            bm25_docs.append({
                'content': corpus_data[idx]['content'],
                'source': corpus_data[idx]['source'],
                'score': float(bm25_scores[idx]),
            })

        # RRF 融合
        doc_map = {}
        for rank, doc in enumerate(vec_docs):
            key = doc['content'][:100]
            if key not in doc_map:
                doc_map[key] = doc.copy()
                doc_map[key]['rrf'] = 0
            doc_map[key]['rrf'] += 1.0 / (60 + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            key = doc['content'][:100]
            if key not in doc_map:
                doc_map[key] = doc.copy()
                doc_map[key]['rrf'] = 0
            doc_map[key]['rrf'] += 1.0 / (60 + rank + 1)

        all_docs = sorted(doc_map.values(), key=lambda x: x['rrf'], reverse=True)
    else:
        all_docs = vec_docs

    elapsed = time.perf_counter() - t0
    return all_docs[:k], elapsed


def call_llm(question: str, docs: list):
    """调用 DeepSeek 生成回答"""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[文档 {i}] 来源: {doc['source']}\n内容: {doc['content']}")
    context_text = "\n---\n".join(context_parts)

    prompt = f"""Please answer the following question using the provided context documents.

Context documents:
{context_text}

Question: {question}

Answer (with source citations in [来源: filename.md] format):"""

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    llm_time = time.perf_counter() - t0
    return resp.choices[0].message.content.strip(), llm_time


def print_result(question: str, answer: str, docs: list, retrieve_time: float, llm_time: float):
    """格式化输出结果"""
    width = 70
    print()
    print("=" * width)
    print(f"  Q: {question}")
    print("=" * width)
    print(f"\n  A: {answer}")
    print()
    print("-" * width)
    print("  溯源 (检索到的文档):")
    for i, doc in enumerate(docs, 1):
        src = doc['source']
        score = doc.get('rrf', doc.get('score', 0))
        content_preview = doc['content'][:80].replace('\n', ' ')
        print(f"  [{i}] {src} (score: {score:.3f})")
        print(f"      {content_preview}...")
    print()
    print("-" * width)
    print(f"  ⏱ 检索: {retrieve_time*1000:.0f}ms  |  生成: {llm_time*1000:.0f}ms  |  总计: {(retrieve_time+llm_time)*1000:.0f}ms")
    print("=" * width)


def main():
    args = parse_args()
    use_bm25 = args.use_bm25 and not args.no_bm25

    chroma_path = os.path.join(PROJECT_ROOT, args.chroma_dir)
    print(f"加载 Chroma 数据库: {chroma_path}")

    print("加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    count = db._collection.count()
    print(f"  ✓ 总文档数: {count}")

    bm25 = None
    corpus_data = None
    if use_bm25:
        bm25, corpus_data = build_bm25(db)
        print("  ✓ BM25 已启用")

    print(f"\n{'='*70}")
    print("  RAG 交互式问答 Demo (输入 'quit' 退出)")
    print(f"{'='*70}")
    print()

    while True:
        try:
            question = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ('quit', 'exit', 'q'):
            break

        docs, retrieve_time = retrieve_top_k(
            db, bm25, corpus_data, question, k=args.top_k, use_bm25=use_bm25
        )

        if not docs:
            print("  ⚠ 未检索到相关文档")
            continue

        answer, llm_time = call_llm(question, docs)
        print_result(question, answer, docs, retrieve_time, llm_time)


if __name__ == '__main__':
    main()
