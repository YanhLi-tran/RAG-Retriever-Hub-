"""
DeepSeek API Query Rewriting & Multi-Query 实验
用途：使用 DeepSeek API 进行查询改写和多查询生成，验证检索增强效果
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, 'data_output', 'experiments')
TOP_K = 8

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def call_deepseek(prompt: str, temperature: float = 0.3, max_tokens: int = 256):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"[ERROR: {e}]"


REWRITE_PROMPT = """You are a query rewriter. Rewrite the following question to be more precise, self-contained, and search-friendly for a RAG retrieval system.

Rules:
1. Expand abbreviations and acronyms
2. Add context from related concepts if the question is vague
3. Keep the original intent
4. Output ONLY the rewritten question, nothing else.

Original question: {question}
Rewritten question:"""

MULTI_QUERY_PROMPT = """Generate 3 different variations of the following question. Each variation should rephrase the question using different wording or perspective, but keep the same intent.

Output EXACTLY 3 lines, each line is one variation. No numbering, no extra text.

Original question: {question}
Variations:"""


def load_all_questions():
    all_q = []
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
                all_q.append(q)
    return all_q


def run_rewrite():
    print("\n" + "=" * 70)
    print("实验 1: Query Rewriting — DeepSeek API")
    print("=" * 70)

    questions = load_all_questions()
    print(f"加载了 {len(questions)} 道测试题")

    results = []
    rewrite_times = []
    success = 0
    fail = 0

    for i, q in enumerate(questions):
        t0 = time.perf_counter()
        rewritten = call_deepseek(REWRITE_PROMPT.format(question=q['question']), temperature=0.3)
        elapsed = time.perf_counter() - t0
        rewrite_times.append(elapsed * 1000)

        if rewritten and not rewritten.startswith('[ERROR'):
            success += 1
        else:
            fail += 1
            rewritten = q['question']

        results.append({
            'id': q['id'],
            'question': q['question'],
            'rewritten': rewritten,
            'rewrite_time_ms': round(elapsed * 1000, 1),
        })

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(questions)} 题")

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    out = os.path.join(EXPERIMENT_DIR, 'deepseek_rewrite.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'query_rewriting',
            'model': 'DeepSeek Chat',
            'total': len(results),
            'success': success,
            'fail': fail,
            'avg_rewrite_time_ms': sum(rewrite_times) / len(rewrite_times),
            'results': results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n成功: {success}, 失败: {fail}, 平均耗时: {sum(rewrite_times)/len(rewrite_times):.0f}ms")
    print(f"💾 已保存: {out}")
    return results


def run_multi_query():
    print("\n" + "=" * 70)
    print("实验 2: Multi-Query ×3 — DeepSeek API")
    print("=" * 70)

    questions = load_all_questions()
    print(f"加载了 {len(questions)} 道测试题")

    results = []
    gen_times = []
    success = 0
    fail = 0

    for i, q in enumerate(questions):
        t0 = time.perf_counter()
        raw = call_deepseek(MULTI_QUERY_PROMPT.format(question=q['question']), temperature=0.7)
        elapsed = time.perf_counter() - t0
        gen_times.append(elapsed * 1000)

        if raw and not raw.startswith('[ERROR'):
            lines = [l.strip().lstrip('-').lstrip('0123456789.').strip() for l in raw.split('\n') if l.strip()]
            lines = [l for l in lines if len(l) > 10]
            if len(lines) >= 2:
                variants = lines[:3]
                success += 1
            else:
                variants = [q['question'], q['question'], q['question']]
                fail += 1
        else:
            variants = [q['question'], q['question'], q['question']]
            fail += 1

        results.append({
            'id': q['id'],
            'question': q['question'],
            'variations': variants,
            'gen_time_ms': round(elapsed * 1000, 1),
        })

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(questions)} 题")

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    out = os.path.join(EXPERIMENT_DIR, 'deepseek_multi_query.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'multi_query_x3',
            'model': 'DeepSeek Chat',
            'total': len(results),
            'success': success,
            'fail': fail,
            'avg_generation_time_ms': sum(gen_times) / len(gen_times),
            'results': results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n成功: {success}, 失败: {fail}, 平均生成耗时: {sum(gen_times)/len(gen_times):.0f}ms")
    print(f"💾 已保存: {out}")
    return results


if __name__ == '__main__':
    exp = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if exp in ['rewrite', 'all']:
        run_rewrite()
    if exp in ['multi_query', 'all']:
        run_multi_query()
    print("\n✅ DeepSeek 实验完成！")
