"""
CSV 解析脚本
用途：将 CSV 表格转换为自然语言 Markdown，便于检索
处理：
  1. 读取 CSV 文件
  2. 将每行数据转为可读的自然语言描述
  3. 输出 Markdown 格式

用法：
  python parse_csv.py
"""
import argparse
import csv
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 数据源配置
SOURCE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'RAG_Data', 'tables')
SOURCE_DIR = os.path.normpath(SOURCE_DIR)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data_output', 'csv_markdown')
OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description='CSV 表格转 Markdown 工具')
    parser.add_argument('--input_dir', type=str, default=SOURCE_DIR,
                        help='输入目录，包含 .csv 文件')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='输出目录 (默认: data_output/csv_markdown)')
    return parser.parse_args()


def csv_to_markdown_table(csv_path: str) -> str:
    """将 CSV 文件转为 Markdown 表格"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return ""

    # 表头
    headers = rows[0]
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    # 数据行
    for row in rows[1:]:
        # 补齐长度
        while len(row) < len(headers):
            row.append("")
        md += "| " + " | ".join(row) + " |\n"

    return md


def csv_to_natural_text(csv_path: str) -> str:
    """将 CSV 每行转为自然语言描述"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return ""

    headers = list(rows[0].keys())
    file_name = os.path.splitext(os.path.basename(csv_path))[0]
    lines = []
    lines.append(f"# {file_name}")
    lines.append("")

    for i, row in enumerate(rows, 1):
        parts = []
        for h in headers:
            val = row[h].strip()
            if val:
                parts.append(f"{h}: {val}")
        lines.append(f"- {'; '.join(parts)}")

    return "\n".join(lines)


def process_file(csv_path: str, output_path: str) -> dict:
    """处理单个 CSV 文件"""
    fname = os.path.basename(csv_path)
    original_size = os.path.getsize(csv_path)

    # 同时输出 Markdown 表格和自然语言描述
    md_table = csv_to_markdown_table(csv_path)
    natural_text = csv_to_natural_text(csv_path)

    # 合并输出：先表格后自然语言描述
    content = f"<!-- 来源: {fname} -->\n\n"
    content += md_table + "\n\n"
    content += natural_text

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return {
        'file': fname,
        'original_size': original_size,
        'output_chars': len(content),
    }


def process_all(input_dir: str, output_dir: str):
    """处理所有 CSV 文件"""
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted([
        f for f in os.listdir(input_dir) if f.endswith('.csv')
    ])

    print(f"开始处理 {len(csv_files)} 个 CSV 文件...")
    print(f"输出目录: {output_dir}")
    print("=" * 70)

    for csv_file in csv_files:
        csv_path = os.path.join(input_dir, csv_file)
        output_name = csv_file.replace('.csv', '.md')
        output_path = os.path.join(output_dir, output_name)

        stats = process_file(csv_path, output_path)
        print(f"  {stats['file']}: {stats['original_size']:,}B -> {stats['output_chars']:,} chars")

    print("=" * 70)
    print(f"处理完成！文件位于: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    print("=" * 70)
    print("CSV 表格转 Markdown")
    print("=" * 70)
    process_all(args.input_dir, args.output_dir)
