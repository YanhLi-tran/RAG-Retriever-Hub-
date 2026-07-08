"""
Markdown 文本切分脚本
用途：将清理后的 Markdown 文件切分为 chunks，用于后续向量化存储
默认参数：chunk_size=500, chunk_overlap=50

用法：
  python text_splitter.py --chunk_size 600 --chunk_overlap 60
  python text_splitter.py --input_dir ../../data_output/html_markdown --output_dir ../../data_output/chunks_html --chunk_size 300 --chunk_overlap 45
"""
import argparse
import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.stdout.reconfigure(encoding='utf-8')

# 默认配置路径
DEFAULT_INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data_output', 'pdf_markdown')
DEFAULT_INPUT_DIR = os.path.normpath(DEFAULT_INPUT_DIR)

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data_output', 'chunks')
DEFAULT_OUTPUT_DIR = os.path.normpath(DEFAULT_OUTPUT_DIR)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Markdown 文本切分工具')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR,
                        help='输入目录，包含 .md 文件 (默认: data_output/pdf_markdown)')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='输出目录 (默认: data_output/chunks)')
    parser.add_argument('--chunk_size', type=int, default=500,
                        help='每个 chunk 的最大字符数 (默认: 500)')
    parser.add_argument('--chunk_overlap', type=int, default=50,
                        help='相邻 chunk 的重叠字符数 (默认: 50)')
    return parser.parse_args()


def split_markdown(text: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    将 Markdown 文本切分为 chunks
    
    使用 RecursiveCharacterTextSplitter，按以下优先级分隔：
    1. 段落分隔 (\n\n)
    2. 换行 (\n)
    3. 句号 (. )
    4. 空格
    5. 字符
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = text_splitter.split_text(text)
    return chunks


def process_file(input_path: str, output_path: str, chunk_size: int, 
                 chunk_overlap: int) -> dict:
    """处理单个文件，返回统计信息"""
    fname = os.path.basename(input_path)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = split_markdown(content, chunk_size, chunk_overlap)
    
    # 保存 chunks 到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"=== Chunk {i+1} ===\n")
            f.write(chunk)
            f.write("\n\n")
    
    return {
        'file': fname,
        'original_chars': len(content),
        'chunk_count': len(chunks),
        'avg_chunk_size': sum(len(c) for c in chunks) // len(chunks) if chunks else 0,
    }


def process_all(input_dir: str, output_dir: str, chunk_size: int, chunk_overlap: int):
    """处理所有 Markdown 文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    md_files = sorted([
        f for f in os.listdir(input_dir) if f.endswith('.md')
    ])
    
    print(f"开始切分 {len(md_files)} 个 Markdown 文件...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"切分参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print("=" * 70)
    
    total_chunks = 0
    for md_file in md_files:
        input_path = os.path.join(input_dir, md_file)
        output_path = os.path.join(output_dir, md_file.replace('.md', '_chunks.txt'))
        
        stats = process_file(input_path, output_path, chunk_size, chunk_overlap)
        total_chunks += stats['chunk_count']
        
        print(f"  {stats['file']}:")
        print(f"    原始字符: {stats['original_chars']:,}")
        print(f"    切分数: {stats['chunk_count']}")
        print(f"    平均 chunk 大小: {stats['avg_chunk_size']}")
    
    print("=" * 70)
    print(f"切分完成！总 chunks 数: {total_chunks}")
    print(f"文件位于: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    print("=" * 70)
    print("Markdown 文本切分")
    print("=" * 70)
    process_all(args.input_dir, args.output_dir, args.chunk_size, args.chunk_overlap)
