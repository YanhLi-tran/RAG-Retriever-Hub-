"""
HTML 解析脚本
用途：使用 trafilatura 将 HTML 文档提取为干净 Markdown
处理：
  1. 自动去除导航栏、侧栏、页脚等 boilerplate
  2. 保留代码块、表格、链接结构
  3. 输出 Markdown 格式，与 PDF 管线一致

用法：
  python parse_html.py
  python parse_html.py --output_dir ../../data_output/html_markdown
"""
import argparse
import os
import sys
import trafilatura

sys.stdout.reconfigure(encoding='utf-8')

# 数据源配置
SOURCE_DIRS = {
    'langchain': os.path.join(os.path.dirname(__file__), '..', '..', 'RAG_Data', 'langchain_docs'),
    'llamaindex': os.path.join(os.path.dirname(__file__), '..', '..', 'RAG_Data', 'llamaindex_docs'),
    'web': os.path.join(os.path.dirname(__file__), '..', '..', 'RAG_Data', 'web'),
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data_output', 'html_markdown')
OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description='HTML 文档解析工具（trafilatura）')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='输出目录 (默认: data_output/html_markdown)')
    parser.add_argument('--include_links', action='store_true', default=True,
                        help='保留链接 (默认: True)')
    parser.add_argument('--include_images', action='store_true', default=False,
                        help='保留图片引用 (默认: False)')
    return parser.parse_args()


def extract_html_to_markdown(html_path: str, include_links: bool = True,
                             include_images: bool = False) -> str:
    """使用 trafilatura 将 HTML 提取为干净 Markdown"""
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()

    md_text = trafilatura.extract(
        html_content,
        output_format='markdown',
        include_links=include_links,
        include_images=include_images,
        include_tables=True,
        include_comments=False,
        no_fallback=False,
    )

    if md_text is None:
        return ""

    return md_text.strip()


def process_html_file(html_path: str, output_path: str, include_links: bool,
                      include_images: bool) -> dict:
    """处理单个 HTML 文件，返回统计信息"""
    fname = os.path.basename(html_path)
    original_size = os.path.getsize(html_path)

    md_text = extract_html_to_markdown(html_path, include_links, include_images)

    if not md_text:
        print(f"  ⚠ {fname}: 提取结果为空，跳过")
        return {'file': fname, 'original_size': original_size, 'output_chars': 0, 'status': 'skipped'}

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_text)

    return {
        'file': fname,
        'original_size': original_size,
        'output_chars': len(md_text),
        'status': 'ok',
    }


def process_all(output_dir: str, include_links: bool, include_images: bool):
    """处理所有 HTML 文件"""
    os.makedirs(output_dir, exist_ok=True)

    total_files = 0
    total_ok = 0
    total_skipped = 0

    print(f"开始解析 HTML 文档...")
    print(f"输出目录: {output_dir}")
    print("=" * 70)

    for category, dir_path in SOURCE_DIRS.items():
        if not os.path.exists(dir_path):
            print(f"\n  ⚠ 目录不存在: {dir_path}")
            continue

        html_files = sorted([
            f for f in os.listdir(dir_path)
            if f.endswith(('.html', '.htm'))
        ])

        if not html_files:
            print(f"\n  [{category}] 没有找到 HTML 文件")
            continue

        print(f"\n  [{category}] {len(html_files)} 个文件:")

        for html_file in html_files:
            html_path = os.path.join(dir_path, html_file)
            # 输出文件名：来源类别_原文件名（去重）
            output_name = f"{category}_{html_file.replace('.html', '.md').replace('.htm', '.md')}"
            output_path = os.path.join(output_dir, output_name)

            stats = process_html_file(html_path, output_path, include_links, include_images)
            total_files += 1

            if stats['status'] == 'ok':
                total_ok += 1
                reduction = (1 - stats['output_chars'] / stats['original_size']) * 100
                print(f"    {output_name}: {stats['original_size']:,}B -> {stats['output_chars']:,} chars "
                      f"(减少 {reduction:.1f}%)")
            else:
                total_skipped += 1

    print("=" * 70)
    print(f"解析完成！成功: {total_ok}, 跳过: {total_skipped}, 总数: {total_files}")
    print(f"文件位于: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    print("=" * 70)
    print("HTML 文档解析 (trafilatura -> Markdown)")
    print("=" * 70)
    process_all(args.output_dir, args.include_links, args.include_images)
