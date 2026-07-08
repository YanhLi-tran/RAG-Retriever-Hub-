"""
PDF 解析 + 清理一体化脚本
用途：使用 pymupdf4llm 将 PDF 转为 Markdown，并自动清理噪声
优势：不需要 Tesseract，自带 RapidOCR，离线处理用时间换精度
处理：
  1. 图表OCR文字（Figure/Table 说明混入正文）
  2. 脚注链接（参考文献、URL）
  3. 多余空行
"""
import os
import re
import sys

import pymupdf4llm

sys.stdout.reconfigure(encoding='utf-8')

# 配置路径
PDF_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'RAG_Data', 'papers')
PDF_DIR = os.path.normpath(PDF_DIR)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data_output', 'pdf_markdown')
OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)


# ===== 清理逻辑 =====

def clean_markdown(text: str) -> str:
    """清理 Markdown 文本中的噪声"""
    lines = text.split('\n')
    cleaned_lines = []

    # 状态标记
    in_reference_section = False  # 是否在参考文献区域

    for i, line in enumerate(lines):
        stripped = line.strip()

        # === 0. 跳过 PDF 输出注释头 ===
        if stripped.startswith('<!-- PDF') or stripped.startswith('<!-- 总字符数'):
            continue

        # === 1. 跳过参考文献区域 ===
        # 检测参考文献标题（支持 **References** 等加粗格式）
        if re.match(r'^#+\s*\*{0,2}\s*(References|Bibliography|参考文献)\s*\*{0,2}\s*$',
                    stripped, re.IGNORECASE):
            in_reference_section = True
            continue

        # 在参考文献区域内，跳过所有内容
        if in_reference_section:
            # 遇到新的 ## 或 # 标题时退出参考文献区域（保留 Appendix 等有用内容）
            # 注意：子标题如 ### A Details 不退出，只在遇到 ## 或 # 顶层标题时退出
            heading_match = re.match(r'^(#{1,2})\s+', stripped)
            if heading_match:
                in_reference_section = False
            else:
                continue

        # === 2. 跳过图表说明文字 ===
        # 匹配 Figure X: / Fig. X: / Table X: 开头的独立说明行
        if re.match(r'^(Figure|Fig\.?|Table|图|表)\s*\d+[：:．.]', stripped):
            continue

        # === 3. 跳过脚注中的 URL ===
        # 纯 URL 行
        if re.match(r'^https?://\S+$', stripped):
            continue

        # 脚注格式：> 数字 + URL 或者 > 包含 URL 的说明
        if stripped.startswith('>') and ('http' in stripped or 'www.' in stripped):
            continue

        # === 4. 跳过引用标记行 ===
        if re.match(r'^\s*-\s*\[\d+\]', stripped):
            continue

        # === 5. 清理行内噪声 ===
        # 移除行内脚注标记 [数字]（但保留 [a-z] 格式的引用）
        line = re.sub(r'\[\d+\]', '', line)
        # 移除残留的脚注标记如 ¹²³
        line = re.sub(r'[¹²³⁴⁵⁶⁷⁸⁹⁰]+', '', line)

        # === 6. 压缩多余空行 ===
        if stripped == '':
            if cleaned_lines and cleaned_lines[-1].strip() == '':
                continue

        cleaned_lines.append(line)

    # 移除开头和结尾的空行
    while cleaned_lines and cleaned_lines[0].strip() == '':
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1].strip() == '':
        cleaned_lines.pop()

    return '\n'.join(cleaned_lines)


# ===== 解析逻辑 =====

def export_pdf_to_cleaned_markdown(pdf_path: str, output_path: str):
    """解析单个 PDF，清理噪声后导出为 Markdown 文件"""
    fname = os.path.basename(pdf_path)
    print(f"\n解析: {fname}")

    # to_markdown 会自动调用 RapidOCR 处理扫描页
    raw_text = pymupdf4llm.to_markdown(pdf_path)
    print(f"  ✓ 原始字符数: {len(raw_text):,}")

    # 清理噪声
    cleaned_text = clean_markdown(raw_text)
    removed = len(raw_text) - len(cleaned_text)
    print(f"  ✓ 清理后字符数: {len(cleaned_text):,} (减少 {removed:,})")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    print(f"  ✓ 输出: {os.path.basename(output_path)}")


def process_all_pdfs():
    """解析所有 PDF 并输出清理后的 Markdown"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = sorted([
        f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')
    ])

    print(f"开始解析 {len(pdf_files)} 个 PDF 文件...")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 80)

    success_count = 0
    fail_count = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        output_path = os.path.join(OUTPUT_DIR, pdf_file.replace('.pdf', '.md'))
        try:
            export_pdf_to_cleaned_markdown(pdf_path, output_path)
            success_count += 1
        except Exception as e:
            print(f"  ✗ 解析失败: {e}")
            fail_count += 1

    print("=" * 80)
    print(f"处理完成！成功: {success_count}, 失败: {fail_count}")
    print(f"文件位于: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("=" * 80)
    print("PDF 解析 + 清理一体化 (pymupdf4llm -> clean Markdown)")
    print("=" * 80)
    print(f"PDF 目录: {PDF_DIR}")

    process_all_pdfs()
