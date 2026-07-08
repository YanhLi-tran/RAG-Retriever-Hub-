"""
Chroma 向量数据库存储脚本
用途：将切分后的 chunks 存入 Chroma 向量数据库，用于后续检索
注意：自动处理中文路径问题（使用临时目录构建后复制）

用法：
  python chroma_store.py --chunk_size 600 --chunk_overlap 60 --model_name BAAI/bge-large-zh
"""
import argparse
import os
import sys
import shutil
import tempfile
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.stdout.reconfigure(encoding='utf-8')

# 配置路径（相对路径，适配项目结构）
INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data_output', 'pdf_markdown')
INPUT_DIR = os.path.normpath(INPUT_DIR)

# Chroma 数据库存储目录（相对路径）
CHROMA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'chroma_db')
CHROMA_DIR = os.path.normpath(CHROMA_DIR)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Chroma 向量数据库构建工具')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='输入目录，包含 .md 文件 (默认: data_output/pdf_markdown)')
    parser.add_argument('--chunk_size', type=int, default=500,
                        help='每个 chunk 的最大字符数 (默认: 500)')
    parser.add_argument('--chunk_overlap', type=int, default=50,
                        help='相邻 chunk 的重叠字符数 (默认: 50)')
    parser.add_argument('--model_name', type=str, default='BAAI/bge-large-zh',
                        help='Embedding 模型名称 (默认: BAAI/bge-large-zh)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='向量化的批次大小 (默认: 100)')
    parser.add_argument('--min_chunk_size', type=int, default=30,
                        help='最小 chunk 字符数，过短的将被丢弃 (默认: 30)')
    parser.add_argument('--tag', type=str, default=None,
                        help='手动指定实验标签，覆盖自动生成的标签 (默认: 根据参数自动生成)')
    return parser.parse_args()


def has_chinese(text: str) -> bool:
    """检测路径中是否包含中文字符"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def get_chroma_build_dir(target_dir: str) -> tuple:
    """
    获取 Chroma 构建目录
    如果目标路径包含中文，使用临时目录构建，完成后复制回目标位置
    
    返回: (构建目录, 是否需要复制)
    """
    if has_chinese(target_dir):
        # 使用系统临时目录
        temp_dir = tempfile.mkdtemp(prefix='chroma_build_')
        print(f"检测到中文路径，使用临时目录构建: {temp_dir}")
        return temp_dir, True
    else:
        os.makedirs(target_dir, exist_ok=True)
        return target_dir, False


def split_markdown(text: str, chunk_size: int, chunk_overlap: int) -> list:
    """将 Markdown 文本切分为 chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return text_splitter.split_text(text)


def build_chroma_db(chunk_size: int, chunk_overlap: int, model_name: str,
                    batch_size: int, min_chunk_size: int = 30,
                    db_dir: str = None, input_dir: str = None):
    """
    构建 Chroma 向量数据库
    
    参数:
        min_chunk_size: 最小 chunk 字符数，过短的 chunk（页码、格式残留）会被丢弃
        db_dir: 输出目录，默认为 CHROMA_DIR
        input_dir: 输入目录，默认为 INPUT_DIR
    """
    if db_dir is None:
        db_dir = CHROMA_DIR
    if input_dir is None:
        input_dir = INPUT_DIR

    # 清理旧数据库
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
        print(f"已清理旧数据库: {db_dir}")
    
    # 获取构建目录（处理中文路径问题）
    build_dir, need_copy = get_chroma_build_dir(db_dir)
    
    md_files = sorted([
        f for f in os.listdir(input_dir) if f.endswith('.md')
    ])
    
    print(f"开始构建 Chroma 向量数据库...")
    print(f"输入目录: {input_dir}")
    print(f"Chroma 目录: {db_dir}")
    print(f"切分参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"Embedding 模型: {model_name}")
    print(f"最小 chunk 长度: {min_chunk_size} (过短丢弃)")
    print("=" * 70)
    
    # 初始化 Embeddings
    print("正在加载 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 收集所有 chunks 和元数据
    all_chunks = []
    all_metadatas = []
    
    for md_file in md_files:
        input_path = os.path.join(input_dir, md_file)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = split_markdown(content, chunk_size, chunk_overlap)
        
        # 过滤过短的 chunks（页码、格式残留、单个字符等噪声）
        valid_chunks = [c for c in chunks if len(c.strip()) >= min_chunk_size]
        skipped = len(chunks) - len(valid_chunks)
        if skipped:
            print(f"    (过滤了 {skipped} 个过短 chunk)")

        for i, chunk in enumerate(valid_chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                'source': md_file,
                'chunk_index': i,
                'chunk_size': len(chunk),
            })
        
        print(f"  {md_file}: {len(chunks)} chunks")
    
    print(f"\n总计: {len(all_chunks)} chunks")
    print("正在生成向量并存入 Chroma...")
    
    # 分批处理，避免内存问题
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_metadatas = all_metadatas[i:i+batch_size]
        
        if i == 0:
            db = Chroma.from_texts(
                texts=batch_chunks,
                embedding=embeddings,
                metadatas=batch_metadatas,
                persist_directory=build_dir,
            )
        else:
            db.add_texts(
                texts=batch_chunks,
                metadatas=batch_metadatas,
            )
        
        print(f"  已处理: {min(i+batch_size, len(all_chunks))}/{len(all_chunks)} chunks")
    
    # 如果使用了临时目录，复制到目标位置
    if need_copy:
        print(f"正在复制数据库到目标位置...")
        shutil.copytree(build_dir, db_dir)
        shutil.rmtree(build_dir)
        print(f"已复制到: {db_dir}")
    
    print("=" * 70)
    print(f"Chroma 数据库构建完成！")
    print(f"数据库位置: {db_dir}")
    print(f"总 chunks 数: {len(all_chunks)}")
    
    return db


if __name__ == "__main__":
    args = parse_args()

    # 自动生成隔离标签：根据所有非默认参数生成目录名
    if args.tag:
        # 手动指定标签，完全自定义
        db_dir = os.path.join(
            os.path.dirname(CHROMA_DIR),
            f'chroma_db_{args.tag}',
        )
    else:
        # 自动从参数生成标签，仅当参数与默认值不同时才加后缀
        parts = []
        if args.chunk_size != 500:
            parts.append(f'cs{args.chunk_size}')
        if args.chunk_overlap != 50:
            parts.append(f'ov{args.chunk_overlap}')
        if args.model_name != 'BAAI/bge-large-zh':
            model_short = args.model_name.replace('BAAI/', '').replace('/', '_')
            parts.append(model_short)
        if args.min_chunk_size != 30:
            parts.append(f'min{args.min_chunk_size}')
        if args.batch_size != 100:
            parts.append(f'bs{args.batch_size}')

        if parts:
            db_dir = os.path.join(
                os.path.dirname(CHROMA_DIR),
                f'chroma_db_{"_".join(parts)}',
            )
        else:
            db_dir = CHROMA_DIR  # 全默认参数，保持向后兼容

    input_dir = args.input_dir if args.input_dir else None

    print("=" * 70)
    print("Chroma 向量数据库构建")
    print("=" * 70)
    build_chroma_db(args.chunk_size, args.chunk_overlap, args.model_name,
                    args.batch_size, args.min_chunk_size, db_dir, input_dir)
