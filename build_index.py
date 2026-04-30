"""
构建向量索引脚本
读取 ./data/ 文件夹中的 PDF 文件，建立并持久化 Chroma 向量数据库索引

使用方法:
    python build_index.py

依赖:
    - langchain
    - chromadb
    - pypdf
    - langchain-openai (用于嵌入模型)

Author: Immunosuppressant Advisor Team
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import streamlit as st


DATA_DIR = "./data"
PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_openai_api_key() -> str:
    """
    获取 OpenAI API 密钥
    优先使用环境变量或 streamlit secrets
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")

    try:
        if hasattr(st, "secrets"):
            api_key = st.secrets.get("OPENAI_API_KEY", api_key)
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    return api_key


def load_pdf_documents(data_dir: str) -> List:
    """
    加载指定目录下的所有 PDF 文件

    参数:
        data_dir: PDF 文件所在目录路径

    返回:
        加载的文档列表
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"错误: 数据目录 {data_dir} 不存在")
        return []

    pdf_files = list(data_path.glob("*.pdf"))

    if not pdf_files:
        print(f"警告: 在 {data_dir} 中未找到 PDF 文件")
        return []

    documents = []

    for pdf_file in pdf_files:
        try:
            print(f"正在加载: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = pdf_file.name
            documents.extend(docs)
            print(f"  成功加载 {len(docs)} 页")
        except Exception as e:
            print(f"  加载失败 {pdf_file.name}: {e}")

    return documents


def split_documents(documents: List, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List:
    """
    将文档分割成小块

    参数:
        documents: 原始文档列表
        chunk_size: 每个文本块的最大字符数
        chunk_overlap: 相邻文本块之间的重叠字符数

    返回:
        分割后的文档列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"文档分割完成，共 {len(split_docs)} 个文本块")
    return split_docs


def create_vector_index(
    documents: List,
    persist_dir: str,
    api_key: str,
    embedding_model: str = "text-embedding-3-small"
) -> Optional[Chroma]:
    """
    创建并持久化向量索引

    参数:
        documents: 分割后的文档列表
        persist_dir: 向量数据库持久化目录
        api_key: OpenAI API 密钥
        embedding_model: 嵌入模型名称

    返回:
        Chroma 向量数据库实例
    """
    if not documents:
        print("错误: 没有文档可供索引")
        return None

    if not api_key:
        print("错误: 未提供 API 密钥")
        return None

    try:
        os.environ["OPENAI_API_KEY"] = api_key

        print(f"正在使用模型 {embedding_model} 生成嵌入...")

        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=api_key,
        )

        print("正在创建向量索引...")

        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
        )

        print(f"向量索引创建成功，已保存至: {persist_dir}")
        print(f"索引包含 {vector_db._collection.count()} 个向量")

        return vector_db

    except Exception as e:
        print(f"创建向量索引时出错: {e}")
        return None


def build_index(
    data_dir: str = DATA_DIR,
    persist_dir: str = PERSIST_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Optional[Chroma]:
    """
    完整的索引构建流程

    参数:
        data_dir: PDF 文件所在目录
        persist_dir: 向量数据库持久化目录
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠大小

    返回:
        Chroma 向量数据库实例
    """
    print("=" * 50)
    print("免疫抑制剂知识库向量索引构建")
    print("=" * 50)

    print("\n[1/4] 加载 PDF 文档...")
    documents = load_pdf_documents(data_dir)
    if not documents:
        print("未找到可加载的文档")
        return None

    print(f"\n总计加载 {len(documents)} 个文档")

    print("\n[2/4] 分割文档...")
    split_docs = split_documents(documents, chunk_size, chunk_overlap)

    print("\n[3/4] 获取 API 密钥...")
    api_key = get_openai_api_key()
    if not api_key:
        print("错误: 无法获取 API 密钥，请设置 OPENAI_API_KEY 环境变量或配置 streamlit secrets")
        return None

    print("\n[4/4] 创建向量索引...")
    vector_db = create_vector_index(split_docs, persist_dir, api_key)

    if vector_db:
        print("\n" + "=" * 50)
        print("索引构建完成!")
        print("=" * 50)
    else:
        print("\n索引构建失败")

    return vector_db


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="构建免疫抑制剂知识库向量索引")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR,
        help=f"PDF 文件所在目录 (默认: {DATA_DIR})"
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default=PERSIST_DIR,
        help=f"向量数据库持久化目录 (默认: {PERSIST_DIR})"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=CHUNK_SIZE,
        help=f"文本块大小 (默认: {CHUNK_SIZE})"
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"文本块重叠大小 (默认: {CHUNK_OVERLAP})"
    )

    args = parser.parse_args()

    build_index(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
