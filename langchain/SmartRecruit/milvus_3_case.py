# milvus_3_case.py
# 这个脚本复刻 vector_store.py 的核心逻辑，并结合 document_processor.py 的层级分块机制。
# 用于演示 PDF 简历的读取、分块（父块和子块）、向量化、存储和混合检索。
# 使用整个文档的哈希值（doc_hash）作为 parent_id 和 doc_hash 字段，子块 ID 包含父块信息。
# 逻辑扁平化，所有操作在 main() 函数中完成，带详细中文注释，易于理解。
# 包含 Milvus (dense + sparse) 混合检索，简化了 Elasticsearch 和 rerank 部分。
# 注意：需确保安装 pymilvus, pypdf, milvus_model.hybrid, langchain 依赖，并调整路径和配置。
import hashlib
import time
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from pypdf import PdfReader
from milvus_model.hybrid import BGEM3EmbeddingFunction
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import os
from datetime import datetime
# --- 1. 配置信息（根据您的环境修改） ---
MILVUS_ENDPOINT = "http://localhost:19530"  # Milvus 服务器地址
COLLECTION_NAME = "resume_collection_demo"  # Milvus 集合名称
PDF_PATH = r"D:\python\workspace\ai_coze\PythonProject\langchain\SmartRecruit\data\resume\刘宝.pdf"  # 简历 PDF 文件路径
EMBEDDING_MODEL_PATH = r"D:\python\workspace\ai_coze\PythonProject\langchain\models\m3e-base"  # BGE-M3 模型路径
PARENT_CHUNK_SIZE = 1000  # 父块大小（参考 document_processor.py）
CHILD_CHUNK_SIZE = 400  # 子块大小
CHUNK_OVERLAP = 50  # 块重叠大小

def read_pdf(file_path: str) -> str:
    """
    纯粹读取 PDF 文件，提取全部文本内容并返回。

    参数:
    - file_path (str): PDF 文件的本地路径

    返回:
    - str: PDF 的全文文本内容
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到 PDF 文件: {file_path}")

    # 1. 初始化读取器
    reader = PdfReader(file_path)
    text_list = []

    # 2. 遍历每一页并提取文本
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # 确保页面有文字（排除空白页）
            text_list.append(page_text)

    # 3. 将所有页面的文本用换行符拼接成一个完整的字符串
    full_text = "\n".join(text_list)

    return full_text
def chunk_text_hierarchical(text: str, parent_chunk_size: int = PARENT_CHUNK_SIZE,
                            child_chunk_size: int = CHILD_CHUNK_SIZE,
                            chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    # 层级分块：先切分为父块，再将每个父块切分为子块，添加 parent_id 和 parent_content
    if chunk_overlap >= min(parent_chunk_size, child_chunk_size):
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    # 创建父块和子块的分块器
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # 创建原始文档
    doc_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    doc = Document(page_content=text,
                   metadata={"file_path": PDF_PATH, "hash": doc_hash, "timestamp": datetime.now().isoformat()})

    # 切分为父块
    parent_docs = parent_splitter.split_documents([doc])
    print(f"生成 {len(parent_docs)} 个父块")

    # 切分为子块并添加元数据
    child_chunks = []
    for j, parent_doc in enumerate(parent_docs):
        parent_id = f"{doc_hash}_parent_{j}"
        parent_doc.metadata["parent_id"] = parent_id
        parent_doc.metadata["parent_content"] = parent_doc.page_content
        parent_doc.metadata.update(doc.metadata)

        # 对每个父块进行子块切分
        sub_chunks = child_splitter.split_documents([parent_doc])
        for k, sub_chunk in enumerate(sub_chunks):
            sub_chunk.metadata["parent_id"] = parent_id
            sub_chunk.metadata["parent_content"] = parent_doc.page_content
            sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
            sub_chunk.metadata.update(parent_doc.metadata)
            child_chunks.append(sub_chunk)

    print(f"生成 {len(child_chunks)} 个子块")
    return child_chunks

def chunk_text_hierarchical(text: str, parent_chunk_size: int = PARENT_CHUNK_SIZE,
                            child_chunk_size: int = CHILD_CHUNK_SIZE,
                            chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    # 层级分块：先切分为父块，再将每个父块切分为子块，添加 parent_id 和 parent_content
    if chunk_overlap >= min(parent_chunk_size, child_chunk_size):
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    # 创建父块和子块的分块器
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # 创建原始文档
    doc_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    doc = Document(page_content=text,
                   metadata={"file_path": PDF_PATH, "hash": doc_hash, "timestamp": datetime.now().isoformat()})

    # 切分为父块
    parent_docs = parent_splitter.split_documents([doc])
    print(f"生成 {len(parent_docs)} 个父块")

    # 切分为子块并添加元数据
    child_chunks = []
    for j, parent_doc in enumerate(parent_docs):
        parent_id = f"{doc_hash}_parent_{j}"
        parent_doc.metadata["parent_id"] = parent_id
        parent_doc.metadata["parent_content"] = parent_doc.page_content
        parent_doc.metadata.update(doc.metadata)

        # 对每个父块进行子块切分
        sub_chunks = child_splitter.split_documents([parent_doc])
        for k, sub_chunk in enumerate(sub_chunks):
            sub_chunk.metadata["parent_id"] = parent_id
            sub_chunk.metadata["parent_content"] = parent_doc.page_content
            sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
            sub_chunk.metadata.update(parent_doc.metadata)
            child_chunks.append(sub_chunk)

    print(f"生成 {len(child_chunks)} 个子块")
    return child_chunks
def csr_to_milvus_sparse(csr_matrix, index: int) -> dict:
    # 从 CSR 格式的稀疏向量中提取指定索引（行）的非零索引和值，转换为 Milvus 兼容的字典格式
    indices = csr_matrix.indices[csr_matrix.indptr[index]:csr_matrix.indptr[index + 1]]
    data = csr_matrix.data[csr_matrix.indptr[index]:csr_matrix.indptr[index + 1]]
    return {int(k): float(v) for k, v in zip(indices, data)}
# --- 3. 主执行流程 ---

if __name__ == '__main__':

    # 步骤 A: 初始化 Milvus 客户端
    print("初始化 Milvus 客户端...")
    client = MilvusClient(uri=MILVUS_ENDPOINT,db_name="milvus_demo")

    # 步骤 B: 清理旧集合（确保脚本可重复运行）
    if client.has_collection(collection_name=COLLECTION_NAME):
        print(f"发现旧集合 {COLLECTION_NAME}，正在删除...")
        client.drop_collection(collection_name=COLLECTION_NAME)

    # 步骤 C: 加载 BGE-M3 嵌入模型
    print(f"加载 BGE-M3 模型: {EMBEDDING_MODEL_PATH}")
    embedding_function = BGEM3EmbeddingFunction(
        model_name=EMBEDDING_MODEL_PATH,
        device='cpu',
        use_fp16=False
    )
    dense_dim = embedding_function.dim["dense"]  # 获取密集向量维度
    print(f"密集向量维度: {dense_dim}")

    # 步骤 D: 读取 PDF 并进行层级分块
    print("读取 PDF 并进行层级分块...")
    document_text = read_pdf(PDF_PATH)
    child_chunks = chunk_text_hierarchical(document_text)

    # 步骤 E: 生成子块的 dense 和 sparse 向量
    print("生成子块的 dense 和 sparse 向量...")
    start_time = time.time()
    texts = [chunk.page_content for chunk in child_chunks]
    embeddings = embedding_function.encode_documents(texts)
    print(f"向量生成完成，耗时: {time.time() - start_time:.2f} 秒")

    # 步骤 F: 创建 Milvus 集合和索引（完全复刻 vector_store.py 的 schema）
    print(f"创建 Milvus 集合: {COLLECTION_NAME}")
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)  # 主键
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=100)  # chunk ID
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)  # 文本块内容
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)  # 密集向量
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量
    schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)  # 父文档 ID
    schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)  # 父文档内容
    schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=255)  # 文件路径
    schema.add_field(field_name="doc_hash", datatype=DataType.VARCHAR, max_length=32)  # 文档哈希
    schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)  # 时间戳

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector", index_name="dense_index", index_type="IVF_FLAT",
        metric_type="IP", params={"nlist": 128}
    )
    index_params.add_index(
        field_name="sparse_vector", index_name="sparse_index", index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP", params={"drop_ratio_build": 0.2}
    )
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)
    print("集合和索引创建完成")

    # 步骤 G: 准备数据并插入 Milvus
    print("准备数据并插入 Milvus...")
    data_to_insert = []
    for i, chunk in enumerate(child_chunks):
        sparse_vector = csr_to_milvus_sparse(embeddings["sparse"], i)
        chunk_data = {
            "id": chunk.metadata["id"],  # 使用子块 ID
            "chunk_id": chunk.metadata["id"],  # chunk_id 与 id 相同
            "text": chunk.page_content,  # 子块内容
            "dense_vector": embeddings["dense"][i].tolist(),  # 密集向量
            "sparse_vector": sparse_vector,  # 稀疏向量
            "parent_id": chunk.metadata["parent_id"],  # 父块 ID
            "parent_content": chunk.metadata["parent_content"],  # 父块内容
            "file_path": chunk.metadata["file_path"],  # 文件路径
            "doc_hash": chunk.metadata["hash"],  # 文档哈希
            "timestamp": chunk.metadata["timestamp"]  # 时间戳
        }
        data_to_insert.append(chunk_data)
    client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {len(data_to_insert)} 条数据到 Milvus")

    # 步骤 H: 执行混合搜索（Milvus dense + sparse）
    print("执行混合搜索...")
    client.load_collection(collection_name=COLLECTION_NAME)
    query = "4. 熟悉机器学习模型训练、优化、评估、部署等工作"
    query_embeddings = embedding_function.encode_queries([query])
    dense_query_vector = query_embeddings["dense"][0].tolist()
    sparse_query_vector = csr_to_milvus_sparse(query_embeddings["sparse"], 0)  # 查询只有一个向量

    dense_request = AnnSearchRequest(
        data=[dense_query_vector], anns_field="dense_vector",
        param={"metric_type": "IP", "params": {"nprobe": 10}}, limit=3
    )
    sparse_request = AnnSearchRequest(
        data=[sparse_query_vector], anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {}}, limit=3
    )
    reranker = WeightedRanker(0.5, 0.5)
    milvus_results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_request, sparse_request],
        ranker=reranker,
        limit=3,
        output_fields=["text", "doc_hash", "file_path", "timestamp", "parent_content", "parent_id"]
    )[0]

    # 步骤 I: 输出搜索结果（优先返回父文档内容）
    print("\n--- 搜索结果 ---")
    if not milvus_results:
        print("未找到相关结果。")
    else:
        print(milvus_results)
        print("============================================")
        for hit in milvus_results:
            entity = hit["entity"]
            print(f"文档哈希: {entity['doc_hash']}")
            print(f"文件路径: {entity['file_path']}")
            print(f"时间戳: {entity['timestamp']}")
            print(f"父块 ID: {entity['parent_id']}")
            print(f"父文档内容: {entity['parent_content'][:250]}...")
            print(f"子块内容: {entity['text'][:100]}...")
            print(f"相似度得分: {hit['distance']:.4f}")
            print("-" * 40)

    # # 步骤 J: 清理 Milvus 集合
    # print("清理 Milvus 集合...")
    # client.drop_collection(collection_name=COLLECTION_NAME)
    # print("集合已删除")