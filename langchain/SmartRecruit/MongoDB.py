from pymongo import MongoClient
from pypdf import PdfReader
import hashlib
from datetime import datetime
import os

# 配置 MongoDB 连接
MONGO_URI = "mongodb://admin:123456/localhost:27017/"
DATABASE_NAME = "my_test_db"
COLLECTION_NAME = "resumes"

# 连接并选择数据库与集合
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# 读取 PDF 文件，提取文本内容
def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    print(f"PDF 读取完成，文本长度: {len(text)} 字符")
    return text

# 存储简历到 MongoDB，包括分块
def store_resume(doc_content: str, file_path: str):
    doc_hash = hashlib.md5(doc_content.encode()).hexdigest()
    timestamp = datetime.now().isoformat()

    # 检查是否已存在
    if collection.find_one({"doc_hash": doc_hash}):
        print(f"简历已存在，hash: {doc_hash}, 文件: {file_path}")
        return False

    # 存储完整文档和分块
    collection.insert_one({
        "doc_hash": doc_hash,
        "content": doc_content,
        "file_path": file_path,
        "timestamp": timestamp,
    })
    print(f"简历存储成功，hash: {doc_hash}")
    return True

# 获取完整简历文本
def get_full_resume(doc_hash: str) -> str:
    result = collection.find_one({"doc_hash": doc_hash})
    if result:
        print(f"成功获取完整简历，hash: {doc_hash}")
        return result["content"]
    print(f"未找到完整简历，hash: {doc_hash}")
    return ""


if __name__ == '__main__':
    # 主流程
    pdf_path = r"/langchain/SmartRecruit/data\resume\刘宝.pdf"  # 替换为实际 PDF 路径
    # 处理 PDF
    doc_content = read_pdf(pdf_path)
    print(f"文档长度: {len(doc_content)} 字符")
    # 存储
    success = store_resume(doc_content, pdf_path)
    print(f"存储结果: {'成功' if success else '已存在'}")

    # 查询单个简历（按 user_id）
    print("\n查询 user_id 为 '001' 的用户:")
    result_one = collection.find_one({"doc_hash": "16509539c25a0e6fb1dd1f4f59df637c"})
    print(result_one['content'])