#模型下载
# from modelscope import snapshot_download
# cache_dir=r"D:\LLM_Codes\Chapter3_RAG\models"
# model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct',cache_dir=cache_dir)
# print("模型下载完成！！！")

from huggingface_hub import login, snapshot_download
from langchain_community.embeddings import HuggingFaceBgeEmbeddings  # 修正导入
import os

# 设置 Hugging Face API 令牌
login(token="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")

# 指定模型名称和缓存路径
model_name = "BAAI/bge-large-zh"
cache_dir = r"/root/LLM_Codes/Chapter5_LLMFineTuning/models"

# 创建缓存目录
os.makedirs(cache_dir, exist_ok=True)

# 下载模型，增加并行线程
model_dir = snapshot_download(
    repo_id=model_name,
    cache_dir=cache_dir,
    resume_download=True,
    max_workers=8
)

print(f"模型下载完成！路径：{model_dir}")

# 设置模型参数
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}

# 初始化模型
model = HuggingFaceBgeEmbeddings(
    model_name=model_dir,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章：",
    cache_folder=cache_dir
)

model.query_instruction = "为这个句子生成表示以用于检索相关文章："