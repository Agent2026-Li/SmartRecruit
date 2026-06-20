# rag/chain.py
import asyncio
import os
import json
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger
from config import config

# 修正导入路径：使用标准的根目录绝对/相对导入
from utils.vector_store import VectorStore

# 初始化LLM和检索器
llm = ChatOpenAI(
    model_name="qwen-plus",
    openai_api_key=config.DASHSCOPE_API_KEY,
    openai_api_base="[https://dashscope.aliyuncs.com/compatible-mode/v1](https://dashscope.aliyuncs.com/compatible-mode/v1)",
    temperature=0.1
)
retriever = VectorStore()

# 定义提示：用于重写复杂或带有上下文的招聘用户输入
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一位专业的招聘助理。请结合历史对话，将用户最新的招聘需求改写为一个适合在简历向量数据库中进行语义检索的单条独立查询语句。请直接输出改写后的查询，不要包含任何解释。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
作为资深技术招聘官和最终审核人，根据【用人需求】和【简历上下文】推荐最匹配的候选人。

【简历上下文】:
---
{context}
---

输出要求：
1. **最终审核**：作为最后一道关卡，确保简历与【用人需求】高度相关。
2. **精准推荐**：仅当简历高度匹配时，生成推荐理由；忽略勉强相关或无关的简历。
3. **结构化 JSON 格式**：你必须严格按照以下 JSON 字典格式输出，不要包含任何开场白或解释：
{{
    "response": "这里填写对本次招聘筛选的简要总结（例如：为您筛选到2位匹配的算法工程师...）",
    "candidates": [
        {{
            "doc_hash": "简历哈希值",
            "file_path": "简历对应的本地文件绝对路径（必须从上下文的file_path中完全一致地复制）",
            "reason": "具体的推荐理由，说明其为什么符合招聘需求"
        }}
    ]
}}
如果上下文中没有任何简历匹配需求，请将 "candidates" 设为空列表并输出。
"""),
    ("user", "【用人需求】: {input}"),
])


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "未在简历库中找到相关信息。"
    formatted_docs = []
    for doc in docs:
        doc_hash = doc.metadata.get("doc_hash", doc.metadata.get("hash", "N/A"))
        doc_str = (
            f"简历来源文件: {doc.metadata.get('file_path', 'N/A')}\n"
            f"简历哈希值: {doc_hash}\n"
            f"内容: {doc.page_content}"
        )
        formatted_docs.append(doc_str)
    return "\n\n---\n\n".join(formatted_docs)


def _clean_chat_history(messages: List[Any]) -> List[Any]:
    """
    清洗聊天历史。
    由于 app.py 中的 AIMessage 内部存放的是大模型输出的结构化 JSON 字符串，
    我们需要将 JSON 里的 response 文本提取出来传给改写模型，防止 JSON 噪音干扰 Query 重写。
    """
    cleaned_history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            cleaned_history.append(msg)
        elif isinstance(msg, AIMessage):
            try:
                data = json.loads(msg.content)
                text_content = data.get("response", msg.content)
                cleaned_history.append(AIMessage(content=text_content))
            except Exception:
                cleaned_history.append(msg)
    return cleaned_history


async def get_rag_chain():
    """构建并返回支持历史记录和动态参数的异步 RAG 链。"""

    async def retrieve_and_format_context(input_dict: dict, config: RunnableConfig) -> str:
        # 清洗 chat_history，防止结构化 JSON 字符串弄脏 Query 改写
        raw_history = input_dict.get("chat_history", [])
        cleaned_history = _clean_chat_history(raw_history)

        # 1. 改写查询
        rewritten_question = await (REWRITE_PROMPT | llm | StrOutputParser()).ainvoke(
            {"input": input_dict["input"], "chat_history": cleaned_history},
            config=config
        )
        logger.info(f"原始查询: '{input_dict['input']}' -> 改写后查询: '{rewritten_question}'")

        # 2. 解析参数并构建 Filter 表达式
        params = input_dict.get("params", {})

        # 3. 异步执行高级混合检索
        try:
            retrieved_docs = await retriever.aget_relevant_documents(
                query=rewritten_question,
                params=params
            )
        except Exception as e:
            logger.error(f"检索器执行失败: {e}", exc_info=True)
            return "检索简历时发生内部错误，请稍后再试。"

        # 4. 格式化
        context = _format_docs(retrieved_docs)
        logger.debug(f"为LLM准备的上下文: \n{context}")
        return context

    def parse_json_response(output: str) -> Dict[str, Any]:
        """完美剥离 Markdown 标记并安全解析为标准 Python 字典"""
        clean_output = output.strip()

        # 稳健地移除前缀
        if clean_output.startswith("```json"):
            clean_output = clean_output[7:]
        elif clean_output.startswith("```"):
            clean_output = clean_output[3:]

        # 稳健地移除后缀
        if clean_output.endswith("```"):
            clean_output = clean_output[:-3]
            clean_output = clean_output.strip()

        try:
            return json.loads(clean_output)
        except Exception as e:
            logger.error(f"LLM 输出未按规范生成 JSON 字典。错误: {e}。原始输出:\n{output}")
            return {
                "response": f"解析异常，AI 原始响应如下：{output}",
                "candidates": []
            }

    # 构建最终链
    conversational_rag_chain = (
            RunnablePassthrough.assign(
                context=retrieve_and_format_context
            )
            | ANSWER_PROMPT
            | llm
            | StrOutputParser()
            | RunnableLambda(parse_json_response)
    )

    return conversational_rag_chain


# --- [验证代码] ---
if __name__ == '__main__':
    async def main():
        """独立验证 RAG Chain 的核心功能"""
        import sys
        # 动态将当前项目根目录挂载到 sys.path，防止本地执行时找不到 config 或 utils
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        if project_root not in sys.path:
            sys.path.append(project_root)

        logger.info("=" * 50)
        logger.info("开始独立验证 chain.py 模块...")

        try:
            rag_chain = await get_rag_chain()

            # --- 测试用例 1: 简单的招聘需求 ---
            print("\n--- 测试用例 1: 简单招聘需求 ---")
            query1 = "我需要一个懂 AI 算法的工程师"
            params1 = {"count": 2}
            print(f"用户: {query1}, 参数: {params1}")

            response1 = await rag_chain.ainvoke({
                "input": query1,
                "chat_history": [],
                "params": params1
            })
            print(f"AI 响应类型: {type(response1)}")
            print(f"AI 响应内容: {json.dumps(response1, ensure_ascii=False, indent=4)}")

            # 断言验证安全性
            assert isinstance(response1, dict), "输出必须已被转换为标准 Python 字典"
            assert "response" in response1, "字典必须包含 'response' 键"
            assert "candidates" in response1, "字典必须包含 'candidates' 键"
            print("【测试用例 1 通过】")

            logger.info("=" * 50)
            logger.success("chain.py 模块所有功能验证通过！")

        except Exception as e:
            logger.critical(f"独立验证发生异常未通过: {e}", exc_info=True)


    asyncio.run(main())