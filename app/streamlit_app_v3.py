# 文件: app/streamlit_app_v3.py
# RAG v3.0 - 真正能思考的版本

import streamlit as st
import tempfile
import os
import gc
import time
import uuid
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="RAG v3.0 - 智能版",
    page_icon="🧠",
    layout="wide"
)

# ==================== 配置 ====================
VECTORDB_BASE_PATH = r"D:\local-rag-chatbot\data\vectordb"
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3:8b"
EMBEDDING_MODEL = "nomic-embed-text"
MAX_HISTORY_TURNS = 5

# ==================== Session State ====================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "debug_info" not in st.session_state:
    st.session_state.debug_info = {}


# ==================== 核心模型 ====================

@st.cache_resource
def get_llm():
    return OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3  # 降低随机性，更稳定
    )


@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )


# ==================== 工具函数 ====================

def cleanup_old_vectorstore():
    if st.session_state.vectorstore is not None:
        try:
            st.session_state.vectorstore = None
            gc.collect()
            time.sleep(0.5)
        except:
            pass


def process_pdf(uploaded_file):
    """处理 PDF"""
    cleanup_old_vectorstore()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # 更细的分块，提高检索精度
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # 更小的块
            chunk_overlap=100,  # 更多重叠
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        chunks = text_splitter.split_documents(pages)

        db_path = f"{VECTORDB_BASE_PATH}_{uuid.uuid4().hex[:8]}"

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=db_path
        )

        return vectorstore, pages, chunks
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


# ==================== 智能检索模块 ====================

def expand_query(original_question: str) -> list:
    """
    查询扩展：让 LLM 生成多个搜索角度
    """
    prompt = f"""用户问题: "{original_question}"

请从不同角度生成 3 个搜索关键词/短语，用于检索相关文档。
要求：
1. 覆盖问题的不同方面
2. 包含同义词或相关概念
3. 每行一个，不要编号，不要解释

关键词："""

    result = get_llm().invoke(prompt)

    # 解析关键词
    queries = [original_question]  # 原始问题
    for line in result.strip().split('\n'):
        line = line.strip().strip('-•*1234567890.').strip()
        if line and 2 < len(line) < 50:
            queries.append(line)

    return queries[:4]  # 最多 4 个查询


def smart_retrieve(question: str, vectorstore, top_k: int = 5) -> list:
    """
    智能检索：多查询 + 去重 + 排序
    """
    # 1. 扩展查询
    queries = expand_query(question)

    # 2. 多次检索
    all_docs = []
    seen_content = set()
    doc_scores = {}

    for i, query in enumerate(queries):
        # 带分数的检索
        results = vectorstore.similarity_search_with_score(query, k=top_k)

        for doc, score in results:
            content_key = doc.page_content[:100]

            if content_key not in seen_content:
                seen_content.add(content_key)
                all_docs.append(doc)
                doc_scores[content_key] = score
            else:
                # 如果重复出现，降低分数（说明更相关）
                doc_scores[content_key] = min(doc_scores[content_key], score)

    # 3. 按分数排序（分数越低越相关）
    all_docs.sort(key=lambda d: doc_scores.get(d.page_content[:100], 999))

    return all_docs[:8], queries  # 返回最相关的 8 个


# ==================== 思考型回答模块 ====================

def format_history(chat_history, max_turns=3):
    """格式化历史对话"""
    if not chat_history:
        return "无"

    recent = chat_history[-max_turns:]
    lines = []
    for chat in recent:
        lines.append(f"用户: {chat['question']}")
        answer_preview = chat['answer'][:150] + "..." if len(chat['answer']) > 150 else chat['answer']
        lines.append(f"助手: {answer_preview}")
    return "\n".join(lines)


def thinking_answer(question: str, docs: list, chat_history: list) -> str:
    """
    思考型回答：让 LLM 真正分析
    """
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    history = format_history(chat_history)

    # 核心：引导 LLM 思考的 Prompt
    prompt = f"""你是一位专业的分析顾问。用户基于一份文档向你提问。

## 参考资料
{context}

## 对话历史
{history}

## 用户问题
{question}

---

## 请按以下步骤思考和回答：

### 第一步：理解问题
这个问题真正想问的是什么？有什么隐含的需求？

### 第二步：信息提取
从参考资料中，找出与问题相关的关键信息。列出要点。

### 第三步：分析推理
基于这些信息，进行分析。
- 如果是事实性问题：提取并整理事实
- 如果是分析性问题：给出你的分析逻辑
- 如果是比较性问题：列出对比要点
- 如果是建议性问题：给出有依据的建议

### 第四步：回答
给出最终回答。要求：
- 直接回应问题
- 言之有物，不说废话
- 如果资料不足以完整回答，明确说明缺少什么信息

---

现在开始你的分析："""

    return get_llm().invoke(prompt)


def check_answer_quality(question: str, answer: str) -> tuple:
    """
    检查回答质量，决定是否需要改进
    """
    check_prompt = f"""评估以下回答的质量：

问题：{question}

回答：{answer}

请评估（只回复一个词）：
- 如果回答直接回应了问题且有实质内容，回复：GOOD
- 如果回答空洞、没有实质内容或完全跑题，回复：BAD

评估："""

    result = get_llm().invoke(check_prompt).strip().upper()
    is_good = "GOOD" in result
    return is_good, result


def improve_answer(question: str, original_answer: str, docs: list) -> str:
    """
    改进不够好的回答
    """
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""原始问题：{question}

参考资料：
{context}

之前的回答（质量不够好）：
{original_answer}

请重新回答这个问题。要求：
1. 直接回应问题核心
2. 从参考资料中提取具体信息
3. 如果资料中确实没有相关信息，直接说明
4. 简洁有力，不要空话

改进后的回答："""

    return get_llm().invoke(prompt)


# ==================== 主回答函数 ====================

def smart_rag_answer(question: str, vectorstore, chat_history: list, debug=False):
    """
    完整的智能 RAG 流程
    """
    debug_info = {}

    # 1. 智能检索
    docs, queries = smart_retrieve(question, vectorstore)
    debug_info["search_queries"] = queries
    debug_info["docs_found"] = len(docs)

    if not docs:
        return "抱歉，没有找到相关信息。", [], debug_info

    # 2. 思考型回答
    answer = thinking_answer(question, docs, chat_history)
    debug_info["first_answer"] = answer[:200] + "..."

    # 3. 质量检查
    is_good, quality = check_answer_quality(question, answer)
    debug_info["quality_check"] = quality

    # 4. 如果不够好，改进
    if not is_good:
        answer = improve_answer(question, answer, docs)
        debug_info["improved"] = True
    else:
        debug_info["improved"] = False

    return answer, docs, debug_info


# ==================== 简洁回答模式 ====================

def concise_answer(question: str, docs: list, chat_history: list) -> str:
    """
    简洁回答模式：直接给答案
    """
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    history = format_history(chat_history)

    prompt = f"""参考资料：
{context}

对话历史：
{history}

问题：{question}

请直接、简洁地回答问题。
- 如果能从资料中找到答案，直接给出
- 如果找不到，说"根据现有资料，未找到相关信息"
- 不要说多余的话

回答："""

    return get_llm().invoke(prompt)


# ==================== 页面布局 ====================

st.title("🧠 RAG v3.0 - 智能版")
st.caption("多角度检索 + 思考型回答 + 质量自检")

col1, col2 = st.columns([1, 2])

# ==================== 左侧控制面板 ====================
with col1:
    st.header("📁 文档")

    uploaded_file = st.file_uploader("上传 PDF", type=["pdf"])

    if uploaded_file:
        if st.button("🚀 处理文档", type="primary", use_container_width=True):
            with st.spinner("处理中..."):
                try:
                    vectorstore, pages, chunks = process_pdf(uploaded_file)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.all_chunks = chunks
                    st.session_state.pdf_processed = True
                    st.session_state.chat_history = []
                    st.success(f"✅ {len(pages)} 页 / {len(chunks)} 块")
                except Exception as e:
                    st.error(f"失败: {e}")

    st.divider()

    # 回答模式选择
    st.header("⚙️ 设置")
    answer_mode = st.radio(
        "回答模式",
        ["🧠 深度思考", "⚡ 快速简洁"],
        help="深度思考更智能但较慢，快速简洁更直接"
    )

    show_debug = st.checkbox("显示调试信息", value=False)

    st.divider()

    # 状态
    st.header("📊 状态")
    if st.session_state.pdf_processed:
        st.success("✅ 就绪")
        st.caption(f"对话轮数: {len(st.session_state.chat_history)}")
    else:
        st.warning("⏳ 等待上传")

    if st.session_state.chat_history:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.debug_info = {}
            st.rerun()

# ==================== 右侧对话区 ====================
with col2:
    st.header("💬 对话")

    # 对话历史
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])

            # 显示参考来源
            with st.expander("📖 参考来源"):
                for i, doc in enumerate(chat.get("sources", [])[:3]):
                    st.caption(f"**[{i + 1}]** {doc.page_content[:200]}...")

            # 显示调试信息
            if show_debug and "debug" in chat:
                with st.expander("🔧 调试信息"):
                    debug = chat["debug"]
                    st.write(f"**检索关键词:** {debug.get('search_queries', [])}")
                    st.write(f"**找到文档:** {debug.get('docs_found', 0)} 块")
                    st.write(f"**质量检查:** {debug.get('quality_check', 'N/A')}")
                    st.write(f"**是否改进:** {debug.get('improved', False)}")

    # 输入
    if st.session_state.pdf_processed:
        question = st.chat_input("输入问题...")

        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    try:
                        if "深度" in answer_mode:
                            answer, docs, debug_info = smart_rag_answer(
                                question,
                                st.session_state.vectorstore,
                                st.session_state.chat_history
                            )
                        else:
                            docs, queries = smart_retrieve(
                                question,
                                st.session_state.vectorstore
                            )
                            answer = concise_answer(
                                question,
                                docs,
                                st.session_state.chat_history
                            )
                            debug_info = {"search_queries": queries, "mode": "concise"}

                        st.write(answer)

                        with st.expander("📖 参考来源"):
                            for i, doc in enumerate(docs[:3]):
                                st.caption(f"**[{i + 1}]** {doc.page_content[:200]}...")

                        if show_debug:
                            with st.expander("🔧 调试信息"):
                                st.write(debug_info)

                        # 保存
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer,
                            "sources": docs,
                            "debug": debug_info
                        })

                    except Exception as e:
                        st.error(f"出错: {e}")
    else:
        st.info("👆 请先上传 PDF 文档")

st.divider()
st.caption("🔒 本地处理 | v3.0 智能检索 + 思考型回答")