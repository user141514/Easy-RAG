# 文件: app/streamlit_app_v2.py
# RAG 知识库 v2.0 - 支持对话记忆 + 全文摘要

import streamlit as st
import tempfile
import os
import shutil
import gc
import time
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="RAG 知识库 v2.0",
    page_icon="📚",
    layout="wide"
)

# ==================== 常量配置 ====================
VECTORDB_BASE_PATH = r"D:\local-rag-chatbot\data\vectordb"
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3:8b"
EMBEDDING_MODEL = "nomic-embed-text"

# 对话记忆配置
MAX_HISTORY_TURNS = 5  # 保留最近 5 轮对话

# ==================== Session State ====================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "document_summary" not in st.session_state:
    st.session_state.document_summary = None


# ==================== 核心函数 ====================

@st.cache_resource
def get_llm():
    return OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL
    )


@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )


def cleanup_old_vectorstore():
    """清理旧的向量数据库连接"""
    if st.session_state.vectorstore is not None:
        try:
            st.session_state.vectorstore = None
            gc.collect()
            time.sleep(0.5)
        except:
            pass


def process_pdf(uploaded_file):
    """处理上传的 PDF 文件"""

    cleanup_old_vectorstore()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 加载 PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # 分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        )
        chunks = text_splitter.split_documents(pages)

        # 创建向量数据库
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


def format_chat_history(chat_history, max_turns=MAX_HISTORY_TURNS):
    """格式化对话历史"""
    if not chat_history:
        return "无"

    # 只取最近 N 轮
    recent_history = chat_history[-max_turns:]

    formatted = []
    for chat in recent_history:
        formatted.append(f"用户: {chat['question']}")
        # 截断过长的回答
        answer = chat['answer'][:200] + "..." if len(chat['answer']) > 200 else chat['answer']
        formatted.append(f"助手: {answer}")

    return "\n".join(formatted)


def get_rag_response_with_memory(question: str, vectorstore, chat_history, top_k: int = 3):
    """带对话记忆的 RAG 问答"""

    # 1. 检索相关文档
    retrieved_docs = vectorstore.similarity_search(question, k=top_k)
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    # 2. 格式化对话历史
    history_text = format_chat_history(chat_history)

    # 3. 构建带记忆的 Prompt
    template = """你是一个专业的知识库助手。请根据参考文档和对话历史来回答用户的问题。

【参考文档】
{context}

【对话历史】
{history}

【当前问题】
{question}

【回答要求】
1. 结合对话历史理解用户意图（如"他"、"它"、"这个"等代词）
2. 只根据参考文档内容回答，不要编造
3. 如果文档中没有相关信息，明确说明
4. 回答简洁准确，使用中文

回答："""

    prompt = PromptTemplate(
        input_variables=["context", "history", "question"],
        template=template
    )

    formatted_prompt = prompt.format(
        context=context,
        history=history_text,
        question=question
    )

    response = get_llm().invoke(formatted_prompt)

    return response, retrieved_docs


def summarize_single_chunk(chunk_text: str, chunk_index: int, total_chunks: int):
    """摘要单个文档块"""

    template = """请用 2-3 句话概括以下文本的核心内容：

{text}

概括："""

    prompt = PromptTemplate(
        input_variables=["text"],
        template=template
    )

    formatted_prompt = prompt.format(text=chunk_text)
    summary = get_llm().invoke(formatted_prompt)

    return summary.strip()


def combine_summaries(summaries: list):
    """合并多个摘要"""

    combined_text = "\n\n".join([f"片段{i + 1}: {s}" for i, s in enumerate(summaries)])

    template = """以下是一份文档各部分的摘要。请将它们整合成一份完整、连贯的总结（300-500字）：

{summaries}

【要求】
1. 保留关键信息和主要观点
2. 逻辑清晰，结构完整
3. 使用中文

综合总结："""

    prompt = PromptTemplate(
        input_variables=["summaries"],
        template=template
    )

    formatted_prompt = prompt.format(summaries=combined_text)
    final_summary = get_llm().invoke(formatted_prompt)

    return final_summary.strip()


def generate_document_summary(chunks, progress_callback=None):
    """
    Map-Reduce 方式生成全文摘要

    Args:
        chunks: 文档块列表
        progress_callback: 进度回调函数
    """

    total_chunks = len(chunks)

    # ===== MAP 阶段：每个块生成摘要 =====
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i + 1, total_chunks, "MAP")

        summary = summarize_single_chunk(chunk.page_content, i, total_chunks)
        chunk_summaries.append(summary)

    # ===== REDUCE 阶段：合并摘要 =====
    if progress_callback:
        progress_callback(0, 0, "REDUCE")

    # 如果摘要太多，分批合并
    while len(chunk_summaries) > 10:
        batch_size = 5
        new_summaries = []

        for i in range(0, len(chunk_summaries), batch_size):
            batch = chunk_summaries[i:i + batch_size]
            merged = combine_summaries(batch)
            new_summaries.append(merged)

        chunk_summaries = new_summaries

    # 最终合并
    final_summary = combine_summaries(chunk_summaries)

    return final_summary


# ==================== 页面布局 ====================

st.title("📚 RAG 知识库 v2.0")
st.caption("支持对话记忆 + 全文摘要 | 数据完全本地处理")

# 三列布局
col1, col2 = st.columns([1, 2])

# ==================== 左侧：控制面板 ====================
with col1:
    st.header("📁 文档管理")

    # 文件上传
    uploaded_file = st.file_uploader("选择 PDF", type=["pdf"])

    if uploaded_file:
        st.success(f"已选择: {uploaded_file.name}")

        if st.button("🚀 处理文档", type="primary", use_container_width=True):
            with st.spinner("处理中..."):
                try:
                    vectorstore, pages, chunks = process_pdf(uploaded_file)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.document_chunks = chunks
                    st.session_state.pdf_processed = True
                    st.session_state.chat_history = []
                    st.session_state.document_summary = None

                    st.success(f"✅ 完成！{len(pages)} 页，{len(chunks)} 块")
                except Exception as e:
                    st.error(f"失败: {e}")

    st.divider()

    # ===== 全文摘要功能 =====
    st.header("📝 全文摘要")

    if st.session_state.pdf_processed:
        if st.session_state.document_summary:
            st.success("✅ 摘要已生成")
            with st.expander("查看全文摘要", expanded=False):
                st.write(st.session_state.document_summary)
        else:
            chunk_count = len(st.session_state.document_chunks)
            st.info(f"文档共 {chunk_count} 块")

            # 预估时间提醒
            estimated_time = chunk_count * 5  # 假设每块 5 秒
            st.caption(f"⏱️ 预计耗时: {estimated_time // 60} 分 {estimated_time % 60} 秒")

            if st.button("📝 生成全文摘要", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()


                def update_progress(current, total, phase):
                    if phase == "MAP":
                        progress = current / total * 0.8  # MAP 占 80%
                        status_text.text(f"正在分析: {current}/{total} 块")
                    else:
                        progress = 0.8 + 0.2  # REDUCE 占 20%
                        status_text.text("正在整合摘要...")
                    progress_bar.progress(progress)


                try:
                    summary = generate_document_summary(
                        st.session_state.document_chunks,
                        progress_callback=update_progress
                    )
                    st.session_state.document_summary = summary
                    progress_bar.progress(1.0)
                    status_text.text("✅ 完成！")
                    st.rerun()
                except Exception as e:
                    st.error(f"摘要生成失败: {e}")
    else:
        st.warning("请先上传文档")

    st.divider()

    # ===== 状态显示 =====
    st.header("📊 状态")

    if st.session_state.pdf_processed:
        st.success("✅ 文档已加载")
        st.info(f"💬 对话轮数: {len(st.session_state.chat_history)}")
        st.caption(f"记忆窗口: 最近 {MAX_HISTORY_TURNS} 轮")
    else:
        st.warning("⏳ 等待上传")

    # 清空按钮
    if st.session_state.chat_history:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ==================== 右侧：对话区 ====================
with col2:
    st.header("💬 智能问答")

    # 显示对话历史
    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history:
            if st.session_state.pdf_processed:
                st.info("👋 文档已就绪，开始提问吧！支持多轮对话和代词指代。")

                # 示例问题
                st.caption("💡 试试这样问：")
                example_cols = st.columns(2)
                with example_cols[0]:
                    st.caption("• CEO是谁？")
                    st.caption("• 他的背景是什么？（追问）")
                with example_cols[1]:
                    st.caption("• 公司有什么产品？")
                    st.caption("• 第一个产品详细说说（追问）")
            else:
                st.info("👆 请先上传并处理 PDF 文档")
        else:
            for i, chat in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(chat["question"])

                with st.chat_message("assistant"):
                    st.write(chat["answer"])

                    with st.expander("📖 参考来源"):
                        for j, doc in enumerate(chat["sources"]):
                            st.caption(f"**来源 {j + 1}:** {doc.page_content[:150]}...")

    st.divider()

    # 输入区
    if st.session_state.pdf_processed:
        question = st.chat_input("请输入问题（支持追问，如'他是谁'、'详细说说'）...")

        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    try:
                        answer, sources = get_rag_response_with_memory(
                            question,
                            st.session_state.vectorstore,
                            st.session_state.chat_history
                        )
                        st.write(answer)

                        with st.expander("📖 参考来源"):
                            for i, doc in enumerate(sources):
                                st.caption(f"**来源 {i + 1}:** {doc.page_content[:150]}...")

                        # 保存到历史
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer,
                            "sources": sources
                        })
                    except Exception as e:
                        st.error(f"出错: {e}")
    else:
        st.chat_input("请先上传文档...", disabled=True)

# ==================== 页脚 ====================
st.divider()
st.caption("🔒 所有数据本地处理 | v2.0 支持对话记忆和全文摘要")