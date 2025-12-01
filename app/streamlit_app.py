# 文件: app/streamlit_app.py
# 本地 RAG 知识库问答 - 修复文件锁定问题

import streamlit as st
import tempfile
import os
import shutil
import gc
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="本地知识库问答",
    page_icon="📚",
    layout="wide"
)

# ==================== 常量配置 ====================
VECTORDB_BASE_PATH = r"D:\local-rag-chatbot\data\vectordb"
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3:8b"
EMBEDDING_MODEL = "nomic-embed-text"

# ==================== 初始化 Session State ====================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "current_db_path" not in st.session_state:
    st.session_state.current_db_path = None


# ==================== 核心函数 ====================

@st.cache_resource
def get_llm():
    """初始化 LLM"""
    return OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL
    )


@st.cache_resource
def get_embeddings():
    """初始化 Embedding 模型"""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )


def cleanup_old_vectorstore():
    """清理旧的向量数据库连接"""
    if st.session_state.vectorstore is not None:
        try:
            # 清除引用
            st.session_state.vectorstore = None
            # 强制垃圾回收
            gc.collect()
            # 等待文件释放
            time.sleep(0.5)
        except Exception as e:
            st.warning(f"清理旧数据库时出现警告: {e}")


def process_pdf(uploaded_file):
    """处理上传的 PDF 文件"""

    # 1. 清理旧连接
    cleanup_old_vectorstore()

    # 2. 保存上传文件到临时位置
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 3. 加载 PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # 4. 分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        )
        chunks = text_splitter.split_documents(pages)

        # 5. 生成唯一的数据库路径（避免锁定问题）
        import uuid
        db_path = f"{VECTORDB_BASE_PATH}_{uuid.uuid4().hex[:8]}"

        # 6. 清理可能存在的旧目录
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

        # 7. 创建向量数据库
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=db_path
        )

        # 8. 保存当前数据库路径
        st.session_state.current_db_path = db_path

        return vectorstore, len(pages), len(chunks)

    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except:
            pass


def get_rag_response(question: str, vectorstore, top_k: int = 3):
    """RAG 问答"""

    # 检索相关文档
    retrieved_docs = vectorstore.similarity_search(question, k=top_k)

    # 拼接上下文
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    # 构建 Prompt
    template = """你是一个专业的知识库助手。请根据以下文档内容回答用户的问题。

【参考文档】
{context}

【用户问题】
{question}

【回答要求】
1. 只根据参考文档回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明
3. 回答要简洁、准确
4. 使用中文回答

回答："""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    formatted_prompt = prompt.format(context=context, question=question)
    response = get_llm().invoke(formatted_prompt)

    return response, retrieved_docs


# ==================== 页面布局 ====================

st.title("📚 本地知识库问答助手")
st.caption("上传 PDF 文档，基于文档内容进行智能问答 | 数据完全本地处理")

col1, col2 = st.columns([1, 2])

# ==================== 左侧：文件上传 ====================
with col1:
    st.header("📁 文档上传")

    uploaded_file = st.file_uploader(
        "选择 PDF 文件",
        type=["pdf"],
        help="上传 PDF 文档"
    )

    if uploaded_file is not None:
        st.success(f"已选择: {uploaded_file.name}")

        if st.button("🚀 处理文档", type="primary", use_container_width=True):
            with st.spinner("正在处理文档..."):
                try:
                    vectorstore, num_pages, num_chunks = process_pdf(uploaded_file)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_processed = True
                    st.session_state.chat_history = []

                    st.success(f"✅ 处理完成！页数: {num_pages}，分块: {num_chunks}")
                except Exception as e:
                    st.error(f"处理失败: {str(e)}")

    st.divider()
    st.header("📊 状态")

    if st.session_state.pdf_processed:
        st.success("✅ 文档已加载")
    else:
        st.warning("⏳ 等待上传")

    # 清空对话按钮
    if st.session_state.chat_history:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ==================== 右侧：对话区 ====================
with col2:
    st.header("💬 智能问答")

    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history:
            st.info("👆 请先上传并处理 PDF 文档")
        else:
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat["question"])

                with st.chat_message("assistant"):
                    st.write(chat["answer"])

                    with st.expander("📖 参考来源"):
                        for i, doc in enumerate(chat["sources"]):
                            st.caption(f"**来源 {i + 1}:** {doc.page_content[:150]}...")

    st.divider()

    if st.session_state.pdf_processed:
        question = st.chat_input("请输入问题...")

        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    try:
                        answer, sources = get_rag_response(
                            question,
                            st.session_state.vectorstore
                        )
                        st.write(answer)

                        with st.expander("📖 参考来源"):
                            for i, doc in enumerate(sources):
                                st.caption(f"**来源 {i + 1}:** {doc.page_content[:150]}...")

                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer,
                            "sources": sources
                        })
                    except Exception as e:
                        st.error(f"出错: {str(e)}")
    else:
        st.chat_input("请先上传文档...", disabled=True)

st.divider()
st.caption("🔒 所有数据本地处理")