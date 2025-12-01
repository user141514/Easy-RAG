# 文件: notebooks/02_pdf_qa.py

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

print("=" * 60)
print("🤖 PDF 知识库问答系统")
print("=" * 60)

# ===== 1. 加载 PDF =====
print("\n[1/4] 加载 PDF...")

pdf_path = r"D:\local-rag-chatbot\data\raw\company_intro.pdf"

if not os.path.exists(pdf_path):
    print(f"❌ 文件不存在: {pdf_path}")
    exit(1)

loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"✅ 加载完成: {len(pages)} 页")

# ===== 2. 文本分块 =====
print("\n[2/4] 文本分块...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)

chunks = text_splitter.split_documents(pages)
print(f"✅ 分块完成: {len(chunks)} 块")

# 合并为知识库
knowledge_base = "\n\n".join([chunk.page_content for chunk in chunks])
print(f"✅ 知识库大小: {len(knowledge_base)} 字符")

# ===== 3. 连接 LLM =====
print("\n[3/4] 连接 Ollama...")

llm = OllamaLLM(
    model="llama3:8b",
    base_url="http://localhost:11434"
)
print("✅ Ollama 连接成功")

# ===== 4. 构建 Prompt =====
print("\n[4/4] 构建问答系统...")

template = """你是一个专业的知识库助手。请根据以下文档内容回答用户的问题。

规则：
1. 只根据文档内容回答，不要编造
2. 如果文档中没有相关信息，说"根据文档，没有找到相关信息"
3. 使用中文回答

文档内容：
{context}

问题：{question}

回答："""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

def ask(question):
    formatted = prompt.format(context=knowledge_base, question=question)
    return llm.invoke(formatted)

# ===== 测试问答 =====
print("\n" + "=" * 60)
print("✅ 系统就绪！")
print("=" * 60)

test_questions = [
    "公司叫什么名字？",
    "CEO是谁？",
    "公司有哪些产品？"
]

print("\n📝 测试问答:")
for q in test_questions:
    print(f"\n🙋 问: {q}")
    print(f"🤖 答: {ask(q)}")
    print("-" * 40)

# 交互模式
print("\n💬 交互模式 (输入 q 退出)")

while True:
    user_q = input("\n🙋 问: ").strip()
    if user_q.lower() in ['q', 'quit', 'exit']:
        print("👋 再见!")
        break
    if user_q:
        print(f"🤖 答: {ask(user_q)}")