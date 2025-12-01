# 文件: notebooks/01_pdf_loading.py
# Phase 2: PDF 文档加载与分块

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

print("=" * 60)
print("📄 Phase 2: PDF 文档加载测试")
print("=" * 60)

# ===== Step 1: 加载 PDF =====
print("\n[1/3] 加载 PDF 文件...")

# PDF 文件路径（改成你的文件名）
pdf_path = r"D:\local-rag-chatbot\data\raw\company_intro.pdf"

# 检查文件是否存在
if not os.path.exists(pdf_path):
    print(f"❌ 文件不存在: {pdf_path}")
    print("\n请确保:")
    print("1. 在 data\\raw\\ 目录下放入 PDF 文件")
    print("2. 修改上面的 pdf_path 变量为正确的文件名")
    exit(1)

# 加载 PDF
loader = PyPDFLoader(pdf_path)
pages = loader.load()

print(f"✅ PDF 加载成功！")
print(f"   - 文件: {pdf_path}")
print(f"   - 页数: {len(pages)}")

# 显示第一页内容预览
print(f"\n📖 第一页内容预览 (前500字):")
print("-" * 40)
print(pages[0].page_content[:500])
print("-" * 40)

# ===== Step 2: 文本分块 =====
print("\n[2/3] 文本分块...")

# 创建分块器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每块最大500字符
    chunk_overlap=50,     # 块之间重叠50字符
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)

# 分块
chunks = text_splitter.split_documents(pages)

print(f"✅ 分块完成！")
print(f"   - 原始页数: {len(pages)}")
print(f"   - 分块数量: {len(chunks)}")

# 显示每个块的信息
print(f"\n📦 分块详情:")
print("-" * 40)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {len(chunk.page_content)} 字符")
    # 显示前100字符预览
    preview = chunk.page_content[:100].replace('\n', ' ')
    print(f"   预览: {preview}...")
    print()

# ===== Step 3: 合并为知识库文本 =====
print("\n[3/3] 准备知识库...")

# 把所有块合并成一个字符串（简单版本）
knowledge_base = "\n\n".join([chunk.page_content for chunk in chunks])

print(f"✅ 知识库准备完成！")
print(f"   - 总字符数: {len(knowledge_base)}")

# ===== 保存分块结果供下一步使用 =====
print("\n" + "=" * 60)
print("✅ Phase 2 完成！文档已加载并分块")
print("=" * 60)