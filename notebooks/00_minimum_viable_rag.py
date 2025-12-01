# 文件: notebooks/00_minimum_viable_rag.py

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

print("=" * 60)
print("🚀 本地 RAG 最小验证程序 (Anaconda 版)")
print("=" * 60)

# ============ Step 1: 连接 Ollama ============
print("\n[1/4] 正在连接 Ollama...")

try:
    # 如果你用的是 qwen2:1.5b，把下面的 llama3:8b 改成 qwen2:1.5b
    llm = Ollama(
        model="llama3:8b",
        base_url="http://localhost:11434"
    )
    print("✅ Ollama 连接成功！")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    print("\n请检查:")
    print("1. Ollama 是否在运行（系统托盘图标）")
    print("2. 模型是否已下载: ollama list")
    exit(1)

# ============ Step 2: 模拟知识库 ============
print("\n[2/4] 加载模拟知识库...")

fake_knowledge_base = """
公司名称：未来科技有限公司
成立时间：2020年3月15日
主营业务：人工智能解决方案
员工人数：150人
CEO：张三
CTO：李四
办公地点：北京市海淀区
"""

print(f"✅ 知识库加载完成")

# ============ Step 3: 构建 Prompt ============
print("\n[3/4] 构建 Prompt...")

template = """你是一个企业知识库助手。根据以下背景信息回答问题。
如果背景信息中没有相关内容，说"抱歉，没有找到相关信息"。

背景信息：
{context}

问题：{question}

回答："""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

print("✅ Prompt 构建完成")

# ============ Step 4: 测试问答 ============
print("\n[4/4] 开始测试...")
print("=" * 60)

def ask(question):
    formatted = prompt.format(context=fake_knowledge_base, question=question)
    return llm.invoke(formatted)

# 测试几个问题
questions = [
    "公司什么时候成立的？",
    "CTO是谁？",
    "公司的股票代码是什么？"  # 不存在的信息
]

for q in questions:
    print(f"\n🙋 问题: {q}")
    print(f"🤖 回答: {ask(q)}")
    print("-" * 40)

# ============ 交互模式 ============
print("\n" + "=" * 60)
print("进入交互模式 (输入 q 退出)")
print("=" * 60)

while True:
    user_q = input("\n🙋 你的问题: ").strip()
    if user_q.lower() in ['q', 'quit', 'exit']:
        print("👋 再见!")
        break
    if user_q:
        print(f"🤖 回答: {ask(user_q)}")