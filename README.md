# 🧠 MindForge RAG

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**🔒 100% 本地运行 · 🚀 智能多轮对话 · 📚 PDF 知识库问答**

[快速开始](#-quick-start) •
[技术原理](#-技术原理) •
[功能特性](#-功能特性) •
[项目结构](#-项目结构) •
[贡献指南](#-contributing)

</div>

---

## 📺 Demo

<div align="center">

<!-- Demo GIF 预留位置 -->
<!-- ![MindForge Demo](docs/assets/demo.gif) -->

*🎬 Demo 动图即将上线...*

</div>

---

## ✨ 功能特性

<table>
<tr>
<td width="50%">

### 🔐 完全本地化
- 数据**永不上传**，100% 隐私安全
- 基于 Ollama 本地大模型
- 支持离线运行

### 🧠 智能问答
- **Chain-of-Thought** 深度思考
- 多轮对话记忆，理解上下文
- 自动质量检查与回答优化

</td>
<td width="50%">

### 🔍 智能检索
- **Query Expansion** 多角度检索
- 语义向量匹配 + 智能去重排序
- 支持长文档全文摘要

### 🎨 优雅界面
- 现代化 Web 交互体验
- 实时显示引用来源
- 支持深度/快速双模式

</td>
</tr>
</table>

---

## 🚀 Quick Start

### 前置要求

- Python 3.11+
- [Ollama](https://ollama.com/) 已安装并运行
- 8GB+ 内存（推荐 16GB）

### 一键启动

```bash
# 1. 克隆项目
git clone https://github.com/yourusername/mindforge-rag.git
cd mindforge-rag

# 2. 安装依赖
pip install -r requirements.txt

# 3. 拉取模型 (首次运行)
ollama pull llama3:8b
ollama pull nomic-embed-text

# 4. 启动应用
streamlit run app/streamlit_app_v3.py
