# 🤖 Agentic RAG 企业级文档分析智能体

这是一个基于 RAG（检索增强生成）架构与 LangGraph 搭建的智能文档问答系统。支持上传 PDF/TXT 文档，进行多轮连贯对话，并具备自我纠错与防幻觉能力。

## 🔗 在线体验
[点击这里访问 Web 应用](https://hyn-agent-lab.streamlit.app)

## ✨ 核心特性
- **混合检索架构 (Hybrid Search)**：结合 Chroma 向量语义检索与 BM25 关键词检索，并手写 RRF（倒数排名融合）算法，大幅提升专有名词的召回率。
- **长文本智能切块**：按 400 字符滑动窗口切分，保留上下文重叠，解决传统切块导致的语义断裂问题。
- **Agentic 自我纠错**：基于 LangGraph 构建状态机，引入 Grade（相关性评估）与 Rewrite（查询重写）节点。检索不相关时自动重写问题并重试，有效降低大模型幻觉。
- **多轮对话记忆**：完美支持上下文追问（如：“厨房有什么问题？” -> “那主卧呢？”）。

## 🛠️ 技术栈
- **前端**：Streamlit (流式打字机输出)
- **编排**：LangGraph (状态图与条件边)
- **大模型**：DeepSeek API (生成) / 智谱 API (Embedding)
- **向量库**：ChromaDB
- **检索优化**：Rank-BM25, Jieba (中文分词)

## 🚀 本地运行
1. 克隆仓库：`git clone https://github.com/HYN123-png/my-agent-lab.git`
2. 安装依赖：`uv sync`
3. 配置环境变量：在根目录创建 `.env` 文件，填入 `DEEPSEEK_API_KEY` 和 `ZHIPU_API_KEY`。
4. 运行应用：`uv run streamlit run web_app.py`