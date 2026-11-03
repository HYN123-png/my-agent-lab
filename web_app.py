import streamlit as st
import os
import re
import requests
import numpy as np
import jieba
from typing import TypedDict, List
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from rank_bm25 import BM25Okapi
from langgraph.graph import StateGraph, END

st.set_page_config(page_title="Agentic RAG 智能体", page_icon="🤖")
st.title("🤖 Agentic RAG 文档助手 (v8.0)")

load_dotenv()
ds_key = os.getenv("DEEPSEEK_API_KEY")
zhipu_key = os.getenv("ZHIPU_API_KEY")
client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")

# ================= 1. 基础函数 =================
def get_embedding(text):
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    res = requests.post(url, headers={"Authorization": f"Bearer {zhipu_key}"}, json={"model": "embedding-2", "input": text})
    return res.json()["data"][0]["embedding"]

def tokenize(text):
    words = jieba.lcut(text.lower())
    return [w for w in words if w.strip() and not re.match(r'^[^\w\s]+$', w)]

def rrf_fuse(list1, list2, k=60):
    scores = {}
    for rank, doc in enumerate(list1):
        scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank + 1)
    for rank, doc in enumerate(list2):
        scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank + 1)
    return [doc for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

# ================= 2. 定义 LangGraph 的状态与节点 =================
class RAGState(TypedDict):
    question: str
    rewritten_question: str
    context: str
    is_relevant: str  # 用 'yes' 或 'no' 表示
    retries: int
    answer: str
    history: List[dict]

# 全局检索器（在文件上传后赋值）
collection = None
bm25 = None
chunks_global = []

def retrieve_node(state: RAGState):
    query = state.get("rewritten_question") or state["question"]
    
    # 混合检索（向量 + BM25 + RRF）
    q_vec = get_embedding(query)
    vector_chunks = collection.query(query_embeddings=[q_vec], n_results=20)['documents'][0]
    
    tokenized_q = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_q)
    top_bm25_indices = np.argsort(bm25_scores)[-20:][::-1]
    bm25_chunks = [chunks_global[i] for i in top_bm25_indices if bm25_scores[i] > 0]
    
    fused = rrf_fuse(vector_chunks, bm25_chunks)
    top_6 = fused[:6]
    context = "\n\n---\n\n".join(top_6)
    
    return {"context": context}

def grade_node(state: RAGState):
    # 让大模型判断检索到的资料是否与问题相关
    prompt = f"请判断以下【检索资料】能否回答【用户问题】。如果能，只输出 'yes'；如果不能，只输出 'no'。\n\n【用户问题】：{state['question']}\n\n【检索资料】：{state['context']}"
    res = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    ).choices[0].message.content.strip().lower()
    
    is_rel = "yes" if "yes" in res else "no"
    return {"is_relevant": is_rel}

def rewrite_node(state: RAGState):
    # 如果资料不相关，让大模型重写问题，提取更精准的关键词
    prompt = f"用户的原问题是：{state['question']}。\n由于当前知识库检索不到相关内容，请将原问题重写为一个更简短、包含更多核心关键词的搜索词，直接输出重写后的问题，不要任何废话。"
    new_q = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    ).choices[0].message.content.strip()
    
    return {
        "rewritten_question": new_q,
        "retries": state.get("retries", 0) + 1
    }

def generate_node(state: RAGState):
    # 融合历史记录和上下文生成答案
    system_prompt = f"请根据以下文档片段回答用户问题。如果资料中没提到，请直接说不知道。\n\n文档片段：\n{state['context']}"
    
    messages = [{"role": "system", "content": system_prompt}]
    # 加入最近 3 轮的历史对话（多轮记忆）
    for msg in state.get("history", [])[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": state["question"]})
    
    res = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=True
    )
    
    # 这里为了简化状态图，我们直接把完整答案生成出来
    answer = ""
    for chunk in res:
        if chunk.choices[0].delta.content:
            answer += chunk.choices[0].delta.content
            
    return {"answer": answer}

def route_after_grade(state: RAGState):
    if state["is_relevant"] == "yes":
        return "generate"
    if state.get("retries", 0) >= 2:
        return "generate" # 重试两次还不行，就直接强行生成（或拒答）
    return "rewrite"

# ================= 3. 构建图 =================
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", route_after_grade, {
    "generate": "generate",
    "rewrite": "rewrite"
})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

app_graph = workflow.compile()

# ================= 4. Streamlit 网页界面 =================
# 初始化 session_state 用于保存多轮对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("请上传你的文档", type=["txt", "pdf"])

if uploaded_file is not None:
    # 处理文档...（这里省略了部分繁琐的解析，和之前一样）
    file_text = ""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages: file_text += page.extract_text() + "\n"
    else:
        file_text = uploaded_file.read().decode("utf-8")
        
    chunks_global = [file_text[i:i+400].strip() for i in range(0, len(file_text), 350) if len(file_text[i:i+400].strip()) > 20]
    
    @st.cache_resource(show_spinner="正在构建数据库...")
    def build_db(chunks_list):
        chroma_client = chromadb.Client()
        col = chroma_client.create_collection(name="my_docs")
        for i, chunk in enumerate(chunks_list):
            col.add(embeddings=[get_embedding(chunk)], documents=[chunk], ids=[f"id_{i}"])
        bm25_obj = BM25Okapi([tokenize(c) for c in chunks_list])
        return col, bm25_obj

    collection, bm25 = build_db(chunks_global)
    st.success(f"文档已加载，共 {len(chunks_global)} 个段落。")

    # 渲染历史对话
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 聊天输入框
    if prompt := st.chat_input("针对这份文档，你可以连续提问..."):
        # 显示用户提问
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent 正在思考（检索、评估、自我纠错中）..."):
                # 调用 LangGraph
                inputs = {
                    "question": prompt,
                    "rewritten_question": "",
                    "retries": 0,
                    "history": st.session_state.messages[:-1] # 历史记录
                }
                result = app_graph.invoke(inputs)
                
                final_answer = result["answer"]
                st.markdown(final_answer)
                
                # 展开查看检索到的上下文
                with st.expander("🔍 查看本轮内部检索逻辑"):
                    if result.get("rewritten_question"):
                        st.write(f"⚠️ **触发了自我纠错！** 原问题检索失败，Agent 已将问题重写为：`{result['rewritten_question']}`")
                    st.write(result["context"])
                
        # 保存 AI 回答到历史
        st.session_state.messages.append({"role": "assistant", "content": final_answer})