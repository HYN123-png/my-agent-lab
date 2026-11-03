import streamlit as st
import os
import re
import requests
import numpy as np
import jieba
import docx
from typing import TypedDict, List
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from rank_bm25 import BM25Okapi
from langgraph.graph import StateGraph, END

st.set_page_config(page_title="企业级 Agentic RAG", page_icon="🤖")
st.title("🤖 企业级 Agentic RAG 智能体 (v8.2)")

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

# ================= 2. 多格式文档解析 =================
def extract_text_from_file(file):
    filename = file.name.lower()
    text = ""
    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        elif filename.endswith(".docx"):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif filename.endswith(".txt") or filename.endswith(".md"):
            text = file.read().decode("utf-8")
    except Exception as e:
        st.error(f"解析文件 {file.name} 失败：{e}")
    return text

def process_files(uploaded_files):
    all_chunks = []
    for file in uploaded_files:
        raw_text = extract_text_from_file(file)
        if not raw_text.strip(): continue
        chunk_size = 400
        chunk_overlap = 50
        for i in range(0, len(raw_text), chunk_size - chunk_overlap):
            chunk = raw_text[i:i+chunk_size].strip()
            if len(chunk) > 20:
                all_chunks.append(f"【来源文档：{file.name}】\n{chunk}")
    return all_chunks

# ================= 3. LangGraph 节点 =================
class RAGState(TypedDict):
    question: str
    rewritten_question: str
    context: str
    is_relevant: str
    retries: int
    answer: str
    history: List[dict]

def retrieve_node(state: RAGState):
    # 关键修复：直接从 session_state 读取建好的数据库
    col = st.session_state.get("collection")
    bm = st.session_state.get("bm25")
    chunks = st.session_state.get("chunks_global", [])
    
    query = state.get("rewritten_question") or state["question"]
    q_vec = get_embedding(query)
    
    # 向量检索
    vector_chunks = col.query(query_embeddings=[q_vec], n_results=20)['documents'][0]
    
    # BM25 检索
    tokenized_q = tokenize(query)
    bm25_scores = bm.get_scores(tokenized_q)
    top_bm25_indices = np.argsort(bm25_scores)[-20:][::-1]
    bm25_chunks = [chunks[i] for i in top_bm25_indices if bm25_scores[i] > 0]
    
    fused = rrf_fuse(vector_chunks, bm25_chunks)
    context = "\n\n---\n\n".join(fused[:6])
    return {"context": context}

def grade_node(state: RAGState):
    prompt = f"请判断以下【检索资料】能否回答【用户问题】。如果能，只输出 'yes'；如果不能，只输出 'no'。\n\n【用户问题】：{state['question']}\n\n【检索资料】：{state['context']}"
    res = client.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0
    ).choices[0].message.content.strip().lower()
    return {"is_relevant": "yes" if "yes" in res else "no"}

def rewrite_node(state: RAGState):
    prompt = f"用户的问题：{state['question']}，当前检索不到内容。请重写为一个更简短、包含核心关键词的搜索词，只输出重写后的问题。"
    new_q = client.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.5
    ).choices[0].message.content.strip()
    return {"rewritten_question": new_q, "retries": state.get("retries", 0) + 1}

def generate_node(state: RAGState):
    system_prompt = f"请根据以下文档片段回答用户问题。如果资料中没提到，请直接说不知道。请尽量引用来源文档名称。\n\n文档片段：\n{state['context']}"
    messages = [{"role": "system", "content": system_prompt}]
    for msg in state.get("history", [])[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": state["question"]})
    
    res = client.chat.completions.create(model="deepseek-chat", messages=messages, stream=True)
    answer = ""
    for chunk in res:
        if chunk.choices[0].delta.content:
            answer += chunk.choices[0].delta.content
    return {"answer": answer}

def route_after_grade(state: RAGState):
    if state["is_relevant"] == "yes" or state.get("retries", 0) >= 2:
        return "generate"
    return "rewrite"

# ================= 4. 构建图 =================
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite"})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)
app_graph = workflow.compile()

# ================= 5. 网页界面 =================
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader(
    "请上传你的文档 (支持 PDF, Word, TXT, MD，可多选)", 
    type=["txt", "pdf", "docx", "md"], 
    accept_multiple_files=True
)

if uploaded_files:
    current_file_names = [f.name for f in uploaded_files]
    # 如果文件变了，重新建库并存入 session_state
    if "current_files" not in st.session_state or st.session_state.current_files != current_file_names:
        with st.spinner("正在解析多份文档并构建知识库..."):
            chunks_global = process_files(uploaded_files)
            
            if len(chunks_global) == 0:
                st.error("文档解析为空。")
            else:
                chroma_client = chromadb.Client()
                
                # 1. 先尝试删除已存在的集合，防止重复创建报错
                try:
                    chroma_client.delete_collection(name="my_docs")
                except Exception:
                    pass  # 如果集合不存在，会报错，这里直接忽略
                
                # 2. 再重新创建全新的集合
                col = chroma_client.create_collection(name="my_docs")
                
                for i, chunk in enumerate(chunks_global):
                    col.add(embeddings=[get_embedding(chunk)], documents=[chunk], ids=[f"id_{i}"])
                    
                bm = BM25Okapi([tokenize(c) for c in chunks_global])
                
                # 核心修复：把建好的对象保存起来，防止刷新丢失
                st.session_state.collection = col
                st.session_state.bm25 = bm
                st.session_state.chunks_global = chunks_global
                st.session_state.current_files = current_file_names
                st.success(f"成功加载 {len(uploaded_files)} 份文档，共切分 {len(chunks_global)} 个知识块。")

    # 渲染历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("你可以针对这些文档连续提问..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent 正在多文档中检索与思考..."):
                inputs = {
                    "question": prompt,
                    "rewritten_question": "",
                    "retries": 0,
                    "history": st.session_state.messages[:-1]
                }
                result = app_graph.invoke(inputs)
                
                final_answer = result["answer"]
                st.markdown(final_answer)
                
                with st.expander("🔍 查看本轮内部检索逻辑"):
                    if result.get("rewritten_question"):
                        st.write(f"⚠️ **触发了自我纠错！** 重写为：`{result['rewritten_question']}`")
                    st.write(result["context"])
                
        st.session_state.messages.append({"role": "assistant", "content": final_answer})