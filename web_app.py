import pandas as pd
from pptx import Presentation
import io
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
st.title("🤖 企业级 Agentic RAG 智能体 (v8.5)")

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

# ================= 2. 新增：多格式文档解析与切块 =================
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
            for table in doc.tables:
                for row in table.rows:
                    text += " | ".join([cell.text.strip() for cell in row.cells]) + "\n"
                text += "\n"
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(file)
            text = df.to_markdown(index=False)
        elif filename.endswith(".pptx"):
            prs = Presentation(file)
            for i, slide in enumerate(prs.slides):
                text += f"\n--- 第 {i+1} 页幻灯片 ---\n"
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        text += shape.text + "\n"
                    if shape.has_table:
                        for row in shape.table.rows:
                            text += " | ".join([cell.text.strip() for cell in row.cells]) + "\n"
        elif filename.endswith(".txt") or filename.endswith(".md"):
            text = file.read().decode("utf-8")
    except Exception as e:
        st.error(f"解析文件 {file.name} 失败：{e}")
    return text

def process_files(uploaded_files):
    all_chunks = []
    for file in uploaded_files:
        raw_text = extract_text_from_file(file)
        if not raw_text.strip():
            continue
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
    
    # ================= 核心升级：按文件配额提取 =================
    # 取出当前上传的所有文件列表
    files_list = st.session_state.get("current_files", [])
    
    # 按文件分组
    file_groups = {f: [] for f in files_list}
    for chunk in fused:
        # 从咱们之前打的标签里提取文件名
        match = re.search(r"【来源文档：(.*?)】", chunk)
        if match:
            source_file = match.group(1)
            if source_file in file_groups:
                file_groups[source_file].append(chunk)
    
    # 每个文件最多强制取 4 个片段（这样即使有10个文件，也能覆盖到）
    final_chunks = []
    for f, chunks in file_groups.items():
        final_chunks.extend(chunks[:4])
        
    # 如果配额加起来不到 12 个，再从全局融合结果里补足
    if len(final_chunks) < 12:
        for chunk in fused:
            if chunk not in final_chunks:
                final_chunks.append(chunk)
            if len(final_chunks) >= 12:
                break
                
    context = "\n\n---\n\n".join(final_chunks)
    # ========================================================
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
        # 获取当前上传的所有文件名
    file_list = ", ".join(st.session_state.get("current_files", []))
    
    system_prompt = f"""你是一个严谨的企业级文档分析助手。
当前用户上传的文档清单如下：
[{file_list}]

请仔细阅读以下检索到的【文档片段】来回答用户问题。
注意：
1. 如果用户的问题涉及上述清单中的某个文件，但你在【文档片段】中找不到该文件的内容，请明确告诉用户：“由于片段限制，我没有检索到该文件的相关细节”。
2. 如果资料中完全没提到，请直接说不知道。
3. 回答时，请务必标注每一条信息的来源文档名称。

【文档片段】：
{state['context']}"""
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
    "请上传你的文档 (支持 PDF, Word, TXT, MD, PPT, Excel，可多选)", 
    type=["txt", "pdf", "docx", "md", "pptx", "xlsx", "xls"], 
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