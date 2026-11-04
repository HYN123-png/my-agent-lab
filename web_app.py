import streamlit as st
import os
import re
import requests
import numpy as np
import jieba
import docx
import pandas as pd
import base64
from pptx import Presentation
from typing import TypedDict, List
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from rank_bm25 import BM25Okapi
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

st.set_page_config(page_title="企业级 Agentic RAG", page_icon="🤖")
st.title("🤖 企业级 Agentic RAG 智能体 (v8.5)")

load_dotenv()
ds_key = os.getenv("DEEPSEEK_API_KEY")
zhipu_key = os.getenv("ZHIPU_API_KEY")
client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

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
def extract_text_from_image_vlm(image_file):
    image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {"Authorization": f"Bearer {zhipu_key}", "Content-Type": "application/json"}
    data = {
        "model": "glm-4v-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "你是一个企业级文档解析助手。请提取这张图片中的所有文字。如果图片包含表格或图表，请用 Markdown 格式尽可能还原它的结构和核心数据。不要输出任何多余的客套话。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    }
    try:
        res = requests.post(url, headers=headers, json=data)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[图片解析失败：{e}]"

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
            for para in doc.paragraphs: text += para.text + "\n"
            for table in doc.tables:
                for row in table.rows: text += " | ".join([cell.text.strip() for cell in row.cells]) + "\n"
                text += "\n"
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(file)
            text = df.to_markdown(index=False)
        elif filename.endswith(".pptx"):
            prs = Presentation(file)
            for i, slide in enumerate(prs.slides):
                text += f"\n--- 第 {i+1} 页幻灯片 ---\n"
                for shape in slide.shapes:
                    if shape.has_text_frame: text += shape.text + "\n"
                    if shape.has_table:
                        for row in shape.table.rows: text += " | ".join([cell.text.strip() for cell in row.cells]) + "\n"
        elif filename.endswith(".txt") or filename.endswith(".md"):
            text = file.read().decode("utf-8")
        elif filename.endswith((".jpg", ".jpeg", ".png")):
            text = extract_text_from_image_vlm(file)
    except Exception as e:
        st.error(f"解析文件 {file.name} 失败：{e}")
    return text

def process_files(uploaded_files):
    all_chunks = []
    for file in uploaded_files:
        raw_text = extract_text_from_file(file)
        if not raw_text.strip(): continue
        for i in range(0, len(raw_text), 350):
            chunk = raw_text[i:i+400].strip()
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
    if not st.session_state.get("current_files"):
        return {"context": "【无本地文档】当前没有上传任何文件，请直接回答或联网搜索。"}
    col = st.session_state.get("collection")
    bm = st.session_state.get("bm25")
    chunks = st.session_state.get("chunks_global", [])
    
    query = state.get("rewritten_question") or state["question"]
    q_vec = get_embedding(query)
    vector_chunks = col.query(query_embeddings=[q_vec], n_results=20)['documents'][0]
    
    tokenized_q = tokenize(query)
    bm25_scores = bm.get_scores(tokenized_q)
    top_bm25_indices = np.argsort(bm25_scores)[-20:][::-1]
    bm25_chunks = [chunks[i] for i in top_bm25_indices if bm25_scores[i] > 0]
    
    fused = rrf_fuse(vector_chunks, bm25_chunks)
    
    # 雨露均沾：按文件配额
    files_list = st.session_state.get("current_files", [])
    file_groups = {f: [] for f in files_list}
    for chunk in fused:
        match = re.search(r"【来源文档：(.*?)】", chunk)
        if match and match.group(1) in file_groups:
            file_groups[match.group(1)].append(chunk)
            
    final_chunks = []
    for f, chunks_list in file_groups.items(): final_chunks.extend(chunks_list[:4])
    if len(final_chunks) < 12:
        for chunk in fused:
            if chunk not in final_chunks: final_chunks.append(chunk)
            if len(final_chunks) >= 12: break
                
    return {"context": "\n\n---\n\n".join(final_chunks)}
    if "【无本地文档】" in state["context"]:
        return {"is_relevant": "no"}

def grade_node(state: RAGState):
    prompt = f"请判断以下【检索资料】能否回答【用户问题】。如果能，只输出 'yes'；如果不能，只输出 'no'。\n\n【用户问题】：{state['question']}\n\n【检索资料】：{state['context']}"
    res = client.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0
    ).choices[0].message.content.strip().lower()
    return {"is_relevant": "yes" if "yes" in res else "no"}

def rewrite_node(state: RAGState):
    prompt = f"用户的原问题是：{state['question']}。\n由于当前知识库检索不到相关内容，请将原问题重写为一个更简短、包含更多核心关键词的搜索词，直接输出重写后的问题，不要任何废话。"
    new_q = client.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.5
    ).choices[0].message.content.strip()
    return {"rewritten_question": new_q, "retries": state.get("retries", 0) + 1}

def web_search_node(state: RAGState):
    """如果本地知识库搞不定，就去互联网上搜索"""
    query = state.get("rewritten_question") or state["question"]
    
    try:
        # 调用 Tavily 专业搜索
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        results = response.get("results", [])
        
        if not results:
            return {"context": "【网络搜索】未找到相关网页结果。", "is_relevant": "yes"}
            
        # 把网页内容拼成上下文
        web_context = "\n\n".join([
            f"【来源网页：{r['url']}】\n{r['content']}" for r in results
        ])
        
        return {"context": web_context, "is_relevant": "yes"}
    except Exception as e:
        return {"context": f"【网络搜索失败：{e}】", "is_relevant": "yes"}

def generate_node(state: RAGState):
    file_list = ", ".join(st.session_state.get("current_files", []))
    
    # 【核心升级】：判断检索到的信息是来自网络还是本地
    if "【来源网页：" in state["context"]:
        source_info = "网络实时搜索结果（请务必在回答中标注这是网络来源，并附带链接）"
    elif "【无本地文档】" in state["context"]:
        source_info = "无（因为用户没有上传文件）"
    else:
        source_info = "用户上传的本地文档片段（请标注来源文件名）"
        
    system_prompt = f"""你是一个严谨的AI助手。
当前用户上传的本地文件清单：[{file_list if file_list else '无'}]

当前为你提供的参考资料来源于：{source_info}。
请仔细阅读以下【参考资料】来回答用户问题。

注意：
1. 如果资料中完全没提到相关信息，请直接说不知道，绝对不能凭空编造。
2. 必须如实标注每一条信息的出处（是网页网址，还是本地文件名）。

【参考资料】：
{state['context']}"""
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in state.get("history", [])[-6:]: messages.append(msg)
    messages.append({"role": "user", "content": state["question"]})
    
    res = client.chat.completions.create(model="deepseek-chat", messages=messages, stream=True)
    answer = ""
    for chunk in res:
        if chunk.choices[0].delta.content: answer += chunk.choices[0].delta.content
    return {"answer": answer}

def route_after_grade(state: RAGState):
    if state["is_relevant"] == "yes": return "generate"
    if state.get("retries", 0) >= 2: return "web_search"
    return "rewrite"

# ================= 4. 构建图 =================
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)
workflow.add_node("web_search", web_search_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite", "web_search": "web_search"})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)
app_graph = workflow.compile()

# ================= 5. 网页界面 =================
if "messages" not in st.session_state: st.session_state.messages = []

uploaded_files = st.file_uploader(
    "请上传你的文档 (支持 PDF, Word, TXT, MD, PPT, Excel, 图片，可多选)", 
    type=["txt", "pdf", "docx", "md", "pptx", "xlsx", "xls", "jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    current_file_names = [f.name for f in uploaded_files]
    if "current_files" not in st.session_state or st.session_state.current_files != current_file_names:
        with st.spinner("正在解析多份文档并构建知识库..."):
            chunks_global = process_files(uploaded_files)
            if len(chunks_global) == 0:
                st.error("文档解析为空。")
            else:
                chroma_client = chromadb.Client()
                try: chroma_client.delete_collection(name="my_docs")
                except Exception: pass
                
                col = chroma_client.create_collection(name="my_docs")
                for i, chunk in enumerate(chunks_global):
                    col.add(embeddings=[get_embedding(chunk)], documents=[chunk], ids=[f"id_{i}"])
                bm = BM25Okapi([tokenize(c) for c in chunks_global])
                
                st.session_state.collection = col
                st.session_state.bm25 = bm
                st.session_state.chunks_global = chunks_global
                st.session_state.current_files = current_file_names
                st.success(f"成功加载 {len(uploaded_files)} 份文档，共切分 {len(chunks_global)} 个知识块。")

    # ================= 核心修改：将聊天界面独立出来 =================
# 无论有没有上传文件，都渲染聊天历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])

# 无论有没有上传文件，都显示输入框
if prompt := st.chat_input("你可以针对文档提问，或者直接让我上网查资料..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考与检索..."):
            # 准备传给 LangGraph 的输入
            inputs = {
                "question": prompt, 
                "rewritten_question": "", 
                "retries": 0, 
                # 把前几轮的对话传进去，这就是多轮记忆的核心
                "history": st.session_state.messages[:-1] 
            }
            
            result = app_graph.invoke(inputs)
            st.markdown(result["answer"])
            
            # 渲染下载按钮
            if result["answer"]:
                st.download_button(
                    label="📥 下载这份总结 (Markdown格式)",
                    data=result["answer"].encode('utf-8'),
                    file_name="AI_分析报告.md",
                    mime="text/markdown"
                )
                
            with st.expander("🔍 查看本轮内部检索逻辑"):
                if result.get("rewritten_question"):
                    st.write(f"⚠️ **触发了自我纠错！** 重写为：`{result['rewritten_question']}`")
                st.write(result["context"])
                
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})