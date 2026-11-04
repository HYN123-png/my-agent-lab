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
st.title("🤖 企业级 Agentic RAG 智能体 (v10.0)")

load_dotenv()
ds_key = os.getenv("DEEPSEEK_API_KEY")
zhipu_key = os.getenv("ZHIPU_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None

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

# ================= 2. 多格式解析 =================
def extract_text_from_image_vlm(image_file):
    image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {"Authorization": f"Bearer {zhipu_key}", "Content-Type": "application/json"}
    data = {
        "model": "glm-4v-flash",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "提取图片文字，表格用Markdown还原。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}]
    }
    try:
        res = requests.post(url, headers=headers, json=data)
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[图片解析失败：{e}]"

def extract_text_from_file(file):
    filename = file.name.lower()
    text = ""
    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages: text += (page.extract_text() or "") + "\n"
        elif filename.endswith(".docx"):
            doc = docx.Document(file)
            for para in doc.paragraphs: text += para.text + "\n"
            for table in doc.tables:
                for row in table.rows: text += " | ".join([cell.text.strip() for cell in row.cells]) + "\n"
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
            text = df.to_markdown(index=False)
        elif filename.endswith(".pptx"):
            prs = Presentation(file)
            for i, slide in enumerate(prs.slides):
                text += f"\n--- 幻灯片 {i+1} ---\n"
                for shape in slide.shapes:
                    if shape.has_text_frame: text += shape.text + "\n"
                    if shape.has_table:
                        for row in shape.table.rows: text += " | ".join([cell.text.strip() for cell in row.cells]) + "\n"
        elif filename.endswith((".txt", ".md")):
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

# ================= 3. 状态定义与核心节点 =================
class RAGState(TypedDict):
    question: str
    standalone_query: str
    rewritten_question: str
    context: str
    is_relevant: str
    retries: int
    answer: str
    history: List[dict]

# 新增：将多轮对话合并为独立问题，解决上下文断片
def rewrite_query_with_history(state: RAGState):
    history = state.get("history", [])
    if not history:
        return {"standalone_query": state["question"]}
    
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]])
    prompt = f"根据以下对话历史，将用户的最后一个问题重写为一个独立的、包含完整上下文的查询语句。只输出重写后的句子，不要任何废话。\n\n历史：\n{history_str}\n最后问题：{state['question']}\n独立查询："
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0)
        q = res.choices[0].message.content.strip()
        return {"standalone_query": q if q else state["question"]}
    except:
        return {"standalone_query": state["question"]}

def retrieve_node(state: RAGState):
    if not st.session_state.get("current_files"):
        return {"context": "【无本地文档】当前没有上传任何文件。"}
    
    col = st.session_state.get("collection")
    bm = st.session_state.get("bm25")
    chunks = st.session_state.get("chunks_global", [])
    
    query = state.get("rewritten_question") or state.get("standalone_query") or state["question"]
    q_vec = get_embedding(query)
    
    vector_chunks = col.query(query_embeddings=[q_vec], n_results=20)['documents'][0]
    tokenized_q = tokenize(query)
    bm25_scores = bm.get_scores(tokenized_q)
    top_bm25_indices = np.argsort(bm25_scores)[-20:][::-1]
    bm25_chunks = [chunks[i] for i in top_bm25_indices if bm25_scores[i] > 0]
    
    fused = rrf_fuse(vector_chunks, bm25_chunks)
    
    # 配额提取
    file_groups = {f: [] for f in st.session_state.get("current_files", [])}
    for chunk in fused:
        match = re.search(r"【来源文档：(.*?)】", chunk)
        if match and match.group(1) in file_groups:
            file_groups[match.group(1)].append(chunk)
            
    final_chunks = []
    for f, chunks_list in file_groups.items(): final_chunks.extend(chunks_list[:4])
    for chunk in fused:
        if chunk not in final_chunks: final_chunks.append(chunk)
        if len(final_chunks) >= 12: break
            
    return {"context": "\n\n---\n\n".join(final_chunks)}

def grade_node(state: RAGState):
    if "【无本地文档】" in state["context"]:
        return {"is_relevant": "no"}
    prompt = f"请判断【资料】能否回答【问题】。如果涉及主观总结/写报告，只要相关就输出'yes'。只输出'yes'或'no'。\n\n【问题】：{state['question']}\n【资料】：{state['context'][:2000]}"
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0)
        return {"is_relevant": "yes" if "yes" in res.choices[0].message.content.lower() else "no"}
    except:
        return {"is_relevant": "no"}

def rewrite_node(state: RAGState):
    prompt = f"原问题：{state['question']}。由于检索不到，请重写为更简短的搜索词，只输出重写结果。"
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.5)
        return {"rewritten_question": res.choices[0].message.content.strip(), "retries": state.get("retries", 0) + 1}
    except:
        return {"rewritten_question": state["question"], "retries": state.get("retries", 0) + 1}

def web_search_node(state: RAGState):
    if not tavily_client:
        return {"context": "【网络搜索接口未配置】请检查 TAVILY_API_KEY。", "is_relevant": "yes"}
    query = state.get("rewritten_question") or state.get("standalone_query") or state["question"]
    try:
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        valid_results = [r for r in response.get("results", []) if r.get("content")]
        if not valid_results:
            return {"context": f"【网络搜索】未找到关于 '{query}' 的有效结果。", "is_relevant": "yes"}
        web_context = "\n\n".join([f"【来源网页：{r['url']}】\n{r['content']}" for r in valid_results])
        return {"context": web_context, "is_relevant": "yes"}
    except Exception as e:
        return {"context": f"【网络搜索报错：{str(e)}】", "is_relevant": "yes"}

def research_node(state: RAGState):
    safe_context = state['context'][:8000]
    if "【网络搜索" in state["context"] and len(safe_context) < 100: # 拦截报错
        return {"answer": f"⚠️ {safe_context}"}
        
    # 联网短查询直接吐结果
    if "【来源网页：" in state["context"]:
        return {"answer": f"[[DIRECT]]\n根据联网搜索结果，为您找到以下信息：\n\n{safe_context}"}
        
    # 本地文档提炼
    prompt = f"提取核心事实整理成大纲。\n资料：\n{safe_context}"
    messages = [{"role": "system", "content": prompt}] + state.get("history", [])[-4:] + [{"role": "user", "content": state["question"]}]
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=messages)
        draft = res.choices[0].message.content
        return {"answer": draft if draft else f"提取失败，原文如下：\n{safe_context}"}
    except Exception as e:
        return {"answer": f"⚠️ 总结失败（{e}），原文如下：\n{safe_context}"}

def writer_node(state: RAGState):
    ans = state.get("answer", "")
    if not ans: return {"answer": "未生成任何内容。"}
    if "[[DIRECT]]" in ans or "⚠️" in ans:
        return {"answer": ans.replace("[[DIRECT]]\n", "")}
        
    prompt = f"作为商业报告主编，请排版以下大纲。\n问题：{state['question']}\n大纲：{ans}"
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}])
        final = res.choices[0].message.content
        return {"answer": final if final else ans}
    except:
        return {"answer": ans}

def route_after_grade(state: RAGState):
    if "【无本地文档】" in state["context"]: return "web_search"
    if state["is_relevant"] == "yes": return "research"
    if state.get("retries", 0) >= 1: return "web_search"
    return "rewrite"

# ================= 4. 构建状态图 =================
workflow = StateGraph(RAGState)
workflow.add_node("rewrite_query", rewrite_query_with_history)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("research", research_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("rewrite_query")
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", route_after_grade, {"research": "research", "rewrite": "rewrite", "web_search": "web_search"})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("web_search", "research")
workflow.add_edge("research", "writer")
workflow.add_edge("writer", END)
app_graph = workflow.compile()

# ================= 5. 前端界面 =================
if "messages" not in st.session_state: st.session_state.messages = []

uploaded_files = st.file_uploader("上传文档 (PDF, Word, Excel, PPT, 图片)", type=["txt", "pdf", "docx", "md", "pptx", "xlsx", "xls", "jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    current_file_names = [f.name for f in uploaded_files]
    if "current_files" not in st.session_state or st.session_state.current_files != current_file_names:
        with st.spinner("正在解析多份文档..."):
            chunks_global = process_files(uploaded_files)
            if chunks_global:
                chroma_client = chromadb.Client()
                try: chroma_client.delete_collection(name="my_docs")
                except: pass
                col = chroma_client.create_collection(name="my_docs")
                for i, chunk in enumerate(chunks_global):
                    col.add(embeddings=[get_embedding(chunk)], documents=[chunk], ids=[f"id_{i}"])
                st.session_state.collection = col
                st.session_state.bm25 = BM25Okapi([tokenize(c) for c in chunks_global])
                st.session_state.chunks_global = chunks_global
                st.session_state.current_files = current_file_names
                st.success(f"成功加载 {len(uploaded_files)} 份文档。")

# 历史渲染
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# 聊天输入
if prompt := st.chat_input("你可以针对文档提问，或者直接让我上网查资料..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考..."):
            try:
                # 调用图
                events = app_graph.stream({"question": prompt, "rewritten_question": "", "retries": 0, "history": st.session_state.messages[:-1]}, stream_mode="values")
                final_result = None
                for event in events: final_result = event
                
                # 绝对兜底
                final_text = final_result.get("answer", "") if final_result else ""
                if not final_text or not final_text.strip():
                    final_text = "⚠️ 抱歉，系统未能生成有效回答。请尝试换种问法或检查网络。"

                st.markdown(final_text)
                
                st.download_button("📥 下载这份总结", data=final_text.encode('utf-8'), file_name="Report.md", mime="text/markdown")
                
                with st.expander("🔍 查看底层检索逻辑"):
                    st.write(f"**重写后的独立搜索词**：{final_result.get('standalone_query', '无')}")
                    st.write(f"**当前上下文**：\n{final_result.get('context', '无')}")
                    
            except Exception as e:
                final_text = f"❌ 执行出错：{str(e)}"
                st.markdown(final_text)
                
    st.session_state.messages.append({"role": "assistant", "content": final_text})