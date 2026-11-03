import streamlit as st
import os
import re
import requests
import numpy as np
import jieba
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="企业级混合检索 Agent", page_icon="🚀")
st.title("🚀 企业级文档分析助手 (v7.2 稳定版)")

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

# ================= 2. 网页交互与数据处理 =================
uploaded_file = st.file_uploader("请上传你的文档 (支持 .txt 或 .pdf)", type=["txt", "pdf"])

if uploaded_file is not None:
    file_text = ""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            file_text += page.extract_text() + "\n"
    else:
        file_text = uploaded_file.read().decode("utf-8")
        
    chunk_size = 400
    chunk_overlap = 50
    chunks = []
    for i in range(0, len(file_text), chunk_size - chunk_overlap):
        chunk = file_text[i:i + chunk_size].strip()
        if len(chunk) > 20:
            chunks.append(chunk)
    
    if len(chunks) == 0:
        st.warning("文件内容太短或为空。")
    else:
        st.success(f"文件读取成功，共切分为 {len(chunks)} 个长段落。")
        
        @st.cache_resource(show_spinner="正在将文档写入向量数据库...")
        def build_vector_db(file_name, chunks_list):
            chroma_client = chromadb.Client()
            collection = chroma_client.create_collection(name="my_docs")
            for i, chunk in enumerate(chunks_list):
                vector = get_embedding(chunk)
                collection.add(embeddings=[vector], documents=[chunk], ids=[f"id_{i}"])
            return collection

        collection = build_vector_db(uploaded_file.name, chunks)
        
        @st.cache_resource
        def build_bm25(chunks_list):
            tokenized_chunks = [tokenize(c) for c in chunks_list]
            return BM25Okapi(tokenized_chunks)
            
        bm25 = build_bm25(chunks)

        question = st.text_input("请输入关于这份文档的问题：")
        
        if st.button("发送"):
            if question:
                with st.spinner("正在进行混合检索..."):
                    
                    # 1. 语义检索
                    q_vec = get_embedding(question)
                    vector_results = collection.query(query_embeddings=[q_vec], n_results=20)
                    vector_chunks = vector_results['documents'][0]
                    
                    # 2. 关键词检索 (BM25)
                    tokenized_q = tokenize(question)
                    bm25_scores = bm25.get_scores(tokenized_q)
                    top_bm25_indices = np.argsort(bm25_scores)[-20:][::-1]
                    bm25_chunks = [chunks[i] for i in top_bm25_indices if bm25_scores[i] > 0]
                    
                    # 3. RRF 融合
                    fused_chunks = rrf_fuse(vector_chunks, bm25_chunks)
                    
                    # 直接取融合后的前 6 个
                    final_chunks = fused_chunks[:6]
                        
                    context = "\n\n---\n\n".join(final_chunks)
                
                if not context:
                    st.warning("抱歉，没有找到相关内容。")
                else:
                    with st.expander("🔍 展开查看混合检索后的 Top-6 精选段落"):
                        st.write(context)
                        
                    prompt = f"请仔细阅读以下文档片段，并全面、完整地回答用户问题。如果需要罗列多个事项，请尽量穷尽片段中的所有细节。\n\n文档片段：\n{context}\n\n问题：{question}"
                    
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "system", "content": prompt}],
                        stream=True
                    )
                    
                    def stream_generator():
                        for chunk in response:
                            if chunk.choices[0].delta.content is not None:
                                yield chunk.choices[0].delta.content
                    
                    st.success("🤖 AI 回答：")
                    st.write_stream(stream_generator)