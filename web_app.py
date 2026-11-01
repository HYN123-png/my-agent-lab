import streamlit as st
import os
import requests
import math
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader

st.set_page_config(page_title="文档问答 Agent", page_icon="📄")
st.title("📄 智能文档分析助手 (v4.0)")

load_dotenv()
ds_key = os.getenv("DEEPSEEK_API_KEY")
zhipu_key = os.getenv("ZHIPU_API_KEY")
client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")

def get_embedding(text):
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    res = requests.post(url, headers={"Authorization": f"Bearer {zhipu_key}"}, json={"model": "embedding-2", "input": text})
    return res.json()["data"][0]["embedding"]

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0: return 0
    return dot_product / (mag1 * mag2)

uploaded_file = st.file_uploader("请上传你的文档 (支持 .txt 或 .pdf)", type=["txt", "pdf"])

if uploaded_file is not None:
    file_text = ""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            file_text += page.extract_text() + "\n"
    else:
        file_text = uploaded_file.read().decode("utf-8")
        
    chunks = [c.strip() for c in file_text.split("\n") if len(c.strip()) > 10]
    
    if len(chunks) == 0:
        st.warning("文件内容似乎为空，或者没有足够长的段落。")
    else:
        st.success(f"文件已成功读取，共切分为 {len(chunks)} 个段落。")
        
        question = st.text_input("请输入关于这份文档的问题：")
        
        if st.button("发送"):
            if question:
                with st.spinner("正在检索文档内容..."):
                    q_vec = get_embedding(question)
                    
                    best_idx, highest_score = -1, -1
                    for i, chunk in enumerate(chunks):
                        chunk_vec = get_embedding(chunk)
                        score = cosine_similarity(q_vec, chunk_vec)
                        if score > highest_score:
                            highest_score = score
                            best_idx = i
                    
                    context = chunks[best_idx]
                
                if highest_score < 0.4:
                    st.warning(f"抱歉，文档中似乎没有相关内容。（最高相似度: {highest_score:.2f}）")
                else:
                    with st.expander("🔍 展开查看检索到的原文档段落"):
                        st.write(context)
                        
                    prompt = f"请严格根据以下文档片段回答用户问题：\n文档片段：{context}\n\n问题：{question}"
                    
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