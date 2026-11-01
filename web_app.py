import streamlit as st
import os
import requests
import math
from openai import OpenAI
from dotenv import load_dotenv

st.set_page_config(page_title="我的专属 AI 助手", page_icon="🤖")
st.title("🤖 我的专属知识库 AI 助手 (v2.0 流式版)")

# 加载本地环境变量（用于本地测试），云端会自动读取 Secrets
load_dotenv()
ds_key = os.getenv("DEEPSEEK_API_KEY")
zhipu_key = os.getenv("ZHIPU_API_KEY")

# 初始化 DeepSeek 客户端
client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")

@st.cache_data
def init_data():
    documents = [
        "李四是我们公司的测试工程师。", 
        "他的主要工作是对新产品进行功能测试和性能测试。",
        "今天天气真不错，适合出去游玩。",
        "苹果是一种非常好吃的水果，富含维生素。"
    ]
    def get_emb(text):
        url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
        res = requests.post(url, headers={"Authorization": f"Bearer {zhipu_key}"}, json={"model": "embedding-2", "input": text})
        return res.json()["data"][0]["embedding"]
    
    doc_vecs = [get_emb(doc) for doc in documents]
    return documents, doc_vecs

documents, doc_vectors = init_data()

question = st.text_input("请输入你的问题：", placeholder="例如：这水果好吃吗？")

if st.button("发送"):
    if question:
        # 1. 检索过程（和之前一样）
        with st.spinner("正在检索知识库..."):
            url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
            res = requests.post(url, headers={"Authorization": f"Bearer {zhipu_key}"}, json={"model": "embedding-2", "input": question})
            q_vec = res.json()["data"][0]["embedding"]
            
            best_idx, highest_score = -1, -1
            for i, doc_vec in enumerate(doc_vectors):
                score = sum(a*b for a,b in zip(q_vec, doc_vec)) / (math.sqrt(sum(a*a for a in q_vec)) * math.sqrt(sum(b*b for b in doc_vec)) or 1)
                if score > highest_score:
                    highest_score = score
                    best_idx = i
                    
            context = documents[best_idx]
        
        # 2. 判断与生成
        if highest_score < 0.4:
            st.warning(f"抱歉，知识库中没有找到相关内容。（最高相似度: {highest_score:.2f}）")
        else:
            st.info(f"🔍 检索到的参考资料：{context}")
            
            prompt = f"请根据以下资料回答用户问题：\n资料：{context}"
            
            # 这就是流式输出的魔法！
            # 我们调用 OpenAI 库的 create 方法，并加上 stream=True
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question}
                ],
                stream=True # 开启流式输出
            )
            
            # 定义一个生成器函数，把收到的文字一块块吐出来
            def stream_generator():
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            
            # Streamlit 专属的流式展示组件
            st.success("🤖 AI 回答：")
            st.write_stream(stream_generator)