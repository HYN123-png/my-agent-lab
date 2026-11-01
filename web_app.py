import streamlit as st
import os
import requests
import math
from dotenv import load_dotenv

# 设置网页的标题和图标
st.set_page_config(page_title="我的专属 AI 助手", page_icon="🤖")
st.title("🤖 我的专属知识库 AI 助手")
st.markdown("这是一个基于 RAG 和 大模型构建的智能助手，你可以问它关于【李四】或者【苹果】的问题。")

load_dotenv()
ds_key = os.getenv("DEEPSEEK_API_KEY")
zhipu_key = os.getenv("ZHIPU_API_KEY")

@st.cache_data # 这个装饰器可以让网页记住算好的向量，不用每次都重新算
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

# 网页上的输入框
question = st.text_input("请输入你的问题：", placeholder="例如：这水果好吃吗？")

if st.button("发送"):
    if question:
        with st.spinner("AI 正在思考中..."):
            # 1. 检索
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
                ds_res = requests.post(
                    "https://api.deepseek.com/chat/completions", 
                    headers={"Authorization": f"Bearer {ds_key}"}, 
                    json={"model": "deepseek-chat", "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": question}]}
                )
                answer = ds_res.json()["choices"][0]["message"]["content"]
                st.success(answer)