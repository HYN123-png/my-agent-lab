import streamlit as st
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb # 引入向量数据库

st.set_page_config(page_title="文档问答 Agent", page_icon="🧠")
st.title("🧠 企业级文档分析助手 (v6.0)")

load_dotenv()
ds_key = os.getenv("DEEPSEEK_API_KEY")
zhipu_key = os.getenv("ZHIPU_API_KEY")
client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")

# 智谱 Embedding 接口
def get_embedding(text):
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    res = requests.post(url, headers={"Authorization": f"Bearer {zhipu_key}"}, json={"model": "embedding-2", "input": text})
    return res.json()["data"][0]["embedding"]

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
        st.warning("文件内容太短或为空。")
    else:
        st.success(f"文件读取成功，共切分为 {len(chunks)} 个段落。")
        
        # ================= 核心升级：构建 ChromaDB =================
        # 为了避免重复计算，我们利用 Streamlit 的缓存机制，只在文件改变时建库
        @st.cache_resource(show_spinner="正在将文档写入向量数据库...")
        def build_vector_db(file_name, chunks_list):
            # 创建一个内存中的临时数据库（真实业务中会存到硬盘）
            chroma_client = chromadb.Client()
            # 创建一个集合（相当于数据库里的一张表）
            collection = chroma_client.create_collection(name="my_docs")
            
            # 批量把文本、向量、ID 塞进去
            for i, chunk in enumerate(chunks_list):
                vector = get_embedding(chunk)
                collection.add(
                    embeddings=[vector],
                    documents=[chunk],
                    ids=[f"id_{i}"]
                )
            return collection

        # 调用函数建库（只要文件没变，就不会重复跑这段代码）
        collection = build_vector_db(uploaded_file.name, chunks)
        # ==========================================================

        question = st.text_input("请输入关于这份文档的问题：")
        
        if st.button("发送"):
            if question:
                with st.spinner("正在数据库中进行语义检索..."):
                    # 把问题变成向量
                    q_vec = get_embedding(question)
                    
                    # 去 ChromaDB 里找前 3 个最相关的段落（极速！）
                    results = collection.query(
                        query_embeddings=[q_vec],
                        n_results=3
                    )
                    
                    # 提取检索到的文本和距离（ChromaDB 默认返回的是距离，越小越相似）
                    retrieved_chunks = results['documents'][0]
                    # 为了兼容咱们之前的拒答逻辑，用距离算个粗略的分数
                    best_distance = results['distances'][0][0]
                    
                    # 拼接上下文
                    context = "\n\n---\n\n".join(retrieved_chunks)
                
                # 阈值判断（Chroma 的 L2 距离一般是 0~2 之间，这里大于 1.2 认为不相关）
                if best_distance > 1.2:
                    st.warning(f"抱歉，文档中没有找到相关内容。（最近距离: {best_distance:.2f}）")
                else:
                    with st.expander("🔍 展开查看从数据库中检索到的 Top-3 段落"):
                        st.write(context)
                        
                    prompt = f"请根据以下文档片段回答用户问题：\n\n文档片段：\n{context}\n\n问题：{question}"
                    
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