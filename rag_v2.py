import os
import requests
import json
import math
from dotenv import load_dotenv

load_dotenv()
ds_key = os.getenv("DEEPSEEK_API_KEY")
zhipu_key = os.getenv("ZHIPU_API_KEY")

documents = [
    "李四是我们公司的测试工程师。", 
    "他的主要工作是对新产品进行功能测试和性能测试。",
    "今天天气真不错，适合出去游玩。",
    "苹果是一种非常好吃的水果，富含维生素。",
    "如果遇到严重的bug，他会第一时间上报。"
]

def get_embedding(text):
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    headers = {
        "Authorization": f"Bearer {zhipu_key}",
        "Content-Type": "application/json"  # 加上这个更保险
    }
    data = {"model": "embedding-2", "input": text}
    response = requests.post(url, headers=headers, json=data)
    
    # 检查 HTTP 状态码，如果不是 200，说明请求被拒绝了
    if response.status_code != 200:
        print(f"请求失败！状态码：{response.status_code}")
        print("服务器返回详情：", response.text)
        return [] # 返回空列表，防止程序崩溃
    
    # 成功时再解析 data 字段
    return response.json()["data"][0]["embedding"]

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

print("正在把知识库转换成向量...")
doc_vectors = []
for doc in documents:
    vector = get_embedding(doc)
    doc_vectors.append(vector)
    if not vector: # 如果向量为空，说明出错了，直接停止
        print("向量化失败，请检查上面的错误信息。")
        exit() # 退出程序
print("向量化完成！\n")

def ask_ai(question):
    print(f"用户提问：{question}")
    question_vector = get_embedding(question)
    if not question_vector:
        return
    
    best_match_index = -1
    highest_score = -1
    for i, doc_vec in enumerate(doc_vectors):
        score = cosine_similarity(question_vector, doc_vec)
        if score > highest_score:
            highest_score = score
            best_match_index = i
            
    context = documents[best_match_index]
    print(f">> 找到了最相似的资料 (相似度得分: {highest_score:.4f})：\n{context}\n")
    
    url = "https://api.deepseek.com/chat/completions"
    system_prompt = f"你是一个助手。请根据以下资料回答用户问题，资料里没有的不要瞎编。\n资料：{context}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    data = {"model": "deepseek-chat", "messages": messages}
    response = requests.post(url, headers={"Authorization": f"Bearer {ds_key}"}, json=data)
    answer = response.json()["choices"][0]["message"]["content"]
    print(f">> AI 最终回答：\n{answer}\n")
    print("-" * 50)

ask_ai("文档里是怎么测试的？")
ask_ai("这水果好吃吗？")