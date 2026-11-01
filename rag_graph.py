import os
import requests
from typing import TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()
ds_key = os.getenv("DEEPSEEK_API_KEY")
zhipu_key = os.getenv("ZHIPU_API_KEY")

# ================= 1. 定义共享状态（State） =================
# 这是所有节点共享的记事本
class GraphState(TypedDict):
    question: str   # 用户问题
    context: str    # 检索到的资料
    answer: str     # 最终答案

# ================= 2. 定义辅助函数（复用之前代码） =================
def get_embedding(text):
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    headers = {"Authorization": f"Bearer {zhipu_key}"}
    data = {"model": "embedding-2", "input": text}
    response = requests.post(url, headers=headers, json=data)
    return response.json()["data"][0]["embedding"]

def cosine_similarity(vec1, vec2):
    import math
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (magnitude1 * magnitude2)

documents = [
    "李四是我们公司的测试工程师。", 
    "他的主要工作是对新产品进行功能测试和性能测试。",
    "今天天气真不错，适合出去游玩。",
    "苹果是一种非常好吃的水果，富含维生素。"
]
doc_vectors = [get_embedding(doc) for doc in documents]

# ================= 3. 定义节点（Node） =================
# 节点 1：检索节点（找资料）
def retrieve_node(state: GraphState):
    print("--- 正在执行：检索节点 ---")
    question = state["question"]
    q_vec = get_embedding(question)
    
    best_idx = -1
    highest_score = -1
    for i, doc_vec in enumerate(doc_vectors):
        score = cosine_similarity(q_vec, doc_vec)
        if score > highest_score:
            highest_score = score
            best_idx = i
            
    context = documents[best_idx]
    print(f"（检索到资料：{context}）")
    # 返回要更新到 State 的字段
    return {"context": context}

# 节点 2：生成节点（让 AI 写回答）
def generate_node(state: GraphState):
    print("--- 正在执行：生成节点 ---")
    question = state["question"]
    context = state["context"]
    
    system_prompt = f"请根据以下资料回答用户问题：\n资料：{context}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    url = "https://api.deepseek.com/chat/completions"
    data = {"model": "deepseek-chat", "messages": messages}
    response = requests.post(url, headers={"Authorization": f"Bearer {ds_key}"}, json=data)
    
    answer = response.json()["choices"][0]["message"]["content"]
    print(f"（AI 回答：{answer}）")
    return {"answer": answer}

# ================= 4. 搭建图（Graph） =================
# 初始化图，并告诉它我们要用 GraphState 这个记事本
workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# 设置入口点（从哪开始）
workflow.set_entry_point("retrieve")

# 添加边（连线：检索完去生成，生成完结束）
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# 编译图（变成可执行的机器）
app = workflow.compile()

# ================= 5. 测试运行 =================
if __name__ == "__main__":
    print("=== 开始运行 LangGraph RAG ===")
    inputs = {"question": "这水果好吃吗？"}
    # 运行图
    result = app.invoke(inputs)
    print("\n=== 最终结果 ===")
    print(result["answer"])