import os
import requests
import math
from typing import TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()
ds_key = os.getenv("DEEPSEEK_API_KEY")
zhipu_key = os.getenv("ZHIPU_API_KEY")

# ================= 1. 升级共享状态 =================
# 新增了 score 字段，用来记录相似度得分
class GraphState(TypedDict):
    question: str   
    context: str    
    score: float    # 新增：记录最高相似度得分
    answer: str     

def get_embedding(text):
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    headers = {"Authorization": f"Bearer {zhipu_key}"}
    data = {"model": "embedding-2", "input": text}
    response = requests.post(url, headers=headers, json=data)
    return response.json()["data"][0]["embedding"]

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0: return 0
    return dot_product / (magnitude1 * magnitude2)

documents = [
    "李四是我们公司的测试工程师。", 
    "他的主要工作是对新产品进行功能测试和性能测试。",
    "今天天气真不错，适合出去游玩。",
    "苹果是一种非常好吃的水果，富含维生素。"
]
print("正在向量化知识库...")
doc_vectors = [get_embedding(doc) for doc in documents]

# ================= 2. 节点定义 =================
# 节点 1：检索节点（现在要把得分也存进去）
def retrieve_node(state: GraphState):
    print("\n--- 执行：检索节点 ---")
    question = state["question"]
    q_vec = get_embedding(question)
    
    best_idx = -1
    highest_score = -1
    for i, doc_vec in enumerate(doc_vectors):
        score = cosine_similarity(q_vec, doc_vec)
        if score > highest_score:
            highest_score = score
            best_idx = i
            
    print(f"（检索到资料：{documents[best_idx]}，得分：{highest_score:.4f}）")
    return {"context": documents[best_idx], "score": highest_score}

# 节点 2：生成节点（正常回答）
def generate_node(state: GraphState):
    print("--- 执行：生成节点 ---")
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
    return {"answer": answer}

# 节点 3：拒答节点（得分太低时触发）
def reject_node(state: GraphState):
    print("--- 执行：拒答节点 ---")
    return {"answer": "抱歉，我的知识库里没有找到与该问题相关的内容，我无法回答。"}

# ================= 3. 路由函数（判断走哪条路） =================
def check_relevance(state: GraphState):
    # 如果最高得分小于 0.4，说明不相关，返回 "reject"
    # 否则，返回 "generate"
    if state["score"] < 0.4:
        return "reject"
    return "generate"

# ================= 4. 搭建图 =================
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("reject", reject_node)

workflow.set_entry_point("retrieve")

# 核心魔法：添加条件边
# 检索完之后，调用 check_relevance 函数来决定下一步去哪
workflow.add_conditional_edges(
    "retrieve",
    check_relevance,
    {
        "generate": "generate", # 如果返回 generate，就去生成节点
        "reject": "reject"      # 如果返回 reject，就去拒答节点
    }
)

# 生成或拒答结束后，流程都结束
workflow.add_edge("generate", END)
workflow.add_edge("reject", END)

app = workflow.compile()

# ================= 5. 测试运行 =================
if __name__ == "__main__":
    print("\n========== 测试 1：知识库里有答案 ==========")
    # "水果" 在知识库里能匹配到"苹果"，得分很高，应该走生成节点
    result1 = app.invoke({"question": "这水果好吃吗？"})
    print("最终结果：", result1["answer"])

    print("\n========== 测试 2：知识库里没答案 ==========")
    # "股市" 在知识库里完全不存在，得分会很低，应该被条件边拦截，走拒答节点
    result2 = app.invoke({"question": "今天股市大盘怎么样？"})
    print("最终结果：", result2["answer"])