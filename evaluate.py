import os
import requests
import math
from dotenv import load_dotenv

load_dotenv()
ds_key = os.getenv("DEEPSEEK_API_KEY")
zhipu_key = os.getenv("ZHIPU_API_KEY")

# (把之前的辅助函数搬过来)
def get_embedding(text):
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    headers = {"Authorization": f"Bearer {zhipu_key}"}
    data = {"model": "embedding-2", "input": text}
    response = requests.post(url, headers=headers, json=data)
    return response.json()["data"][0]["embedding"]

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0: return 0
    return dot_product / (mag1 * mag2)

documents = [
    "李四是我们公司的测试工程师。", 
    "他的主要工作是对新产品进行功能测试和性能测试。",
    "今天天气真不错，适合出去游玩。",
    "苹果是一种非常好吃的水果，富含维生素。"
]
doc_vectors = [get_embedding(doc) for doc in documents]

# ================= 测试数据集 =================
# 模拟真实业务场景：一半能回答，一半应该拒答
test_cases = [
    {"question": "这水果好吃吗？", "expect_answer": True},
    {"question": "李四的工作是什么？", "expect_answer": True},
    {"question": "今天股市大盘怎么样？", "expect_answer": False},
    {"question": "地球是圆的吗？", "expect_answer": False},
    {"question": "外面天气好吗？", "expect_answer": True},
]

print("开始自动化评估...\n")
correct_count = 0

for case in test_cases:
    q = case["question"]
    expect = case["expect_answer"]
    
    # 1. 检索
    q_vec = get_embedding(q)
    highest_score = -1
    for doc_vec in doc_vectors:
        score = cosine_similarity(q_vec, doc_vec)
        if score > highest_score:
            highest_score = score
            
    # 2. 判断（模拟我们在图里的条件边）
    # 如果得分 >= 0.4，我们认为它会给出回答；否则拒答
    actual_answer = True if highest_score >= 0.6 else False
    
    # 3. 记录结果
    is_correct = (actual_answer == expect)
    if is_correct:
        correct_count += 1
        print(f"✅ 通过 | 问题: {q} (得分 {highest_score:.2f})")
    else:
        print(f"❌ 失败 | 问题: {q} (得分 {highest_score:.2f}) -> 期望: {expect}, 实际: {actual_answer}")

# 4. 输出报告
accuracy = (correct_count / len(test_cases)) * 100
print(f"\n========== 评估报告 ==========")
print(f"测试总数: {len(test_cases)}")
print(f"正确数量: {correct_count}")
print(f"准确率: {accuracy:.1f}%")