import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
url = "https://api.deepseek.com/chat/completions"

# 1. 读取我们的知识库文件
with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. 把长文本切成一小段一小段的（这里我们先按换行符切）
chunks = text.split("\n")
chunks = [c.strip() for c in chunks if c.strip()] # 去掉空白行

def ask_ai(question):
    print(f"\n用户提问：{question}")
    
    # 3. 【检索阶段】找相关的段落
    # 这里我们用最笨的“关键词包含”方法，看哪个段落包含了问题里的字
    relevant_chunks = []
    for chunk in chunks:
        # 如果段落里有“李四”或者“苹果”，就把它算作相关资料（你可以根据你的文档改关键词）
        # 真实业务里，这里会换成复杂的向量检索
        if "李四" in chunk or "苹果" in chunk or "测试" in chunk:
            relevant_chunks.append(chunk)
            
    # 把找出来的段落拼成一段资料
    context = "\n".join(relevant_chunks)
    print(">> 检索到相关资料：", context if context else "没有找到相关资料")
    
    # 4. 【生成阶段】把资料和问题一起发给 AI
    if context:
        system_prompt = f"你是一个基于资料回答问题的助手。请严格根据以下资料回答用户问题，如果资料里没有，就说不知道。\n资料：\n{context}"
    else:
        system_prompt = "你是一个助手，请回答问题。但如果遇到不懂的，请说不知道。"
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    data = {"model": "deepseek-chat", "messages": messages}
    response = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=data)
    
    answer = response.json()["choices"][0]["message"]["content"]
    print(">> AI 回答：", answer)

# 测试一下
ask_ai("文档里是怎么测试的？") 