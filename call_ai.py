import os
import requests
from dotenv import load_dotenv

# 1. 读取 .env 文件里的密钥
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# 2. 告诉程序我们要去哪家餐厅（DeepSeek 的服务器地址）
url = "https://api.deepseek.com/chat/completions"

# 3. 准备我们的“通行证”和格式
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# 4. 准备菜单：我们要对 AI 说什么
data = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": "用一句话介绍你自己"}
    ]
}

# 5. 发送请求！(就像服务员把菜单递给后厨)
print("正在呼叫 AI，请稍等...")
response = requests.post(url, headers=headers, json=data)

# 这一行排错用的代码可以删掉了
# print("服务器返回的原始信息：", response.json())

# 把之前注释掉的提取代码放出来
answer = response.json()["choices"][0]["message"]["content"]
print("\nAI 的回答是：")
print(answer)