import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
url = "https://api.deepseek.com/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# 准备一段乱七八糟的日报文本
my_work_text = "今天早上开了个会，然后写了两小时代码。下午本来想把数据库弄好，结果报错超时了，搞了一下午没弄好，烦死了。"

data = {
    "model": "deepseek-chat", 
    # 这里我们强制要求返回 JSON
    "response_format": {"type": "json_object"},
    "messages": [
        # 这里的 system 提示词非常关键，告诉 AI 它的角色和任务
        {"role": "system", "content": "你是一个严格的日报整理助手。你必须只输出一个 JSON 格式，不要输出任何多余的文字。JSON 必须包含两个字段：done（今天完成的事情）和 blocked（今天卡住的事情）。"},
        {"role": "user", "content": my_work_text}
    ]
}

print("正在整理日报...")
response = requests.post(url, headers=headers, json=data)

# 提取 AI 返回的 JSON 字符串
json_string = response.json()["choices"][0]["message"]["content"]

# 把字符串转换成真正的 Python 字典
report = json.loads(json_string)

print("\n整理结果：")
print("【已完成】", report["done"])
print("【被卡住】", report["blocked"])