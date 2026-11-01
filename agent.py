import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
url = "https://api.deepseek.com/chat/completions"

# 1. 定义真实的工具函数：获取当前时间
def get_current_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# 2. 告诉 AI 它有哪些工具可以使用（用一段描述告诉它）
system_prompt = """
你是一个智能助手。你可以使用以下工具：
- 工具名：get_current_time
- 作用：获取当前的准确时间，不需要任何参数。
如果你需要使用工具，请直接输出以下严格的 JSON 格式：
{"action": "get_current_time"}
如果你不需要使用工具，或者已经得到了结果，请直接输出普通文本。
"""

# 3. 准备对话历史
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "现在几点了？"} # 用户提问
]

print("用户提问：现在几点了？")

# 4. 开启 Agent 的循环（只跑前两轮看看效果）
for step in range(2):
    print(f"\n--- 第 {step+1} 轮思考 ---")
    
    # 发送请求给 AI
    data = {
        "model": "deepseek-chat",
        "messages": messages
    }
    response = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=data)
    reply = response.json()["choices"][0]["message"]["content"]
    
    print("AI 说：", reply)
    
    # 把 AI 的回复加入历史记录
    messages.append({"role": "assistant", "content": reply})
    
    # 5. 判断 AI 是不是想用工具
    try:
        # 尝试把 AI 的输出当做 JSON 解析
        action_data = json.loads(reply)
        
        # 如果解析成功，并且里面有 action 字段
        if "action" in action_data and action_data["action"] == "get_current_time":
            print(">> 发现 AI 想要调用工具！")
            
            # 执行真实的函数拿到时间
            time_result = get_current_time()
            print(">> 工具执行结果：", time_result)
            
            # 把工具的结果塞回给 AI（非常重要！）
            messages.append({
                "role": "user", 
                "content": f"工具执行结果：{time_result}，请根据这个结果回答我。"
            })
    except json.JSONDecodeError:
        # 如果 AI 输出的不是 JSON，说明它没想用工具，直接给出了最终答案
        print(">> AI 给出了最终回答，循环结束。")
        break