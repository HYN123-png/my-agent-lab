import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
url = "https://api.deepseek.com/chat/completions"

# ================= 1. 定义真实的工具 =================
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_notes_file():
    try:
        with open("notes.json", "r", encoding="utf-8") as f:
            notes = json.load(f)
            return json.dumps(notes, ensure_ascii=False)
    except Exception as e:
        return f"读取文件失败：{e}"

# ================= 2. 提示词 =================
system_prompt = """
你是一个智能助手。你可以使用以下工具：
1. get_current_time：获取当前时间，不需要参数。
2. read_notes_file：读取本地笔记文件，不需要参数。

【重要规则】：每次你只能调用【一个】工具！必须输出且只能输出一个 JSON 对象。
如果你还需要用其他工具，请在收到结果后再次输出 JSON。
当你不需要工具或已经知道答案时，直接输出普通文本。
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "现在几点了？另外，帮我读一下我的笔记文件，看看里面有什么？"}
]

print("用户提问：", messages[-1]["content"])

# ================= 3. 开始 Agent 循环 =================
for step in range(5):
    print(f"\n--- 第 {step+1} 轮 ---")
    
    data = {"model": "deepseek-chat", "messages": messages}
    response = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=data)
    reply = response.json()["choices"][0]["message"]["content"]
    
    print("AI 说：", reply)
    messages.append({"role": "assistant", "content": reply})
    
    # ================= 4. 增强版的解析和容错逻辑 =================
    try:
        clean_reply = reply.strip().replace("```json", "").replace("```", "").strip()
        lines = clean_reply.split('\n')
        valid_json_str = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                valid_json_str = line
                break
        
        if not valid_json_str:
            print(">> AI 给出了最终回答，循环结束。")
            break
            
        action_data = json.loads(valid_json_str)
        
        # 兼容性补丁：不管 AI 输出的是 action 还是 tool，我们都能抓到
        tool_name = action_data.get("action") or action_data.get("tool")
        
        if tool_name:
            print(f">> 发现 AI 想要调用工具：{tool_name}")
            
            if tool_name == "get_current_time":
                result = get_current_time()
            elif tool_name == "read_notes_file":
                result = read_notes_file()
            else:
                result = "没有找到这个工具"
                
            print(">> 工具执行结果：", result)
            
            messages.append({
                "role": "user", 
                "content": f"工具执行结果：{result}。请继续下一步，如果已经能回答用户，就直接回答；如果需要用其他工具，继续输出唯一的 JSON。"
            })
        else:
            # 如果 AI 既没写 action 也没写 tool，我们强行纠正它
            print(">> AI 格式写错了，强迫它重写")
            messages.append({
                "role": "user", 
                "content": "你的 JSON 格式不正确，必须严格使用 {\"action\": \"工具名\"} 的格式！请重新输出。"
            })
            
    except Exception as e:
        print(f">> 解析出现异常：{e}，循环结束。")
        break