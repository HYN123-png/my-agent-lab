import os
import requests
from typing import TypedDict, List
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

load_dotenv()
ds_key = os.getenv("DEEPSEEK_API_KEY")
zhipu_key = os.getenv("ZHIPU_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None

class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    file_data: bytes
    file_name: str

# ================= 1. 意图判断节点 =================
def intent_router(state: RAGState):
    q = state["question"]
    if any(kw in q for kw in ["画一张", "画个", "生成图片", "画图"]):
        return "image"
    if any(kw in q for kw in ["表格", "Excel", "excel"]):
        return "excel"
    if any(kw in q for kw in ["PPT", "ppt", "幻灯片"]):
        return "ppt"
    return "search"

# ================= 2. 功能节点 =================
def image_generator_node(state: RAGState):
    prompt = state["question"]
    url = "https://open.bigmodel.cn/api/paas/v4/images/generations"
    headers = {"Authorization": f"Bearer {zhipu_key}", "Content-Type": "application/json"}
    data = {"model": "cogview-3-flash", "prompt": prompt} # 直接把用户的话扔给画图模型，效果更好
    
    try:
        res = requests.post(url, headers=headers, json=data)
        res.raise_for_status()
        img_bytes = requests.get(res.json()["data"][0]["url"]).content
        return {
            "file_data": img_bytes,
            "file_name": "AI生成图片.png",
            "answer": f"✅ 图片已生成完毕！\n\n**画图提示词**：{prompt}"
        }
    except Exception as e:
        return {"answer": f"⚠️ 图片生成失败：{e}"}

def excel_generator_node(state: RAGState):
    return {"answer": "✅ Excel 已生成（API 版暂略）", "file_name": "test.xlsx", "file_data": b"fake_excel_data"}

def ppt_generator_node(state: RAGState):
    return {"answer": "✅ PPT 已生成（API 版暂略）", "file_name": "test.pptx", "file_data": b"fake_ppt_data"}

def web_search_node(state: RAGState):
    if not tavily_client: return {"context": "无网络"}
    query = state["question"]
    try:
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        web_context = "\n\n".join([f"【来源网页：{r['url']}】\n{r['content'][:500]}" for r in response.get("results", [])])
        return {"context": web_context}
    except Exception as e:
        return {"context": f"【搜索报错：{e}】"}

def research_node(state: RAGState):
    prompt = f"请根据以下搜索片段回答用户问题：\n{state['context'][:3000]}\n\n问题：{state['question']}"
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}])
        return {"answer": res.choices[0].message.content}
    except:
        return {"answer": state["context"][:200]}

# ================= 3. 重新构建图（核心修复） =================
# 一个什么都不干的空节点，专门用来做分流起点
def router_node(state: RAGState):
    return state

workflow = StateGraph(RAGState)
workflow.add_node("router", router_node) # 纯路由节点
workflow.add_node("image", image_generator_node)
workflow.add_node("excel", excel_generator_node)
workflow.add_node("ppt", ppt_generator_node)
workflow.add_node("search", web_search_node)
workflow.add_node("research", research_node)

# 1. 起点设在 router
workflow.set_entry_point("router")

# 2. 在 router 处分流
workflow.add_conditional_edges("router", intent_router, {
    "image": "image",
    "excel": "excel",
    "ppt": "ppt",
    "search": "search"
})

# 3. 后续连线
workflow.add_edge("search", "research")
workflow.add_edge("image", END)
workflow.add_edge("excel", END)
workflow.add_edge("ppt", END)
workflow.add_edge("research", END)

app_graph = workflow.compile()