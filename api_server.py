from fastapi import FastAPI, Header, HTTPException
from functools import wraps
from pydantic import BaseModel
import uvicorn
import sqlite3
# 假设你把你刚才的 app_graph 抽离到了 core_agent.py 里
from core_agent import app_graph 

from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
# ================= 计费系统数据库初始化 =================
def init_db():
    conn = sqlite3.connect('billing.db')
    cursor = conn.cursor()
    # 创建用户表，存储 API Key 和余额
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key TEXT UNIQUE NOT NULL,
            balance REAL NOT NULL DEFAULT 0.0
        )
    ''')
    # 插入一个测试用户，分配 Key "test_123"，初始余额 10.0 元
    cursor.execute("INSERT OR IGNORE INTO users (api_key, balance) VALUES (?, ?)", ("test_123", 10.0))
    conn.commit()
    conn.close()

# 启动服务时执行一次初始化
init_db()
# =======================================================

app = FastAPI(title="全能办公 Agent API", docs_url=None) # 关掉默认的 docs

# 自定义 docs 路由，使用国内 CDN
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        # 替换为国内 BootCDN 源
        swagger_js_url="https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.9.0/swagger-ui.css",
    )
# ================= 计费中间件 =================
# 模拟每次调用的价格（单位：元）
REQUEST_COST = 0.01 

def require_payment(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # 从请求头中获取 API Key
        api_key = kwargs.get('x_api_key')
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API Key")

        conn = sqlite3.connect('billing.db')
        cursor = conn.cursor()
        
        # 开始事务，确保扣费操作的原子性
        cursor.execute("BEGIN IMMEDIATE")
        try:
            # 1. 查询用户余额
            cursor.execute("SELECT id, balance FROM users WHERE api_key = ?", (api_key,))
            user = cursor.fetchone()
            if not user:
                raise HTTPException(status_code=403, detail="Invalid API Key")
            
            user_id, balance = user

            # 2. 检查余额是否充足
            if balance < REQUEST_COST:
                raise HTTPException(status_code=402, detail="Insufficient balance")

            # 3. 执行扣费
            cursor.execute(
                "UPDATE users SET balance = balance - ? WHERE id = ? AND balance >= ?",
                (REQUEST_COST, user_id, REQUEST_COST)
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            # 👇 增加这个判断：如果抛出的是 HTTPException（比如 401/402/403），直接原样抛出，不要包装成 500
            if isinstance(e, HTTPException):
                raise e
            # 如果是其他未知错误，才返回 500
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            conn.close()

        # 扣费成功后，放行执行真正的业务逻辑
        return await func(*args, **kwargs)
    return wrapper
# ==============================================    


class AgentRequest(BaseModel):
    query: str
    

@app.post("/chat")
@require_payment  # 👈 加上计费拦截器
async def chat_with_agent(
    req: AgentRequest, 
    x_api_key: str = Header(..., description="请在请求头中填入您的API Key") # 👈 从 Header 接收 Key
):
    # 1. 验证 api_key 和 扣费逻辑 (略)
    
    # 2. 调用你的 LangGraph
    events = app_graph.stream({
        "question": req.query, 
        "rewritten_question": "", "retries": 0, "history": [], "ppt_style": "AI智能推荐"
    }, stream_mode="values")
    
    final_result = None
    for event in events: final_result = event
    
    # 3. 返回结果，如果是文件，转换为 Base64 返回
    response_data = {"text": final_result.get("answer", "")}
    if final_result.get("file_data"):
        import base64
        response_data["file_base64"] = base64.b64encode(final_result["file_data"]).decode('utf-8')
        response_data["file_name"] = final_result.get("file_name")
        
    return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)