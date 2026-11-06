import streamlit as st
import os
import concurrent.futures
import re
import requests
import numpy as np
import jieba
import docx
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor as DocxRGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import io
import json
import sqlite3
import datetime
import base64
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from typing import TypedDict, List
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from rank_bm25 import BM25Okapi
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

st.set_page_config(page_title="企业级 Agentic RAG", page_icon="🤖")
st.title("🤖 企业级 Agentic RAG 智能体 (v10.0)")

load_dotenv()
ds_key = os.getenv("DEEPSEEK_API_KEY")
zhipu_key = os.getenv("ZHIPU_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None
# ================= 0. PPT 风格配置 =================
PPT_THEMES = {
    "科技深邃风": {
        "bg": RGBColor(30, 32, 38),      # 极深灰背景
        "title": RGBColor(255, 255, 255), # 纯白标题
        "text": RGBColor(200, 205, 210),  # 灰白正文
        "accent": RGBColor(0, 150, 255),  # 亮蓝色装饰条
    },
    "极简商务风": {
        "bg": RGBColor(248, 249, 250),    # 极浅灰背景
        "title": RGBColor(33, 37, 41),    # 深黑灰标题
        "text": RGBColor(73, 80, 87),     # 深灰正文
        "accent": RGBColor(40, 167, 69),  # 墨绿色装饰条
    },
    "活力创意风": {
        "bg": RGBColor(255, 248, 225),    # 奶黄色背景
        "title": RGBColor(211, 47, 47),   # 砖红色标题
        "text": RGBColor(62, 39, 35),     # 深咖色正文
        "accent": RGBColor(255, 152, 0),  # 亮橙色装饰条
    }
}
# ================= 数据库管理 =================
DB_FILE = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS experiences
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, task_type TEXT, lesson TEXT)''')
    # 用户表 (简单模拟多租户)
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id TEXT PRIMARY KEY, balance REAL DEFAULT 100.0)''') # 默认送100额度
    # 会话表
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, title TEXT, created_at TEXT, files TEXT)''')
    # 消息表 (增加成本记录)
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, conv_id INTEGER, user_id TEXT, role TEXT, content TEXT, cost REAL DEFAULT 0, created_at TEXT)''')
    
    # 兼容旧表
    try: c.execute("ALTER TABLE conversations ADD COLUMN files TEXT")
    except: pass
    try: c.execute("ALTER TABLE messages ADD COLUMN cost REAL DEFAULT 0")
    except: pass
        
    conn.commit()
    conn.close()
def update_conv_files(conv_id, filenames):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # 将文件名列表转为字符串存入
    files_str = ", ".join(filenames)
    c.execute("UPDATE conversations SET files = ? WHERE id = ?", (files_str, conv_id))
    conn.commit()
    conn.close()

def get_conv_files(conv_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT files FROM conversations WHERE id = ?", (conv_id,))
    res = c.fetchone()
    conn.close()
    return res[0] if res and res[0] else ""

def get_conversations():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, title FROM conversations ORDER BY id DESC")
    convs = c.fetchall()
    conn.close()
    return convs

def create_conversation(title="新对话"):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO conversations (title, created_at) VALUES (?, ?)", (title, now))
    conv_id = c.lastrowid
    conn.commit()
    conn.close()
    return conv_id

def save_message(conv_id, role, content, cost=0):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 【核心修复】：自动兼容云端旧版数据库，防止没有 cost 字段报错
    try:
        c.execute("ALTER TABLE messages ADD COLUMN cost REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # 如果字段已经存在，忽略报错
        
    # 插入消息
    c.execute("INSERT INTO messages (conv_id, role, content, cost, created_at) VALUES (?, ?, ?, ?, ?)", 
              (conv_id, role, content, cost, now))
    conn.commit()
    conn.close()

def get_messages(conv_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE conv_id = ? ORDER BY id ASC", (conv_id,))
    msgs = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in msgs]

def update_conv_title(conv_id, title):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
    conn.commit()
    conn.close()

# 初始化数据库
init_db()

# ================= 文件本地持久化工具 =================
def save_uploaded_files(conv_id, uploaded_files):
    """把上传的文件保存到本地磁盘"""
    dir_path = f"./uploads/{conv_id}"
    os.makedirs(dir_path, exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join(dir_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

def load_local_files(conv_id):
    """从本地磁盘读取该会话的所有文件"""
    dir_path = f"./uploads/{conv_id}"
    if not os.path.exists(dir_path):
        return []
    files = []
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        with open(file_path, "rb") as f:
            content = f.read()
            # 构造一个类似 UploadedFile 的对象，骗过 Streamlit
            class LocalFile:
                def __init__(self, name, content):
                    self.name = name
                    self._content = content
                def getbuffer(self):
                    return self._content
                def read(self, *args):
                    return self._content
            files.append(LocalFile(filename, content))
    return files
# =======================================================
# ================= 1. 基础函数 =================
def get_embedding(text):
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    res = requests.post(url, headers={"Authorization": f"Bearer {zhipu_key}"}, json={"model": "embedding-2", "input": text})
    return res.json()["data"][0]["embedding"]

def tokenize(text):
    words = jieba.lcut(text.lower())
    return [w for w in words if w.strip() and not re.match(r'^[^\w\s]+$', w)]

def rrf_fuse(list1, list2, k=60):
    scores = {}
    for rank, doc in enumerate(list1):
        scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank + 1)
    for rank, doc in enumerate(list2):
        scores[doc] = scores.get(doc, 0) + 1.0 / (k + rank + 1)
    return [doc for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

# ================= 2. 多格式解析 =================
def extract_text_from_image_vlm(image_file):
    image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {"Authorization": f"Bearer {zhipu_key}", "Content-Type": "application/json"}
    data = {
        "model": "glm-4v-flash",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "提取图片文字，表格用Markdown还原。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}]
    }
    try:
        res = requests.post(url, headers=headers, json=data)
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[图片解析失败：{e}]"

def extract_text_from_file(file):
    filename = file.name.lower()
    text = ""
    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages: text += (page.extract_text() or "") + "\n"
        elif filename.endswith(".docx"):
            doc = docx.Document(file)
            for para in doc.paragraphs: text += para.text + "\n"
            for table in doc.tables:
                for row in table.rows: text += " | ".join([cell.text.strip() for cell in row.cells]) + "\n"
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
            text = df.to_markdown(index=False)
        elif filename.endswith(".pptx"):
            prs = Presentation(file)
            for i, slide in enumerate(prs.slides):
                text += f"\n--- 幻灯片 {i+1} ---\n"
                for shape in slide.shapes:
                    if shape.has_text_frame: text += shape.text + "\n"
                    if shape.has_table:
                        for row in shape.table.rows: text += " | ".join([cell.text.strip() for cell in row.cells]) + "\n"
        elif filename.endswith((".txt", ".md")):
            text = file.read().decode("utf-8")
        elif filename.endswith((".jpg", ".jpeg", ".png")):
            text = extract_text_from_image_vlm(file)
    except Exception as e:
        st.error(f"解析文件 {file.name} 失败：{e}")
    return text

def process_files(uploaded_files):
    all_chunks = []
    for file in uploaded_files:
        raw_text = extract_text_from_file(file)
        if not raw_text.strip(): continue
        for i in range(0, len(raw_text), 350):
            chunk = raw_text[i:i+400].strip()
            if len(chunk) > 20:
                all_chunks.append(f"【来源文档：{file.name}】\n{chunk}")
    return all_chunks

# ================= 3. 状态定义与核心节点 =================
class RAGState(TypedDict):
    question: str
    standalone_query: str
    rewritten_question: str
    context: str
    is_relevant: str
    retries: int
    answer: str
    history: List[dict]
    file_data: bytes  # 用来存生成的 Excel 文件内容
    file_name: str    # 文件名
    ppt_style: str  # 👈 新增这一行，用于接收前端选择的风格

# 新增：将多轮对话合并为独立问题，解决上下文断片
def rewrite_query_with_history(state: RAGState):
    history = state.get("history", [])
    if not history:
        return {"standalone_query": state["question"]}
    
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]])
    prompt = f"""根据以下对话历史，将用户的最后一个问题重写为一个独立的、包含完整上下文的查询语句。

【极其重要的规则】：
1. 如果用户的最后问题是“那明天呢”、“后天呢”、“未来几天呢”，你必须去历史记录里找到对应的【地点】或【实体】。
2. 绝对禁止自己编造或猜测未出现的地点（例如，如果历史里说的是南昌，重写结果必须是“南昌明天天气”，绝不能是纽约）。
3. 如果历史记录中实在找不到相关实体，请直接输出原问题。
4. 只输出重写后的句子，绝对不要任何解释。

历史：
{history_str}
最后问题：{state['question']}
独立查询："""
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0)
        q = res.choices[0].message.content.strip()
        return {"standalone_query": q if q else state["question"]}
    except:
        return {"standalone_query": state["question"]}

def retrieve_node(state: RAGState):
    if not st.session_state.get("current_files"):
        return {"context": "【无本地文档】当前没有上传任何文件。"}
    
    col = st.session_state.get("collection")
    bm = st.session_state.get("bm25")
    chunks = st.session_state.get("chunks_global", [])
    
    query = state.get("rewritten_question") or state.get("standalone_query") or state["question"]
    q_vec = get_embedding(query)
    
    vector_chunks = col.query(query_embeddings=[q_vec], n_results=20)['documents'][0]
    tokenized_q = tokenize(query)
    bm25_scores = bm.get_scores(tokenized_q)
    top_bm25_indices = np.argsort(bm25_scores)[-20:][::-1]
    bm25_chunks = [chunks[i] for i in top_bm25_indices if bm25_scores[i] > 0]
    
    fused = rrf_fuse(vector_chunks, bm25_chunks)
    
    # 配额提取
    file_groups = {f: [] for f in st.session_state.get("current_files", [])}
    for chunk in fused:
        match = re.search(r"【来源文档：(.*?)】", chunk)
        if match and match.group(1) in file_groups:
            file_groups[match.group(1)].append(chunk)
            
    final_chunks = []
    for f, chunks_list in file_groups.items(): final_chunks.extend(chunks_list[:4])
    for chunk in fused:
        if chunk not in final_chunks: final_chunks.append(chunk)
        if len(final_chunks) >= 12: break
            
    return {"context": "\n\n---\n\n".join(final_chunks)}

def grade_node(state: RAGState):
    if "【无本地文档】" in state["context"]:
        return {"is_relevant": "no"}
        
    # 【核心强制】：如果问题或重写后的问题里包含“结合文档”、“根据资料”等词，强制认为相关
    q_check = state.get("standalone_query", "") + state["question"]
    force_local_keywords = ["结合文档", "根据文档", "根据资料", "结合资料", "文中", "根据上文"]
    if any(kw in q_check for kw in force_local_keywords):
        return {"is_relevant": "yes"}

    prompt = f"请判断【资料】能否回答【问题】。如果涉及主观总结/写报告，只要相关就输出'yes'。只输出'yes'或'no'。\n\n【问题】：{state['question']}\n【资料】：{state['context'][:2000]}"
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0)
        return {"is_relevant": "yes" if "yes" in res.choices[0].message.content.lower() else "no"}
    except:
        return {"is_relevant": "no"}

def rewrite_node(state: RAGState):
    prompt = f"原问题：{state['question']}。由于检索不到，请重写为更简短的搜索词，只输出重写结果。"
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.5)
        return {"rewritten_question": res.choices[0].message.content.strip(), "retries": state.get("retries", 0) + 1}
    except:
        return {"rewritten_question": state["question"], "retries": state.get("retries", 0) + 1}

def web_search_node(state: RAGState):
    if not tavily_client:
        return {"context": "【网络搜索接口未配置】请检查 TAVILY_API_KEY。", "is_relevant": "yes"}
    
    # 优先用重写后的问题去搜
    query = state.get("rewritten_question") or state.get("standalone_query") or state["question"]
    
    # 【兜底】：如果经过层层重写，搜索词居然还是只有两三个字（如“明天呢”），强制加上历史里的实体
    if len(query) <= 4 and state.get("history"):
        # 简单粗暴：把上一句用户的话和这句话拼起来搜
        last_user_msg = next((m['content'] for m in reversed(state['history']) if m['role'] == 'user'), "")
        query = f"{last_user_msg} {query}"

    try:
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        valid_results = [r for r in response.get("results", []) if r.get("content")]
        if not valid_results:
            return {"context": f"【网络搜索】未找到关于 '{query}' 的有效结果。", "is_relevant": "yes"}
        web_context = "\n\n".join([f"【来源网页：{r['url']}】\n{r['content'][:800]}" for r in valid_results])
        return {"context": web_context, "is_relevant": "yes"}
    except Exception as e:
        return {"context": f"【网络搜索报错：{str(e)}】", "is_relevant": "yes"}

def research_node(state: RAGState):
    safe_context = state['context'][:3000]
    
    if "【网络搜索" in state["context"] and len(safe_context) < 150:
        return {"answer": f"⚠️ {safe_context}"}
        
    # 【核心：读取经验池】
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT lesson FROM experiences ORDER BY id DESC LIMIT 3")
    lessons = c.fetchall()
    conn.close()
    experience_prompt = ""
    if lessons:
        experience_prompt = "【历史经验教训，请务必吸取】：\n" + "\n".join([f"- {l[0]}" for l in lessons])

    if "【来源网页：" in state["context"]:
        prompt = f"""你是一个极简的搜索问答助手。请根据以下搜索片段，直接提取出用户问题的核心答案。
{experience_prompt}  # 👈 注入经验
要求：如果查到了数据，直接给结果。如果没有查到确切数据，直接回答“未搜索到最新实时数据”。
搜索片段：
{safe_context}
用户问题：{state['question']}
答案："""
        try:
            res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0)
            draft = res.choices[0].message.content
            if not draft or not draft.strip():
                return {"answer": f"[[DIRECT]]\n根据网络检索，为您找到以下参考信息：\n{safe_context[:200]}..."}
            return {"answer": f"[[DIRECT]]\n{draft}"}
        except Exception:
            return {"answer": f"[[DIRECT]]\n根据网络检索，为您找到以下参考信息：\n{safe_context[:200]}..."}

    # 处理本地文档
    prompt = f"提取核心事实整理成大纲。\n资料：\n{safe_context}"
    messages = [{"role": "system", "content": prompt}] + state.get("history", [])[-4:] + [{"role": "user", "content": state["question"]}]
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=messages)
        return {"answer": res.choices[0].message.content}
    except Exception as e:
        return {"answer": f"⚠️ 本地文档解析失败：{e}"}

def writer_node(state: RAGState):
    ans = state.get("answer", "")
    if not ans: return {"answer": "未生成任何内容。"}
    
    # 【核心修复】：认出 [[DIRECT]] 标记，或者非常短的文本，直接原样返回，不加排版！
    if "[[DIRECT]]" in ans or len(ans) < 100:
        return {"answer": ans.replace("[[DIRECT]]\n", "")}
        
    prompt = f"作为商业报告主编，请排版以下大纲。\n问题：{state['question']}\n大纲：{ans}"
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}])
        final = res.choices[0].message.content
        return {"answer": final if final else ans}
    except:
        return {"answer": ans}

def route_after_grade(state: RAGState):
    # 👇 新增：拦截画图请求
    if any(kw in state["question"] for kw in ["画一张", "画个", "生成图片", "画图", "文生图"]):
        return "generate_image"
    # 1. 拦截各类文件生成请求（必须放在最前面）
    if any(kw in state["question"] for kw in ["PPT", "ppt", "幻灯片", "演示文稿"]):
        return "generate_ppt"
    if any(kw in state["question"] for kw in ["Word", "word", "公文"]):
        return "generate_word"
    if any(kw in state["question"] for kw in ["表格", "Excel", "excel"]):
        return "generate_excel"

    # 2. 如果没有本地文档，直接去联网
    if "【无本地文档】" in state["context"]: 
        return "web_search"

    # 3. 强制本地搜索逻辑（用户说了“结合文档”等）
    q_check = state.get("standalone_query", "") + state["question"]
    force_local_keywords = ["结合文档", "根据文档", "根据资料", "结合资料", "文中", "根据上文"]
    if any(kw in q_check for kw in force_local_keywords):
        return "research"

    # 4. 常规判断逻辑
    if state["is_relevant"] == "yes": 
        return "research"
    if state.get("retries", 0) >= 1: 
        return "web_search"
        
    return "rewrite"
        
import io  # 如果最上面没有，记得加上
import json
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def excel_generator_node(state: RAGState):
    """生成带高级样式的 Excel 表格"""
    prompt = f"""请根据以下信息，提取或生成用户需要的表格数据。
要求：必须输出标准的 JSON 格式，包含两个键：'headers'（表头列表）和 'data'（二维数组数据）。不要输出任何其他废话。

用户需求：{state['question']}
参考资料：{state['context'][:4000]}
"""
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0)
        content = res.choices[0].message.content.strip()
        if content.startswith("```json"): content = content.replace("```json", "").replace("```", "").strip()
        
        json_data = json.loads(content)
        
        # 1. 用 pandas 生成基础数据
        df = pd.DataFrame(json_data["data"], columns=json_data["headers"])
        output = io.BytesIO()
        
        # 2. 用 openpyxl 进行深度美化
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='核心数据')
            workbook = writer.book
            worksheet = writer.sheets['核心数据']
            
            # 定义样式
            header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid") # 深蓝背景
            header_font = Font(color="FFFFFF", bold=True, size=12) # 白色加粗字体
            center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
            thin_border = Border(left=Side(style='thin', color='D3D3D3'), 
                                 right=Side(style='thin', color='D3D3D3'), 
                                 top=Side(style='thin', color='D3D3D3'), 
                                 bottom=Side(style='thin', color='D3D3D3'))
            
            # 美化表头
            for col_num, column_title in enumerate(df.columns, 1):
                cell = worksheet.cell(row=1, column=col_num)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = center_align
                cell.border = thin_border
                worksheet.row_dimensions[1].height = 25  # 调高表头行高
                
            # 美化正文（斑马线、居中、边框）
            for row_num in range(2, len(df) + 2):
                for col_num in range(1, len(df.columns) + 1):
                    cell = worksheet.cell(row=row_num, column=col_num)
                    cell.alignment = center_align
                    cell.border = thin_border
                    # 偶数行加浅灰背景
                    if row_num % 2 == 0:
                        cell.fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
                        
            # 自适应列宽
            for col_num, column_title in enumerate(df.columns, 1):
                column_letter = get_column_letter(col_num)
                max_length = max(len(str(column_title)), max([len(str(val)) for val in df.iloc[:, col_num-1]]))
                worksheet.column_dimensions[column_letter].width = min(max_length + 4, 50) # 限制最大宽度
                
            # 冻结首行
            worksheet.freeze_panes = "A2"

        output.seek(0)
        return {
            "file_data": output.getvalue(),
            "file_name": "高颜值数据报表.xlsx",
            "answer": "✅ 带高级排版的 Excel 数据表已生成，请下载查看！"
        }
    except Exception as e:
        return {"answer": f"表格生成失败：{e}"}
    

def word_generator_node(state: RAGState):
    """生成带高级排版的 Word 文档"""
    prompt = f"""请根据以下信息，撰写一份完整的 Word 文档。
要求：只输出正文内容，使用 '#' 表示一级标题，'##' 表示二级标题，'###' 表示三级标题，'-' 表示列表项。不要包含任何解释。

用户需求：{state['question']}
参考资料：{state['context'][:5000]}
"""
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.3)
        text_content = res.choices[0].message.content.strip()

        doc = Document()
        
        # 设置全局默认字体
        style = doc.styles['Normal']
        style.font.name = 'Microsoft YaHei'  # 微软雅黑
        style.font.size = Pt(11)
        
                # 解析文本并写入
        for line in text_content.split('\n'):
            line = line.strip()
            if not line: continue
            # 忽略掉大模型可能输出的四级标题符号
            if line.startswith('#### '):
                heading = doc.add_heading(line.replace('#### ', ''), level=3)
                heading.style.font.name = 'Microsoft YaHei'
                heading.runs[0].font.color.rgb = DocxRGBColor(0x1F, 0x4E, 0x78)
            elif line.startswith('### '):
                heading = doc.add_heading(line.replace('## ', ''), level=2)
                heading.style.font.name = 'Microsoft YaHei'
                heading.runs[0].font.color.rgb = DocxRGBColor(0x1F, 0x4E, 0x78) # 统一用 DocxRGBColor
            elif line.startswith('# '):
                heading = doc.add_heading(line.replace('# ', ''), level=1)
                heading.style.font.name = 'Microsoft YaHei'
                heading.runs[0].font.color.rgb = DocxRGBColor(0x00, 0x00, 0x00) # 统一用 DocxRGBColor
                heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif line.startswith('- '):
                p = doc.add_paragraph(style='List Bullet')
                run = p.add_run(line.replace('- ', ''))
                run.font.name = 'Microsoft YaHei'
                p.paragraph_format.line_spacing = 1.5
            else:
                p = doc.add_paragraph()
                run = p.add_run(line)
                run.font.name = 'Microsoft YaHei'
                p.paragraph_format.line_spacing = 1.5
                # 注意这里是 12 个空格缩进，必须在 else 块里
                p.paragraph_format.first_line_indent = Inches(0.3) # 用 Inches 替代 Pt 避免冲突

        output = io.BytesIO()
        doc.save(output)
        output.seek(0)

        return {
            "file_data": output.getvalue(),
            "file_name": "专业公文报告.docx",
            "answer": "✅ 带公文排版的 Word 文档已生成，请点击下方按钮下载。"
        }
    except Exception as e:
        return {"answer": f"Word 生成失败：{e}"}
def ppt_generator_node(state: RAGState):
    """带风格切换的 PPT 生成节点"""
    # 1. 确定风格：如果是智能推荐，让大模型判断
    style_name = state.get("ppt_style", "极简商务风")
    if style_name == "AI智能推荐":
        style_prompt = f"分析以下内容，判断最适合的PPT风格。只输出风格名称，必须从以下三个中选一个：[科技深邃风, 极简商务风, 活力创意风]。内容：{state['question']} {state['context'][:500]}"
        try:
            style_res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": style_prompt}], temperature=0)
            style_name = style_res.choices[0].message.content.strip()
        except:
            style_name = "极简商务风"
    
    # 如果大模型乱说，默认兜底为商务风
    if style_name not in PPT_THEMES:
        style_name = "极简商务风"
    theme = PPT_THEMES[style_name]

    # 2. 生成内容结构
    prompt = f"""请根据以下信息策划 PPT。输出严格 JSON 格式：'slides' 列表。
每页包含 'title'（标题）和 'bullet_points'（3-5个要点）。不要有任何多余解释。

用户需求：{state['question']}
参考资料：{state['context'][:5000]}
"""
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.2)
        content = res.choices[0].message.content.strip()
        if content.startswith("```json"): content = content.replace("```json", "").replace("```", "").strip()
        ppt_data = json.loads(content)
        
        # 3. 渲染引擎
        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        blank_layout = prs.slide_layouts[6]
        
        # --- 封面页 ---
        slide = prs.slides.add_slide(blank_layout)
        bg = slide.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid(); bg.fill.fore_color.rgb = theme["bg"]; bg.line.fill.background()

        title_box = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11), Inches(2))
        p = title_box.text_frame.add_paragraph()
        p.text = ppt_data["slides"][0].get("title", "AI 汇报")
        p.font.size = Pt(54); p.font.bold = True; p.font.color.rgb = theme["title"]; p.alignment = PP_ALIGN.CENTER

        subtitle_box = slide.shapes.add_textbox(Inches(1), Inches(4.5), Inches(11), Inches(1))
        p_sub = subtitle_box.text_frame.add_paragraph()
        p_sub.text = f"由 Agentic RAG 生成 | 风格：{style_name}"
        p_sub.font.size = Pt(24); p_sub.font.color.rgb = theme["text"]; p_sub.alignment = PP_ALIGN.CENTER

        # --- 内容页 ---
        for slide_data in ppt_data["slides"][1:]:
            slide = prs.slides.add_slide(blank_layout)
            
            # 背景
            bg = slide.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
            bg.fill.solid(); bg.fill.fore_color.rgb = theme["bg"]; bg.line.fill.background()
            
            # 左侧强调竖条装饰
            accent_bar = slide.shapes.add_shape(1, Inches(0.8), Inches(1), Inches(0.15), Inches(5.5))
            accent_bar.fill.solid(); accent_bar.fill.fore_color.rgb = theme["accent"]; accent_bar.line.fill.background()

            # 标题
            title_box = slide.shapes.add_textbox(Inches(1.2), Inches(1), Inches(11), Inches(1))
            p = title_box.text_frame.add_paragraph()
            p.text = slide_data.get("title", "标题")
            p.font.size = Pt(36); p.font.bold = True; p.font.color.rgb = theme["title"]

            # 正文
            body_box = slide.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(10.5), Inches(4.5))
            tf_body = body_box.text_frame; tf_body.word_wrap = True
            for i, point in enumerate(slide_data.get("bullet_points", [])):
                p = tf_body.paragraphs[0] if i == 0 else tf_body.add_paragraph()
                p.text = f"•  {point}"
                p.font.size = Pt(24); p.font.color.rgb = theme["text"]; p.space_after = Pt(15)

        output = io.BytesIO()
        prs.save(output); output.seek(0)
        return {
            "file_data": output.getvalue(),
            "file_name": f"汇报_{style_name}.pptx",
            "answer": f"✅ 已生成【{style_name}】风格的 PPT 演示文稿！"
        }
    except Exception as e:
        return {"answer": f"PPT 生成失败：{e}"}

def image_generator_node(state: RAGState):
    """文生图节点（带提示词扩写）"""
    user_prompt = state.get("rewritten_question") or state["question"]
    
    # 1. 让 DeepSeek 先扩写提示词
    prompt_enhance = f"""你是一个顶级的 AI 绘画提示词专家。请将用户的简单指令扩写为一段细节丰富、画面精美的绘画提示词。
要求：包含主体、环境、光影、画质、风格等描述。直接输出扩写后的提示词，不要任何废话。

用户指令：{user_prompt}
扩写后的提示词："""
    
    try:
        res_llm = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt_enhance}], temperature=0.7)
        enhanced_prompt = res_llm.choices[0].message.content.strip()
    except:
        enhanced_prompt = f"高质量，杰作，精美的画面：{user_prompt}"

    # 2. 调用智谱 CogView 画图
    url = "https://open.bigmodel.cn/api/paas/v4/images/generations"
    headers = {"Authorization": f"Bearer {zhipu_key}", "Content-Type": "application/json"}
    data = {
        "model": "cogview-3-flash",
        "prompt": enhanced_prompt
    }
    
    try:
        res = requests.post(url, headers=headers, json=data)
        res.raise_for_status()
        image_url = res.json()["data"][0]["url"]
        
        # 下载图片到内存
        img_bytes = requests.get(image_url).content
        
        return {
            "file_data": img_bytes,
            "file_name": "AI生成图片.png",
            "answer": f"✅ 您的图片已生成完毕！\n\n**画图原指令**：{user_prompt}\n\n**AI扩写提示词**：{enhanced_prompt}"
        }
    except Exception as e:
        return {"answer": f"⚠️ 图片生成失败：{e}"}
# ================= 反思节点（数据飞轮） =================
def reflection_node(state: RAGState):
    """当任务失败时，自己反思并存入经验池"""
    prompt = f"""任务失败了。用户的问题是：{state['question']}。
我们得到的上下文是：{state.get('context', '无')[:500]}
请分析失败原因（比如：搜索词太宽泛？没抓到实时数据？），并生成一条简短的“经验教训”，用于下次改进。只输出教训，不要废话。"""
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.3)
        lesson = res.choices[0].message.content.strip()
        
        # 存入数据库
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO experiences (task_type, lesson) VALUES (?, ?)", ("general", lesson))
        conn.commit()
        conn.close()
        return {"answer": f"⚠️ 任务执行失败，但 AI 已经学到了：{lesson}"}
    except Exception as e:
        return {"answer": f"任务失败，反思也失败了：{e}"}
    
    
# ================= 4. 构建状态图 =================
workflow = StateGraph(RAGState)
workflow.add_node("rewrite_query", rewrite_query_with_history)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("research", research_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reflection", reflection_node) # 👈 新增反思节点
workflow.add_node("generate_excel", excel_generator_node)
workflow.add_node("generate_word", word_generator_node)
workflow.add_node("generate_ppt", ppt_generator_node)
workflow.add_node("generate_image", image_generator_node) # 👈 新增节点

workflow.set_entry_point("rewrite_query")
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("retrieve", "grade")

# 👇 核心修改：条件边里加上 generate_image
workflow.add_conditional_edges("grade", route_after_grade, {
    "research": "research", 
    "rewrite": "rewrite", 
    "web_search": "web_search",
    "generate_excel": "generate_excel",
    "generate_word": "generate_word",
    "generate_ppt": "generate_ppt",
    "generate_image": "generate_image" # 👈 新增
})

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("web_search", "research")
workflow.add_edge("research", "writer")
workflow.add_edge("writer", END)
workflow.add_edge("generate_excel", END)
workflow.add_edge("generate_word", END)
workflow.add_edge("generate_ppt", END)
workflow.add_edge("generate_image", END) # 👈 新增终点

app_graph = workflow.compile()

# ================= 5. 侧边栏与聊天界面 =================

# 1. 初始化当前会话状态
if "current_conv_id" not in st.session_state:
    # 默认加载最新一个会话，如果没有则创建新的
    convs = get_conversations()
    if convs:
        st.session_state.current_conv_id = convs[0][0]
    else:
        st.session_state.current_conv_id = create_conversation("欢迎对话")

# 2. 侧边栏：历史记录
with st.sidebar:
    st.title("💬 历史对话")
    # 显示用户余额
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT balance FROM users WHERE id = 'default_user'")
    balance = c.fetchone()
    conn.close()
    if not balance:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("INSERT INTO users (id) VALUES ('default_user')")
        conn.commit(); conn.close()
        balance = (100.0,)
    st.markdown(f"💰 **账户余额：¥{balance[0]:.2f}**")
    st.title("💬 历史对话")
    
    # 新建对话按钮
    if st.button("➕ 新建对话", use_container_width=True):
        new_id = create_conversation("新对话")
        st.session_state.current_conv_id = new_id
        st.rerun() # 刷新页面载入新会话

    st.divider()
    
    # 列出所有历史
    convs = get_conversations()
    for cid, ctitle in convs:
        # 高亮当前选中的对话
        if cid == st.session_state.current_conv_id:
            st.button(f"👉 {ctitle}", key=f"btn_{cid}", disabled=True, use_container_width=True)
        else:
            if st.button(f"{ctitle}", key=f"btn_{cid}", use_container_width=True):
                st.session_state.current_conv_id = cid
                st.rerun()
    st.divider()
    
    st.subheader("🎨 PPT 生成设置")
    ppt_style_choice = st.selectbox(
        "选择风格（生成PPT时生效）:",
        ["AI智能推荐", "科技深邃风", "极简商务风", "活力创意风"]
    )

# 3. 加载当前会话的消息记录
current_messages = get_messages(st.session_state.current_conv_id)

# 4. 文件上传区
st.markdown("上传文档 (PDF, Word, Excel, PPT, 图片)")
uploaded_files = st.file_uploader("", type=["txt", "pdf", "docx", "md", "pptx", "xlsx", "xls", "jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed")

# 【新增】：如果用户上传了新文件，保存到本地磁盘
if uploaded_files:
    save_uploaded_files(st.session_state.current_conv_id, uploaded_files)

# 【新增】：如果用户没传文件，但本地有这个会话的历史文件，就自动加载！
if not uploaded_files:
    uploaded_files = load_local_files(st.session_state.current_conv_id)
    if uploaded_files:
        st.info(f"💡 自动为您加载该会话的历史文档：{', '.join([f.name for f in uploaded_files])}")

# 建库逻辑
if uploaded_files:
    current_file_names = [f.name for f in uploaded_files]
    if "current_files" not in st.session_state or st.session_state.current_files != current_file_names:
        with st.spinner("正在解析文档并构建索引..."):
            chunks_global = process_files(uploaded_files)
            if chunks_global:
                chroma_client = chromadb.Client()
                try: chroma_client.delete_collection(name="my_docs")
                except: pass
                col = chroma_client.create_collection(name="my_docs")
                
                # ================= 【核心加速替换区】 =================
                # 删掉你原来的 for i, chunk in enumerate(chunks_global): col.add(...) 循环
                # 换成下面这个多线程并发版：
                def add_chunk_to_col(i, chunk):
                    try:
                        vec = get_embedding(chunk)
                        col.add(embeddings=[vec], documents=[chunk], ids=[f"id_{i}"])
                    except Exception:
                        pass 
                
                # 开启 5 个并发线程同时算向量，速度提升 5 倍！
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(add_chunk_to_col, i, chunk) for i, chunk in enumerate(chunks_global)]
                    concurrent.futures.wait(futures)
                # ======================================================

                st.session_state.collection = col
                st.session_state.bm25 = BM25Okapi([tokenize(c) for c in chunks_global])
                st.session_state.chunks_global = chunks_global
                st.session_state.current_files = current_file_names
                
                st.success(f"成功加载 {len(uploaded_files)} 份文档（已开启多线程加速）。")
# 5. 渲染历史消息
for msg in current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. 输入框与对话逻辑
if prompt := st.chat_input("你可以针对文档提问，或者直接让我上网查资料..."):
    # 每次发新消息，先清空上一轮的文件状态
    st.session_state.generated_file = None
    
    # 如果是新会话的第一句话，更新标题
    if len(current_messages) == 0:
        update_conv_title(st.session_state.current_conv_id, prompt[:15])

    # 保存用户消息到数据库
    save_message(st.session_state.current_conv_id, "user", prompt)
    
    # 显示用户气泡
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示 AI 气泡
    with st.chat_message("assistant"):
        with st.spinner("Agent 正在思考与检索..."):
            try:
                history_for_llm = current_messages[-6:]
                ppt_style = st.session_state.get("ppt_style_choice", "AI智能推荐")
                
                # 调用图
                events = app_graph.stream({
                    "question": prompt, 
                    "rewritten_question": "", 
                    "retries": 0, 
                    "history": history_for_llm,
                    "ppt_style": ppt_style
                }, stream_mode="values")
                
                final_result = None
                for event in events: 
                    final_result = event

                # 提取文本并兜底
                final_text = final_result.get("answer", "") if final_result else ""
                
                # 👇 新增：如果大模型返回空、或者带有报错字眼，触发反思！
                if not final_text.strip() or "⚠️" in final_text:
                    # 直接调用你的反思节点逻辑
                    reflection_result = reflection_node(final_result if final_result else {"question": prompt, "context": ""})
                    final_text = reflection_result.get("answer", "任务失败")

                st.markdown(final_text)
                if not final_text or not final_text.strip():
                    final_text = "⚠️ 抱歉，系统未能生成有效回答。请尝试换种问法或检查网络。"

                # 核心：把文件流存进全局状态
                if final_result and final_result.get("file_data") is not None:
                    st.session_state.generated_file = final_result

                # 👇 只在这里打印一次！
                st.markdown(final_text)

            except Exception as e:
                final_text = f"❌ 执行出错：{str(e)}"
                st.markdown(final_text)

    # 保存 AI 消息到数据库，并记录成本（这里简单按次扣费，真实场景按 Token 算）
    call_cost = 0.05 
    save_message(st.session_state.current_conv_id, "assistant", final_text, call_cost)
    
    # 扣除用户余额
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET balance = balance - ? WHERE id = 'default_user'", (call_cost,))
    conn.commit()
    conn.close()
    # 强制刷新页面，加载最新历史
    st.rerun()


# ================= 底部固定渲染最新文件下载 =================
if "generated_file" in st.session_state and st.session_state.generated_file is not None:
    res = st.session_state.generated_file
    fname = res.get("file_name", "data")
    
    st.markdown("---")
    st.success(f"📄 最新文件已就绪：{fname} （请点击下载）")
    
    if fname.endswith(".xlsx"):
        st.download_button("📥 下载 Excel 表格", data=res["file_data"], file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx_btn")
    elif fname.endswith(".docx"):
        st.download_button("📥 下载 Word 文档", data=res["file_data"], file_name=fname, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="dl_docx_btn")
    elif fname.endswith(".pptx"):
        st.download_button("📥 下载 PPT 演示文稿", data=res["file_data"], file_name=fname, mime="application/vnd.openxmlformats-officedocument.presentationml.presentation", key="dl_pptx_btn")
    
    # 👇 新增：图片展示与下载
    elif fname.endswith(".png") or fname.endswith(".jpg"):
        st.image(res["file_data"], caption="AI 为你生成的图片", use_container_width=True)
        st.download_button("📥 下载图片", data=res["file_data"], file_name=fname, mime="image/png", key="dl_img_btn")