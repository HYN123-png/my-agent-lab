import json

# 1. 打开刚才建的 notes.json 文件，把它读出来
with open("notes.json", "r", encoding="utf-8") as f:
    notes = json.load(f)

# 2. 遍历每一条笔记，在标题前面加上【待办】
for note in notes:
    note["title"] = "【待办】" + note["title"]

# 3. 把修改后的结果存到一个新文件 notes_todo.json 里
with open("notes_todo.json", "w", encoding="utf-8") as f:
    json.dump(notes, f, ensure_ascii=False, indent=2)

print("处理完成，共处理了", len(notes), "条笔记！")