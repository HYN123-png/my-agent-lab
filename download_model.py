from modelscope import snapshot_download

print("正在从 ModelScope 下载模型...")
snapshot_download(
    'BAAI/bge-reranker-base',
    local_dir='./models/bge-reranker-base'
)
print("🎉 模型下载完成！")