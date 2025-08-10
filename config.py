# config.py
import os
# 可用的語言模型選項
available_llm_models = [
    "deepseek-r1:8b",
    "qwen3:14b",
    "qwen2.5:7b",
    "gemma3:27b",
    "deepseek-r1:32b",
    "llava:7b",
    "llava-llama3:8b",
    "llama4:16x17b",
    "gpt-oss:20b"
]

# 可用的圖像描述模型選項
available_img_models = [
    "gemma3:27b",
    "llava:7b",
    "llava-llama3:8b",
    "llama4:16x17b"
]

# 預設參數設定
default_model_settings = {
    "llm_model": "deepseek-r1:8b",
    "img_model": "gemma3:27b",
    "embedding_model": "bge-m3",
    "temperature": 0.0,
    "top_p": 0.9,
    "top_n": 10,
    "top_k": 5,
}


# 預設 Ollama API URL
# ollama_url = "http://172.20.5.116:11434"
ollama_url = "http://127.0.0.1:11434"   # After ssh server connect

embedding_model = "bge-m3"
reranking_url = "https://api.siliconflow.cn/v1/rerank"
reranking_api = "sk-apvcmrsiyuolomnhiieejdgvndouwqalhzwhgxdzldrdipxc"


#特別憑證路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cert_datapath = os.path.join(BASE_DIR, "PHCheng.pem")

#資料庫路徑
VECTOR_DB_DIR = "vectorstores"