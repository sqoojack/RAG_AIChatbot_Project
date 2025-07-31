
# 測試能不能呼叫reranking model
def test_reranking_model():
    import requests

    url = "https://api.siliconflow.cn/v1/rerank"
    payload = {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "蘋果有什麼功能？",
        "documents": [
            "蘋果有防癌功效",
            "蘋果是iPhone的品牌"
        ]
    }
    headers = {"Authorization": "Bearer sk-apvcmrsiyuolomnhiieejdgvndouwqalhzwhgxdzldrdipxc"}

    response = requests.post(url, json=payload, headers=headers, verify=r"D:\PHCheng\certificate\PHCheng.pem")
    print(f"HTTP code: {response.status_code}")
    print("API response:", response.text)


import sys
import streamlit as st

print("✅ Python 執行路徑：", sys.executable)
print("✅ Streamlit 版本：", st.__version__)
