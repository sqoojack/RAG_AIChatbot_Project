# Command line: streamlit run app.py
from utils import st
from config import available_img_models, available_llm_models  # 匯入模型選項
# Question: PROD備份到CLONE5之後要做哪些工作項目？

st.set_page_config(
    page_title="RAG 問答系統",
    page_icon="🤖",
    layout="wide"
)

st.title("RAG 問答系統 🎯")
st.markdown("""
### 歡迎使用本系統，請透過左側選單選擇功能：

- 📂 **Knowledge_Interface**: 上傳檔案並建立向量資料庫  
- 💬 **Answering_Interface**: 輸入問題並獲取LLM回答
""")

# 初始化模型設定參數（僅第一次執行時）
if "model_settings" not in st.session_state:
    st.session_state.model_settings = {
        "img_model": available_img_models[0],
        "llm_model": available_llm_models[0],
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 10,
        "search_method": "Basic",
        "chunk_size": 200,
        "chunk_overlap": 50
    }

