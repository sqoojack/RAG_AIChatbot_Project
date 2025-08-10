# UI_model_select.py
from utils import st, datetime
from config import available_llm_models, available_img_models

# 知識庫建立的設定介面
def render_create_DB_select():
    
    with st.expander("⚙️ **建立知識庫參數設定**", expanded=True):
        # 選擇圖片 Embedding 模型，預設是 session_state 中的 img_model
        img_model = st.selectbox(
            "**選擇處理圖片的模型**",
            options=available_img_models,
            index=available_img_models.index(st.session_state.model_settings.get("img_model", available_img_models[0]))
        )
        chunk_size = st.slider("chunk_size", 200, 2000, st.session_state.model_settings["chunk_size"])
        chunk_overlap =  st.slider("chunk_overlap", 50, 600, st.session_state.model_settings["chunk_overlap"])
        
        if st.button("✅ 儲存參數", key="save_img_model_settings"):
            st.session_state.model_settings.update({
                "img_model": img_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            })
            st.success("參數已儲存！")

# 渲染問答模型相關參數設定的 UI，讓使用者能在介面上調整參數
def render_model_settings_ui():
    
    with st.expander("⚙️ **問答模型參數設定**", expanded=True):

        if "model_settings" not in st.session_state:
            st.session_state.model_settings = {
                "llm_model": available_llm_models[0],  # 預設語言模型
                "temperature": 0.7,                     # 預設溫度
                "top_p": 0.9,                           # 預設 top_p
                "top_n": 10,
                "top_k": 5,                             # 預設 top_k
                "search_method": "Basic",                 # 預設檢索模式
                "chunk_size": 200,
                "chunk_overlap": 50
            }
        
        # 選擇使用的語言模型，預設值從 session_state 讀取
        llm_model = st.selectbox(
            "**選擇語言模型**",
            options=available_llm_models,
            index=available_llm_models.index(st.session_state.model_settings["llm_model"])
        )
        
        search_method = st.radio(
            "🔍 **選擇檢索模式**",
            options=["MMR", "Reranking", "Custom RAG", "Basic"],
            index=0,
            horizontal=True
        )
        # 使用滑桿設定
        temperature = st.slider("溫度 (temperature)", 0.0, 1.0, st.session_state.model_settings["temperature"])
        top_p = st.slider("Top-p", 0.0, 1.0, st.session_state.model_settings["top_p"])
        if search_method != "Basic":
            top_n = st.slider("Top-n", 2, 30, st.session_state.model_settings["top_n"])
        top_k = st.slider("Top-k", 1, 20, st.session_state.model_settings["top_k"])

        # 用 radio 按鈕讓使用者選擇檢索模式，有 MMR、Reranking、Custom RAG、Basic 四種選項，預設選擇第 0 個（MMR）
        # horizontal=True 讓選項橫向排列，更節省空間


        if search_method == "Basic":
            # 按鈕用於儲存目前 UI 上調整的參數，按下後會更新 session_state
            if st.button("✅ 儲存參數", key="save_qa_model_settings"):
                st.session_state.model_settings.update({
                    "llm_model": llm_model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "search_method": search_method
                })
                st.success("參數已儲存！")
        else:
            if st.button("✅ 儲存參數", key="save_qa_model_settings"):
                st.session_state.model_settings.update({
                    "llm_model": llm_model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_n": top_n,
                    "top_k": top_k,
                    "search_method": search_method
                })
                st.success("參數已儲存！")

# 將向量資料庫、文件列表與檔案資訊存入 session_state，方便跨頁面或重新整理後持續使用
def save_vectorstore_to_session(vectorstore, docs, uploaded_files):
    
    st.session_state.vectorstore = vectorstore
    st.session_state.docs = docs
    # 取得上傳檔案名稱並排序，方便後續顯示與比對
    st.session_state.file_names = sorted([f.name for f in uploaded_files or []])
    
    # 取得當下時間作為上傳時間戳記
    st.session_state.upload_timestamp = datetime.now()

