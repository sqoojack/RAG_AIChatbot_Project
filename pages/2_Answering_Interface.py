# Answering_Interface.py
from utils import os, json, st 
from langchain.schema import Document  
from UI_model_select import render_model_settings_ui  
from RAG_Embedding import load_vectorstore
from RAG_LLM_Generator import llm_generator, extract_answer_and_thought 
from reranking import search_top_k, rerank_chunks_top_k  
from full_file import generate_full_files_answer  # Custom RAG 處理邏輯
from config import default_model_settings, ollama_url, VECTOR_DB_DIR, reranking_url, reranking_api, cert_datapath  

# 設定 Streamlit 頁面資訊
st.set_page_config(page_title="問答介面", page_icon="💬")
st.title("💬 問答介面")

# 初始化 model_settings（從 Session State 取得或套用預設值）
if "model_settings" not in st.session_state:
    st.session_state.model_settings = default_model_settings.copy()

# 初始化向量資料庫與文件列表
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs" not in st.session_state:
    st.session_state.docs = []

# 搜尋本地資料夾中所有向量資料庫（資料夾名稱 = DB 名稱）
vector_db_names = [f for f in os.listdir(VECTOR_DB_DIR) if os.path.isdir(os.path.join(VECTOR_DB_DIR, f))]

# 使用 selectbox 讓使用者選擇一個資料庫
selected_db = st.selectbox("📂 **選擇已建立的向量資料庫**", options=vector_db_names)

# 如果使用者有選擇資料庫，則嘗試讀取向量資料與原始文件
if selected_db:
    try:
        db_path = os.path.join(VECTOR_DB_DIR, selected_db)

        # 載入向量資料庫（透過 Ollama 的 Embedding Model）
        vectorstore = load_vectorstore(
            db_path,
            embedding_model=st.session_state.model_settings.get("embedding_model", "bge-m3"),
        )
        st.session_state.vectorstore = vectorstore

        # 讀取原始文字內容（例如頁面內容與來源資訊）放入 docs
        metadata_path = os.path.join(db_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_list = json.load(f)

            docs_list = []
            if isinstance(metadata_list, list):
                # metadata.json 是 List 格式
                for meta in metadata_list:
                    page_content = meta.get("page_content", "")
                    meta_data = meta.get("metadata", {})
                    docs_list.append(Document(page_content=page_content, metadata=meta_data))
            elif isinstance(metadata_list, dict):
                # metadata.json 是單一 Document 格式
                page_content = metadata_list.get("page_content", "")
                meta_data = metadata_list.get("metadata", {})
                docs_list.append(Document(page_content=page_content, metadata=meta_data))
            st.session_state.docs = docs_list
        else:
            st.session_state.docs = []

    except Exception as e:
        st.error(f"❌ 向量資料庫載入失敗：{e}")
        st.stop()
else:
    st.warning("⚠️ 尚未建立向量資料庫，請先至【建立知識庫】頁上傳檔案。")
    st.stop()

# 顯示 LLM 模型與參數設定選單（top_p、temperature 等）
render_model_settings_ui()

# 使用者輸入問題
query = st.text_area("**請輸入你的問題**", value="PROD備份到CLONE5之後要做哪些工作項目?")

# 啟動問答流程
if st.button("回答問題"):
    if not query.strip():
        st.warning("請輸入問題")
    else:
        try:
            with st.spinner("檢索中..."):
                # 根據設定選擇搜尋方法：Basic / Reranking / MMR / Custom RAG
                search_method = st.session_state.model_settings.get("search_method", "Basic")
                vectorstore = st.session_state.vectorstore

                if search_method == "MMR":
                    # 採用 Maximal Marginal Relevance 檢索
                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
                    )
                    mmr_docs = retriever.invoke(query)
                    top_chunks = [(doc, 1.0) for doc in mmr_docs]
                    st.success("✅ MMR檢索完成")

                elif search_method in ["Reranking", "Custom RAG"]:
                    # 使用向量搜尋初步取得 top_k 候選片段
                    candidates = search_top_k(query, vectorstore, top_k=20)
                    # 再透過 Reranker 精選 top_k 片段
                    top_chunks = rerank_chunks_top_k(
                        query,
                        candidates,
                        top_k=st.session_state.model_settings.get("top_k", 5),
                        reranking_url=reranking_url,
                        reranking_api=reranking_api,
                        cert_datapath=cert_datapath
                    )
                    st.success("✅ Reranking完成")

                else:
                    # 採用最基本的向量 Top-K 搜尋
                    top_chunks = search_top_k(
                        query,
                        vectorstore,
                        top_k=st.session_state.model_settings.get("top_k", 5)
                    )
                    st.success("✅ Basic 檢索完成")

            with st.spinner("生成回答中..."):
                if search_method == "Custom RAG":
                    # 使用完整檔案資訊結合片段做回答（適用長文件）
                    file_chunks, answer = generate_full_files_answer(
                        top_chunks,
                        st.session_state.docs,
                        query,
                        ollama_url,
                        st.session_state.model_settings
                    )
                else:
                    # 傳送 query + top_chunks 給 LLM 模型生成回答
                    answer = llm_generator(
                        query,
                        top_chunks,
                        ollama_url,
                        llm_model=st.session_state.model_settings.get("llm_model"),
                        temperature=st.session_state.model_settings.get("temperature", 0.0),
                        top_p=st.session_state.model_settings.get("top_p", 1.0),
                    )

                # 分離 LLM 輸出：最終答案 vs 思考過程
                ans, thought = extract_answer_and_thought(answer)

                # 分為左右區塊顯示回答與來源片段
                left, right = st.columns([2, 1])
                with left:
                    st.markdown("### 📘 回答：")
                    st.markdown(ans)
                    if thought:
                        with st.expander("💭 思考過程"):
                            st.markdown(thought)
                with right:
                    st.markdown("### 🔍 匹配段落：")
                    if search_method == "Custom RAG":
                        for i, doc in enumerate(file_chunks):
                            with st.expander(f"段落 {i+1} | 來源: {doc.metadata.get('source')}"):
                                st.write(doc.page_content)
                    else:
                        for i, (doc, score) in enumerate(top_chunks):
                            with st.expander(f"Rank {i+1} | 來源: {doc.metadata.get('source')} | 分數: {score:.2f}"):
                                st.write(doc.page_content)

        except Exception as e:
            st.error(f"❌ 發生錯誤：{e}")
