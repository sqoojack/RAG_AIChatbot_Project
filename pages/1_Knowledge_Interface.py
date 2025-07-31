# Knowledge_Interface.py

from utils import os, st
from UI_model_select import render_create_DB_select
from config import VECTOR_DB_DIR

from RAG_Embedding import load_metadata, delete_files_from_db, create_new_vector_db, delete_vector_db, rebuild_vector_db

# 初始化使用者的最後操作狀態（用於觸發 UI 更新）
if "last_action" not in st.session_state:
    st.session_state.last_action = None  # 可為 add/delete/create/delete_db

# 設定頁面為寬版，並建立左右兩欄
st.set_page_config(layout="wide")
col_left, col_right = st.columns([1, 1])

# ===== 左邊：建立新的資料庫 =====
with col_left:
    st.header("📂 新增知識庫")

    # 模型選擇（例如選擇圖像處理模型 gemma3:27b 等）
    render_create_DB_select()

    # 上傳用來建立新資料庫的檔案
    uploaded_files = st.file_uploader(
        "**上傳 PDF, Word, PPT, 純文字或圖片檔案**",
        type=["pdf", "ppt", "pptx", "docx", "txt", "png", "jpg", "jpeg", "mp3"],
        accept_multiple_files=True,
        key="new_db_files"
    )

    # 資料庫命名輸入欄位
    db_name = st.text_input("**請為這次建立的資料庫命名 (英文或數字)**", key="new_db_name")
    build_btn = st.button("🚧 建立向量資料庫", key="build_new_db")

    # 建立新資料庫並進行向量化處理
    if build_btn:
        if not db_name:
            st.warning("請輸入資料庫名稱")
        elif not uploaded_files:
            st.warning("請先上傳檔案")
        else:
            try:
                chunk_size = st.session_state.model_settings["chunk_size"]
                chunk_overlap = st.session_state.model_settings["chunk_overlap"]
                count = create_new_vector_db(db_name, uploaded_files, st.session_state.model_settings["img_model"], first_time=True, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.success(f"✅ 資料庫「{db_name}」建立完成，共 {count} 個chunk")
                st.session_state.last_action = "create"
                st.stop()
            except FileExistsError:
                st.error(f"❗ 資料庫「{db_name}」已存在，請換名稱")


# ===== 右邊：知識庫管理 =====
with col_right:
    st.header("📂 管理知識庫")
    chunk_size = st.session_state.model_settings["chunk_size"]
    chunk_overlap = st.session_state.model_settings["chunk_overlap"]
    
    # 取得已存在的資料庫資料夾列表
    db_list = sorted([d for d in os.listdir(VECTOR_DB_DIR) if os.path.isdir(os.path.join(VECTOR_DB_DIR, d))])
    selected_db = None

    # 資料庫選單區塊
    with st.expander("📚 **已有資料庫列表**", expanded=True):
        if db_list:
            selected_db = st.selectbox("**選擇資料庫查看細節與管理**", [""] + db_list)
        else:
            st.write("尚無任何已建立的資料庫")
    
    # 如果使用者選取了某個資料庫，顯示其詳細資訊與操作功能
    if selected_db:
        meta = load_metadata(selected_db)  # 讀取 metadata.json
        st.markdown(f"### 資料庫：`{selected_db}`")
        st.write("上次編輯時間:", meta.get("last_edit", "未知"))
        st.write("chunk_size：", meta.get("chunk_size", 0))

        files = meta.get("files", [])
        st.write("包含檔案：")
        

        if files:
            # 建立一個 dict 用來儲存勾選狀態
            selected_to_delete = {}
            for idx, f_name in enumerate(files, start=1):
                col1, col2 = st.columns([0.1, 0.9])
                col1.markdown(f"{idx}.")
                selected_to_delete[f_name] = col2.checkbox(
                    label=f_name,
                    key=f"checkbox_{f_name}"
                )

            # 統一刪除按鈕
            if st.button("❌ 刪除勾選檔案"):
                files_to_delete = [f for f, checked in selected_to_delete.items() if checked]
                count = delete_files_from_db(selected_db, files_to_delete, chunk_overlap)
                st.success(f"✅ 資料庫「{selected_db}」更新完成，共 {count} 個chunk")     
        else:
            st.write("（目前無任何檔案）")

    # ===== 整庫刪除功能 =====
        with st.expander("🗑️ 刪除整個資料庫", expanded=False):
            confirmed = st.checkbox("⚠️ 確認刪除此資料庫，操作不可恢復！", key="confirm_delete_db")
            if st.button("刪除資料庫", key="delete_db_btn"):
                if not confirmed:
                    st.warning("請先勾選確認刪除")
                else:
                    success = delete_vector_db(selected_db)
                    if success:
                        st.success(f"✅ 資料庫 `{selected_db}` 已成功刪除！")
                        st.session_state.last_action = "delete_db"
                        st.stop()
                    else:
                        st.error("❌ 資料庫刪除失敗，請稍後再試。")
    
        # ===== 為現有資料庫新增檔案功能 =====
        st.markdown("### ➕ 新增檔案到此資料庫")
        uploaded_files = st.file_uploader(
            "**選擇檔案**",
            type=["pdf", "ppt", "pptx", "docx", "txt", "png", "jpg", "jpeg", "mp3"],
            accept_multiple_files=True,
            key="add_files"
        )

        # 按下按鈕後，將檔案新增到對應資料庫，並重建向量資料庫
        if st.button("🚀 新增並重建向量資料庫", key="upload_new_files"):
            if not uploaded_files:
                st.warning("請先上傳檔案")
            else:
                count = rebuild_vector_db(selected_db, chunk_overlap, uploaded_files)
                st.success(f"✅ 建立完成，共 {count} 個chunk")
                st.session_state.last_action = "add"
                st.stop()
    
