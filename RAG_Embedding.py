# RAG_Embedding.py
from utils import os, json, List, requests, datetime, shutil
from io import BytesIO
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from Upload_file import process_uploaded_files
from config import embedding_model, ollama_url, VECTOR_DB_DIR

# 確保資料夾存在
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

#========== Metadata 儲存與載入 =============

# 儲存 info.json：包含資料庫描述資訊，如上傳檔案、模型名稱、切片大小等
def save_metadata(metadata: dict, db_name: str):

    path = os.path.join(VECTOR_DB_DIR, db_name, "info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

# 載入資料庫描述資訊（info.json）
def load_metadata(db_name: str) -> dict:

    path = os.path.join(VECTOR_DB_DIR, db_name, "info.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


#========== Document 儲存與載入 =============

# 儲存 metadata.json：包含所有 Document 的 page_content 與 metadata
def save_documents(docs: list[Document], db_name: str):

    serialized_docs = [{
        "page_content": doc.page_content,
        "metadata": doc.metadata
    } for doc in docs]

    path = os.path.join(VECTOR_DB_DIR, db_name, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized_docs, f, indent=2, ensure_ascii=False)

# 載入 metadata.json，轉換成 Document list
def load_documents(db_name: str) -> list[Document]:

    path = os.path.join(VECTOR_DB_DIR, db_name, "metadata.json")
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        serialized_docs = json.load(f)

    return [
        Document(
            page_content=item.get("page_content", ""),
            metadata=item.get("metadata", {})
        )
        for item in serialized_docs
    ]


#========== 向量資料 儲存與載入 =============

# 自訂 Ollama Embedding 類別
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

# 文件切割並建立 FAISS 向量庫
def build_FAISS(documents, chunk_size, chunk_overlap,  embedding_model) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(documents)
    for idx, doc in enumerate(split_docs):
        doc.metadata['chunk_id'] = idx
    return FAISS.from_documents(split_docs, embedding_model)

# 建立並儲存 FAISS 向量庫
def save_vectorstore(docs, chunk_size, chunk_overlap, db_path):
    """
    - 使用 OllamaEmbeddings 做嵌入
    - 透過 build_FAISS 處理 chunk 與嵌入
    """
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_url)
    vectorstore = build_FAISS(docs, chunk_size, chunk_overlap, embeddings)
    vectorstore.save_local(db_path)
    return len(vectorstore.docstore._dict)

# 從本地載入 FAISS 向量庫
def load_vectorstore(db_path: str, embedding_model: str) -> FAISS:

    if not os.path.exists(os.path.join(db_path, "index.faiss")) or \
       not os.path.exists(os.path.join(db_path, "index.pkl")):
        raise FileNotFoundError(f"FAISS 向量庫檔案不完整: {db_path}")

    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_url)
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)


#========== 向量資料庫建立、重建、刪除、更新  =============

# 新建一個向量資料庫
def create_new_vector_db(db_name: str, uploaded_files, img_model: str, first_time: bool, chunk_size, chunk_overlap):

    # 1. 建立資料夾與儲存檔案
    db_path = os.path.join(VECTOR_DB_DIR, db_name)
    if os.path.exists(db_path):
        raise FileExistsError(f"資料庫 {db_name} 已存在")

    os.makedirs(db_path, exist_ok=True)
    source_dir = os.path.join(db_path, "source_files")
    os.makedirs(source_dir, exist_ok=True)

    for f in uploaded_files:
        with open(os.path.join(source_dir, f.name), "wb") as out_f:
            out_f.write(f.getbuffer())

    # 2. 使用 process_uploaded_files 做前處理
    docs = process_uploaded_files(uploaded_files, img_model)

    # 3. save_vectorstore() 建立向量庫並儲存
    chunk_nums = save_vectorstore(docs, chunk_size, chunk_overlap, db_path)

    # 4. 寫入 info.json 和 metadata.json
    save_metadata({
        "last_edit": str(datetime.now()),
        "files": [f.name for f in uploaded_files],
        "img_model": img_model,
        "chunk_size": chunk_size,
        "chunk_nums": chunk_nums
    }, db_name)

    save_documents(docs, db_name)
    return chunk_nums

# 重建向量庫：在原有資料庫上加入新文件
def rebuild_vector_db(db_name: str, chunk_overlap, uploaded_files):

    # 1. 載入以前的 metadata & documents
    db_path = os.path.join(VECTOR_DB_DIR, db_name)
    source_dir = os.path.join(db_path, "source_files")
    os.makedirs(source_dir, exist_ok=True)

    meta = load_metadata(db_name)
    docs = load_documents(db_name)
    files = meta.get("files", [])

    # 2. 加入新檔案並合併至 documents
    for f in uploaded_files:
        file_path = os.path.join(source_dir, f.name)
        with open(file_path, "wb") as out_f:
            out_f.write(f.getbuffer())

        if f.name not in files:
            files.append(f.name)

        fake_upload = BytesIO(f.getbuffer())
        fake_upload.name = f.name
        docs += process_uploaded_files([fake_upload], meta["img_model"])

    # 3. 重新建立 vectorstore 與儲存
    chunk_nums = save_vectorstore(docs, meta["chunk_size"], chunk_overlap, db_path)

    # 4. 寫入 info.json 和 metadata.json
    meta["files"] = files
    meta["chunk_nums"] = chunk_nums
    meta["last_edit"] = str(datetime.now())
    save_metadata(meta, db_name)
    save_documents(docs, db_name)

    return chunk_nums

# 刪除指定檔案，並重新整理向量庫與 metadata
def delete_files_from_db(db_name: str, files_to_delete: list[str], chunk_overlap):
    
    # 1. 載入以前的 metadata
    db_path = os.path.join(VECTOR_DB_DIR, db_name)
    source_dir = os.path.join(db_path, "source_files")

    meta = load_metadata(db_name)
    existing_files = meta.get("files", [])
    remaining_files = [f for f in existing_files if f not in files_to_delete]

    # 2. 刪除檔案
    for file_name in files_to_delete:
        file_path = os.path.join(source_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

    # 3. 整理 docs，排除被刪除的檔案來源
    all_docs = load_documents(db_name)
    docs = [doc for doc in all_docs if doc.metadata.get("source") not in files_to_delete]
    
    # 4. 重新建立 vectorstore 與儲存
    chunk_nums = save_vectorstore(docs, meta["chunk_size"], chunk_overlap, db_path)

    # 5. 寫入 info.json 和 metadata.json
    meta["files"] = remaining_files
    meta["last_edit"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["chunk_nums"] = chunk_nums
    save_metadata(meta, db_name)
    save_documents(docs, db_name)

    return chunk_nums

# 完全刪除一個資料庫（整個資料夾）
def delete_vector_db(db_name: str) -> bool:

    db_path = os.path.join(VECTOR_DB_DIR, db_name)
    try:
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        return True
    except Exception as e:
        print(f"❌ 資料庫刪除失敗: {e}")
        return False
