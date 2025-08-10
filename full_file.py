# full_file.py
from langchain.schema import Document
from RAG_LLM_Generator import llm_generator

""" 支援不只取得第一高chunk的源頭檔(k個) """
def generate_full_files_answer(top_chunks, all_docs, query, ollama_url, model_settings):
    sources = []
    result_same_docs = []  # 用來儲存原始的 same_docs
    combined_texts = []  # 用來儲存每個文檔的內容
    source_info = []  # 用來儲存每個來源的資訊
    
    # 找出前 k 個最相關的檔案的 source
    for doc, _ in top_chunks:
        src = doc.metadata.get("source")    # 找到最高相關度的 chunk 所在的檔案名
        if src not in sources:
            sources.append(src)
    
    # 合併來源檔案的內容
    for src in sources:
        same_docs = [d for d in all_docs if d.metadata.get("source") == src]
        same_docs.sort(key=lambda d: d.metadata.get("page", 0))  # 根據頁碼排序
        
        # 儲存原始的 same_docs 供後續使用
        result_same_docs.extend(same_docs)
        
        # 合併文檔內容並標記來源
        for doc in same_docs:
            combined_texts.append(doc.page_content)
            source_info.append(doc.metadata.get("source"))
    
    # 將所有文檔內容合併為一個長文本
    full_text = "\n\n".join(combined_texts)
    
    # 將來源的 source 信息合併為一個字串
    metadata = {"source": ", ".join(source_info)}
    
    # 使用合併後的文檔生成答案
    answer = llm_generator(
        query,
        [(Document(page_content=full_text, metadata=metadata), 1.0)],  # 合併文檔
        ollama_url,
        llm_model=model_settings["llm_model"],
        temperature=model_settings["temperature"],
        top_p=model_settings["top_p"]
    )
    return result_same_docs, answer  # 只返回原始的 same_docs 和生成的答案