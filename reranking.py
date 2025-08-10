#reranking.py
from utils import requests, List, CrossEncoder

def rerank_with_siliconflow(query, passages: List[str], top_k, reranking_url, API_key, cert_datapath) -> List[tuple]:

    # reranking_url = reranking_url    # api_url在cherry studio的siliconflow中可找到, 因為ollama目前沒辦法使用reranking model
    headers = {
        "Content-Type": "application/json",     # 用來告訴伺服器傳的data是json類別
        "Authorization": f"Bearer {API_key}"   # <---- 記得換成自己的API
    }
    data = {
        "model": "BAAI/bge-reranker-v2-m3",  # reranking model name
        "query": query,
        "documents": passages                    # 欄位改為 documents
    }
    response = requests.post(reranking_url, headers=headers, json=data, verify=cert_datapath)     # 這邊要建立公司的憑證, 不然會SLL Error
    result = response.json()
    # 根據 API 實際回傳格式解析
    scores = [r['relevance_score'] for r in result["results"]]  
    #print(scores)
    reranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)  # 回傳 chunk內容 以及reranking幫他評估的分數
    return reranked[:top_k]     # 只回傳top_k筆

def rerank_with_HuggingFace(query, passages: List[str], top_k: int):
    model_name = "BAAI/bge-reranker-v2-m3" 
    reranker = CrossEncoder(model_name)     # 從載入reranker model

    pairs = [[query, p] for p in passages]  # 組成 (query, passage) 對列表

    
    scores = reranker.predict(pairs)  # 3. 預測相關度分數（分數越高代表越相關）

    # 4. 依分數由大到小排序並取前 top_k
    ranked = sorted(zip(passages, scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# 進行語意相似搜尋，找出與 query (user輸入的prompt) 最相近的 k個段落與分數,  Reranking情況: 先找出 top n (ex: 20~50)候選段落
def search_top_k(query, vectorstore, top_k):
    chunks_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    return chunks_with_scores

def rerank_chunks_top_k(query, chunks_with_scores, top_k, reranking_url, reranking_api, cert_datapath):
    docs = [doc for doc, _ in chunks_with_scores]   # 取得所有文檔
    contents = [doc.page_content for doc in docs]   # 取得所有檔案內容

    # reranked = rerank_with_siliconflow(query, contents, top_k=top_k, reranking_url=reranking_url, API_key=reranking_api, cert_datapath=cert_datapath)
    reranked = rerank_with_HuggingFace(query, contents, top_k)
    
    passage2doc = {doc.page_content: doc for doc in docs}     # 建立一個字典，將每個文檔的內容與其對應的 Document 物件進行映射
    reranked_docs = [(passage2doc[passage], score) for passage, score in reranked]  # 將文字和分數轉回Document型式
    return reranked_docs