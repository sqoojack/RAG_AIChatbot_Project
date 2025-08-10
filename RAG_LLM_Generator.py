
# RAG_LLM_Generator.py
from utils import List, re
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# 呼叫 LLM 根據 prompt 生成回應 
def llm_generator(
    query: str,
    chunks_with_scores: List,  # [(Document, score), ...] 結構
    ollama_url: str,
    llm_model: str = "deepseek-r1:8b",
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    使用 LLM (預設為deepseek-r1) 根據檢索到的內容與問題, 生成繁體中文的回應。

    參數:
    - query: 使用者的問題
    - chunks_with_scores: 相似度搜尋得到的 [(文件區塊, 分數)] 清單
    - ollama_url: 本地或遠端 Ollama API 端點
    - llm_model: 使用的 LLM 模型名稱 (預設 deepseek-r1:8b)
    - temperature: 控制輸出多樣性（愈高愈隨機）
    - top_p: nucleus sampling 的參數（通常搭配 temperature 使用）

    回傳:
    - 回答文字 (str)
    """

    # 建立 prompt 模板，支援 context 與 question 的格式化插入
    prompt_template = PromptTemplate.from_template(
        """你是一個AI的助理, 請使用繁體中文回答使用者的問題。
        以下是檢索資料，如果沒有可以回答問題資料，請表示"檢索無相關資料，我不知道"。
        {context}

        問題：{question}

        如果有衝突的資料，請告訴我資料來源，以繁體中文作答："""
    )

    # 將相似文件格式化為 context 字串，加入段落索引與來源
    context = "\n\n".join([
        f"段落{i+1}，資料來源: {doc.metadata.get('source')} || 內容: {doc.page_content.strip()}"
        for i, (doc, _) in enumerate(chunks_with_scores)
    ])

    # 初始化 LLM（如 deepseek, llama3 等）並建立推論鏈
    llm = OllamaLLM(
    model=llm_model,
    base_url=ollama_url,  # 這裡放遠端 Ollama 服務的網址
    temperature=temperature,
    top_p= top_p
    )
    # 建立推論鏈（Runnable）
    chain = prompt_template | llm

    # 執行推論，傳入 query 與 context到LLM裡
    result = chain.invoke({
        "context": context,
        "question": query
    })

    return result

# 將思考過程 (<think>標記)與最終答案 分開來
def extract_answer_and_thought(text: str):
    match = re.search(r"<think>([\s\S]*?)</think>", text)
    if match:
        thought = match.group(1).strip()
        answer = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    else:
        answer = text.strip()
        thought = ""
    return answer, thought