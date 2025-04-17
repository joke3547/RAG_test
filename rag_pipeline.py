import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 載入 .env 的 API KEY
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 設定 OpenRouter 客戶端
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

# 載入向量模型
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# 讀取所有 index 檔與對應的文本
def load_all_indexes_and_texts(index_folder="index", text_folder="texts"):
    index_text_pairs = []
    for file in os.listdir(index_folder):
        if file.endswith(".index"):
            base_name = file.replace(".index", "")
            index_path = os.path.join(index_folder, file)
            text_path = os.path.join(text_folder, f"{base_name}.txt")
            if not os.path.exists(text_path):
                continue  # 若找不到對應 text，跳過

            index = faiss.read_index(index_path)
            with open(text_path, "r", encoding="utf-8") as f:
                text_chunks = f.read().split("\n\n")

            index_text_pairs.append((index, text_chunks))
    return index_text_pairs


# 查詢向量資料庫
def search_similar_texts(query, text_chunks, index, top_k=3):
    query_vector = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(text_chunks):
            results.append((text_chunks[idx], dist))
    return results


# 問 OpenAI / OpenRouter
def ask_openai(question, context):
    prompt = f"""你是一個 PDF 文件助手，請根據下方內容回答問題。
內容：
{context}

問題：
{question}

請根據內容簡潔地回答。"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-maverick:free",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response.choices[0].message.content.strip()


# 主流程
def search_and_ask(question, index_folder="index", text_folder="texts", top_k=3):
    index_text_pairs = load_all_indexes_and_texts(index_folder, text_folder)
    all_results = []

    for index, text_chunks in index_text_pairs:
        results = search_similar_texts(question, text_chunks, index, top_k)
        all_results.extend(results)

    # 根據距離排序（距離越小越相似）
    all_results.sort(key=lambda x: x[1])
    top_chunks = [chunk for chunk, _ in all_results[:top_k]]

    context = "\n".join(top_chunks)
    return ask_openai(question, context)
