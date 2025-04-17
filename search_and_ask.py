# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import openai
# import os
# from datetime import datetime

# # 設定 OpenAI API 金鑰
# openai.api_key = os.getenv("OPENAI_API_KEY")  # 確保在環境變數中設定金鑰

# # 載入向量模型
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # 讀取最新 index 檔案
# def load_latest_index(index_folder="index"):
#     index_files = sorted(
#         [f for f in os.listdir(index_folder) if f.endswith(".index")],
#         key=lambda x: os.path.getmtime(os.path.join(index_folder, x)),
#         reverse=True
#     )
#     if not index_files:
#         raise FileNotFoundError("找不到任何 index 檔案")
#     return faiss.read_index(os.path.join(index_folder, index_files[0]))

# # 查詢向量資料庫
# def search_similar_texts(query, texts, index, top_k=3):
#     query_vector = embedding_model.encode([query]).astype("float32")
#     distances, indices = index.search(query_vector, top_k)
#     return [texts[i] for i in indices[0] if i < len(texts)]

# # 與 OpenAI 進行對話
# def ask_openai(question, context):
#     prompt = f"""你是一個 PDF 文件助手，請根據下方內容回答問題。
# 內容：
# {context}

# 問題：
# {question}

# 請根據內容簡潔地回答。"""

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     return response.choices[0].message.content.strip()

# # 主流程
# def search_and_ask(question, texts_file="texts.txt", index_folder="index"):
#     index = load_latest_index(index_folder)

#     # 讀取原始文字資料
#     with open(texts_file, "r", encoding="utf-8") as f:
#         all_text = f.read()
#         all_chunks = all_text.split("\n\n")  # 假設每段是個 chunk

#     # 找出相似段落
#     top_chunks = search_similar_texts(question, all_chunks, index)

#     # 合併 context 並送給 OpenAI
#     context = "\n".join(top_chunks)
#     answer = ask_openai(question, context)
#     return answer
