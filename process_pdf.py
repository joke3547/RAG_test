# 若要完善圖片識讀(識讀出圖片中文字), 需要安裝Tesseract OCR, 並設定系統環境變數
import fitz  # PyMuPDF
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import torch  # BLIP 圖片描述模型需要用到 PyTorch
import numpy as np
import faiss
from PIL import Image
import io
import os
from datetime import datetime

# 1. 提取純文字
def extract_text_from_pdf(pdf_path):
    text = ''
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

# 2. 僅使用 BLIP 圖片描述
def extract_image_descriptions_from_pdf(pdf_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    doc = fitz.open(pdf_path)
    descriptions = []

    for page in doc:
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # 使用 BLIP 生成圖片描述
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            description = processor.decode(out[0], skip_special_tokens=True)

            combined = f"Image Description: {description}"
            descriptions.append(combined.strip())

    return descriptions

# 3. Chunk text
def split_text_into_chunks(text, chunk_size=300, overlap=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

# 4. 向量化
def vectorize_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(chunks)

# 5. 整合流程
def process_pdf(pdf_path):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{filename}_{timestamp}"

    os.makedirs("index", exist_ok=True)
    os.makedirs("texts", exist_ok=True)

    text = extract_text_from_pdf(pdf_path)
    descriptions = extract_image_descriptions_from_pdf(pdf_path)
    full_text = text + "\n\n" + "\n\n".join(descriptions)

    chunks = split_text_into_chunks(full_text)
    vectors = vectorize_chunks(chunks)

    # 儲存 index
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors, dtype=np.float32))
    index_path = os.path.join("index", f"{base_name}.index")
    faiss.write_index(index, index_path)

    # 儲存 chunks 文字
    text_path = os.path.join("texts", f"{base_name}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n\n")

    return base_name, len(chunks)
