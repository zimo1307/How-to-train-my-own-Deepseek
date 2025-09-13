# build_embedding_index.py

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

import torch

# ✅ Step 1: 加载文本
with open("Hogwarts.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# ✅ Step 2: 文本切分
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_text(raw_text)
print(f"🧩 文本被切分为 {len(chunks)} 个段落")

# ✅ Step 3: 加载 embedding 模型（GPU/CPU 自适应）
device = "cuda" if torch.cuda.is_available() else "cpu"
# model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
model = SentenceTransformer("/scratch/network/xz4883/all-MiniLM-L6-v2", device=device, local_files_only=True)
print(f"🚀 使用设备：{device}")

# ✅ Step 4: 计算 embedding
# embeddings = model.encode(chunks, show_progress_bar=True)
# ✅ Step 4: 计算 embedding（带 normalize）
embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)

# ✅ Step 5: 构建 FAISS 索引
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
print("✅ 向量索引构建完成")

# ✅ Step 6: 保存文件
faiss.write_index(index, "Hogwarts_index.faiss")
with open("Hogwarts_chunks.npy", "wb") as f:
    np.save(f, chunks)

print("📦 已保存 Hogwarts_index.faiss 与 Hogwarts_chunks.npy")
