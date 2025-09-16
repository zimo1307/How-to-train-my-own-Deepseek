import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: Load FAISS index and text chunks
index = faiss.read_index("Hogwarts_index.faiss")
chunks = np.load("Hogwarts_chunks.npy", allow_pickle=True)

# Step 2: Load embedding model (MiniLM for English)
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Step 3: Hardcoded query
query = "Who is the headmaster of Hogwarts?"

# Step 4: Encode query and perform FAISS search
query_embedding = embed_model.encode([query], normalize_embeddings=True)
top_k = 5
scores, indices = index.search(np.array(query_embedding), top_k)

# Step 5: Load DeepSeek LLM
llm_model_path = "deepseek_7b_base/deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(llm_model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    llm_model_path,
    torch_dtype=torch.float32,
    local_files_only=True
).to(device)

# Step 6: Prepare prompt
context = "\n".join([chunks[i] for i in indices[0]])
prompt = f"Here is some reference information:\n\n{context}\n\nQuestion: {query}\nAnswer:"

# Step 7: Generate answer
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
)

# Step 8: Display result
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🧙‍♂️ Generated Answer:\n", answer.split("Answer:")[-1].strip())
