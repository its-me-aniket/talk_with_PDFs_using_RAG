import os
import hashlib
import faiss
import pickle
from typing import List, Tuple, Optional
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load summarization model (generative)
gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")

INDEX_DIR = "indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

def read_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def compute_embeddings(chunks: List[str]):
    return embedding_model.encode(chunks, convert_to_numpy=True)

def save_index(index: faiss.Index, chunks: List[str], file_hash: str, meta: Optional[dict] = None):
    faiss.write_index(index, os.path.join(INDEX_DIR, f"{file_hash}.index"))
    with open(os.path.join(INDEX_DIR, f"{file_hash}_chunks.pkl"), "wb") as f:
        pickle.dump({"chunks": chunks, "meta": meta}, f)

def load_index(file_hash: str) -> Tuple[Optional[faiss.Index], Optional[List[str]], Optional[dict]]:
    index_path = os.path.join(INDEX_DIR, f"{file_hash}.index")
    chunks_path = os.path.join(INDEX_DIR, f"{file_hash}_chunks.pkl")
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None, None, None
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data.get("meta")

def hash_file(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def search_index(index: faiss.Index, chunks: List[str], query: str, top_k: int = 5) -> List[str]:
    query_emb = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)
    return [chunks[i] for i in indices[0] if i != -1]

def answer_with_llm(query: str, retrieved_chunks: List[str]) -> str:
    context = " ".join(retrieved_chunks)[:3000]  # avoid long inputs
    prompt = f"Answer the following question based on the given context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer in detail:"
    try:
        result = gen_pipeline(prompt, max_length=300, do_sample=False)
        return result[0]["generated_text"]
    except Exception as e:
        return f"Error generating answer: {e}"

def clear_cache():
    for file in os.listdir(INDEX_DIR):
        os.remove(os.path.join(INDEX_DIR, file))
