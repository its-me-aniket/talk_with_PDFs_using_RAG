# 📄 QnA from PDF using RAG (Retrieval-Augmented Generation)

This project allows you to upload **multiple PDF documents** and ask **questions** from them using a **Retrieval-Augmented Generation** (RAG) pipeline.

It uses:
- **Sentence Transformers** for semantic embeddings.
- **FAISS** for fast similarity search.
- **FLAN-T5** (generative model) for context-aware, detailed answers.
- **Streamlit** for a simple, interactive web UI.

---

## 🚀 Features
- Upload **multiple PDFs** at once.
- Semantic search using **FAISS**.
- **Generative answering** (no short snippet issues).
- Local processing (no API key required).
- Cache index for faster repeated queries.
- **Clear cache** button in UI.

---

## 🛠 Installation
```bash
git clone https://github.com/its-me-aniket/talk_with_PDFs_using_RAG.git
cd talk_with_PDFs_using_RAG
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
