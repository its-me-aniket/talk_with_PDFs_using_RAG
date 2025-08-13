import streamlit as st
import os
from utils import (
    read_pdf, chunk_text, compute_embeddings, save_index, load_index,
    hash_file, search_index, answer_with_llm, clear_cache
)
import faiss

st.set_page_config(page_title="PDF Q&A with RAG", layout="wide")

st.title("📄 PDF Q&A with RAG and AI Answer Refinement")

st.sidebar.header("⚙ Options")
use_llm = st.sidebar.checkbox("Use AI Answer Refinement", value=True)
if st.sidebar.button("Clear Cache"):
    clear_cache()
    st.sidebar.success("Cache cleared!")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        file_hash = hash_file(file_path)
        index, chunks, _ = load_index(file_hash)

        if index is None:
            text = read_pdf(file_path)
            chunks = chunk_text(text)
            embeddings = compute_embeddings(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            save_index(index, chunks, file_hash)
        
        all_chunks.extend(chunks)

    query = st.text_input("Ask a question from the uploaded PDFs:")
    if query:
        if use_llm:
            retrieved_chunks = search_index(index, all_chunks, query)
            final_answer = answer_with_llm(query, retrieved_chunks)
            st.subheader("📜 Answer")
            st.write(final_answer)
        else:
            retrieved_chunks = search_index(index, all_chunks, query)
            st.subheader("📜 Retrieved Chunks")
            for chunk in retrieved_chunks:
                st.write(chunk)
