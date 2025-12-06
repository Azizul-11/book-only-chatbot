# src/app.py

import streamlit as st
import os
from chunker import chunk_text
from retriever import Retriever
from mmr_and_answer import build_answer

BOOK_PATH = os.path.join("data", "book.txt")
INDEX_PATH = os.path.join("data", "index.joblib")

st.set_page_config(page_title="Book Chatbot — Demo", layout="wide")
st.title("📕 Book-only Chatbot — Offline demo (highlighted source)")

# ❗ NO CACHE — NEVER USE CACHE IN A RAG PIPELINE
def build_system(chunk_size: int = 500, overlap: int = 50, rebuild_index: bool = False):
    if not os.path.exists(BOOK_PATH):
        st.error("Book file not found in data/book.txt")
        return None

    # Load book
    with open(BOOK_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # Build chunks consistently
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    # Rebuild index or embeddings every run → stable system
    retriever = Retriever(
        chunks=chunks,
        index_path=INDEX_PATH,
        rebuild=rebuild_index  # user can force rebuild
    )

    return {"chunks": chunks, "retriever": retriever}


# -------------------- SIDEBAR CONTROLS --------------------
with st.sidebar:
    st.header("Settings")
    chunk_size = st.number_input("Chunk size (words)", min_value=200, max_value=2000, value=500, step=100)
    overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=500, value=50, step=10)
    top_k = st.number_input("Top chunks to fetch", min_value=1, max_value=10, value=3, step=1)
    top_sent = st.number_input("Answer sentences", min_value=1, max_value=6, value=3, step=1)
    rebuild = st.button("Force rebuild index")


# Build system fresh EVERY run → prevents ALL index mismatch errors
system = build_system(chunk_size=int(chunk_size), overlap=int(overlap), rebuild_index=rebuild)

if system is None:
    st.stop()

# Conversation history
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("Ask any question about the book. Answers are extracted directly from the text and show sources.")

query = st.text_input("Your question:")

# -------------------- MAIN LOGIC --------------------
if st.button("Ask"):
    if not query.strip():
        st.warning("Type a question.")
    else:
        retriever = system["retriever"]
        chunks = system["chunks"]

        # Get top chunks
        top = retriever.top_k_chunks(query, top_k=int(top_k))

        if not top:
            st.info("No relevant content found.")
        else:
            # Safely extract chunks WITHOUT index errors
            valid_chunks = []
            for idx, score in top:
                if 0 <= idx < len(chunks):
                    valid_chunks.append(chunks[idx])
                else:
                    # If any stale index appears, skip it (should never happen anymore)
                    continue

            polished_answer, top_sentence = build_answer(
                query,
                valid_chunks,
                retriever.vectorizer,
                top_k_sentences=int(top_sent)
            )

            st.session_state.history.append((query, polished_answer))

            st.subheader("Answer (polished, extractive)")
            st.write(polished_answer)

            if top_sentence:
                st.markdown("**Top contributing sentence (source):**")
                st.info(top_sentence)

            st.subheader("Sources (top chunks)")
            for idx, score in top:
                if 0 <= idx < len(chunks):
                    excerpt = chunks[idx][:600]
                    st.write(f"Chunk {idx} — score {score:.4f}")
                    st.write(excerpt + ("..." if len(chunks[idx]) > 600 else ""))


# -------------------- HISTORY --------------------
st.markdown("---")
st.header("Recent conversation")

for q, a in reversed(st.session_state.history[-10:]):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
    st.write("---")
