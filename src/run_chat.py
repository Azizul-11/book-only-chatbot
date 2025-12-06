# src/run_chat.py
import os
from chunker import chunk_text
from retriever import Retriever
from mmr_and_answer import build_answer

BOOK_PATH = os.path.join("data", "book.txt")
INDEX_PATH = os.path.join("data", "index.joblib")

def build_system(chunk_size=500, overlap=50, rebuild=False):
    if not os.path.exists(BOOK_PATH):
        raise FileNotFoundError("data/book.txt not found")
    with open(BOOK_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    retriever = Retriever(chunks, index_path=INDEX_PATH, rebuild=rebuild)
    return {"chunks": chunks, "retriever": retriever}

def repl(system, top_k=3, top_sentences=3):
    print("Offline Book Chatbot — CLI. Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ("exit","quit"):
            print("Bye.")
            break
        hits = system["retriever"].top_k_chunks(q, top_k=top_k)
        if not hits:
            print("(no relevant content found)\n")
            continue
        retrieved = [system["chunks"][idx] for idx,_ in hits]
        ans = build_answer(q, retrieved, system["retriever"].vectorizer, top_k_sentences=top_sentences)
        print("\nAnswer:\n" + ans + "\n")
        print("Sources:")
        for idx, score in hits:
            print(f"  Chunk {idx} — Score: {score:.4f}")
        print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    sys = build_system(chunk_size=500, overlap=50, rebuild=False)
    repl(sys, top_k=3, top_sentences=3)
