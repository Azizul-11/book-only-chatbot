# 📕 Book-Only Chatbot (Offline RAG Demo)

This project is a **book-constrained question-answering system** built without any external APIs.

It answers questions strictly from a provided book using:
- Text chunking
- TF-IDF retrieval
- Extractive answer selection (MMR-style)

## ✅ Key Features
- No OpenAI / no external APIs
- Answers strictly grounded in book content
- Supports definition, explanation, examples, and summaries
- Shows source chunks for transparency
- Built for clarity and explainability

## 🧠 How It Works
1. Book text is split into overlapping chunks
2. TF-IDF is used to retrieve the most relevant chunks
3. Key sentences are extracted and ranked
4. Answers are composed only from book text

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run src/app.py
