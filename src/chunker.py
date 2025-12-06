# src/chunker.py
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Sliding-window chunking: chunk_size words per chunk with overlap words repeated.
    """
    words = text.split()
    if chunk_size <= overlap:
        overlap = max(1, chunk_size // 10)
    chunks = []
    n = len(words)
    i = 0
    while i < n:
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks
