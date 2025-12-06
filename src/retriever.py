# src/retriever.py  (FULLY FIXED — AUTO REBUILD SAFE VERSION)

import os
import hashlib
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def hash_chunks(chunks):
    """Hash all chunks to detect dataset changes."""
    joined = "\n".join(chunks).encode("utf-8")
    return hashlib.md5(joined).hexdigest()

class Retriever:
    """
    TF-IDF retriever that automatically rebuilds the index when:
    - The book text changes
    - Chunking parameters change
    - index.joblib is missing or outdated
    """

    def __init__(self, chunks, index_path="data/index.joblib",
                 max_features=50000, ngram_range=(1,2), rebuild=False):

        self.chunks = chunks
        self.index_path = index_path

        # Compute a hash for the chunk dataset
        self.hash = hash_chunks(chunks)

        # If index exists AND rebuild=False, try loading it
        if os.path.exists(index_path) and not rebuild:
            try:
                data = joblib.load(index_path)
                if data.get("hash") == self.hash:
                    # safe to reuse
                    self.vectorizer = data["vectorizer"]
                    self.tfidf_matrix = data["tfidf"]
                    return
            except:
                pass  # fall through to rebuild

        # Otherwise rebuild EVERYTHING
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)

        # Save to disk
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "tfidf": self.tfidf_matrix,
            "hash": self.hash
        }, index_path, compress=3)

    def query_scores(self, question: str):
        q_vec = self.vectorizer.transform([question])
        sims = linear_kernel(q_vec, self.tfidf_matrix)[0]
        return sims

    def top_k_chunks(self, question: str, top_k=3):
        sims = self.query_scores(question)
        top_k = min(top_k, len(sims))
        idxs = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idxs]
