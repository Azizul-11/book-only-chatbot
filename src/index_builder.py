# src/index_builder.py
import re

def clean_for_tfidf(text: str) -> str:
    """
    Clean text for TF-IDF:
    - Lowercase
    - Remove non-alphanumeric chars (keep spaces)
    - Collapse whitespace
    """
    if text is None:
        return ""
    text = text.lower()
    # replace non-alphanumeric characters with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
