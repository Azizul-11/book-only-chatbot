# src/utils_cleaner.py
import re

def clean_ocr_artifacts(text: str) -> str:
    # remove common "Page X" lines and repeated header/footer patterns
    text = re.sub(r'page\s*\d+','', text, flags=re.IGNORECASE)
    # remove multiple short lines that look like headers
    text = re.sub(r'\n{2,}', '\n\n', text)
    # fix weird spacing
    text = re.sub(r'\s+\n', '\n', text)
    return text

if __name__ == "__main__":
    import sys, os
    p = os.path.join("data", "book.txt")
    if not os.path.exists(p):
        print("data/book.txt not found")
        sys.exit(1)
    s = open(p, "r", encoding="utf-8").read()
    s2 = clean_ocr_artifacts(s)
    open(p, "w", encoding="utf-8").write(s2)
    print("Cleaned data/book.txt")
