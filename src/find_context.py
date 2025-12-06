# src/find_context.py
import os
import sys

BOOK = os.path.join("data", "book.txt")

def show_context(needle, window_chars=300):
    if not os.path.exists(BOOK):
        print("data/book.txt not found")
        return
    text = open(BOOK, "r", encoding="utf-8").read()
    idx = text.lower().find(needle.lower())
    if idx == -1:
        print("Phrase not found in book:", needle)
        return
    start = max(0, idx - window_chars)
    end = min(len(text), idx + window_chars)
    print("---- CONTEXT AROUND MATCH ----")
    print(text[start:end])
    print("---- MATCH LOCATION: chars", start, "to", end, "----")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/find_context.py \"phrase to find\"")
    else:
        phrase = sys.argv[1]
        show_context(phrase)
