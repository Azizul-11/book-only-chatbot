# src/mmr_and_answer.py
from typing import List, Tuple
import re
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

_sent_end_re = re.compile(r'(?<=[\.\?\!])\s+|\n+')
_bracket_footnote_re = re.compile(r'[\[\(]\s*\d{1,3}\s*[\]\)]')
_short_bracket_re = re.compile(r'\[[^\]]{1,60}\]')
_multi_space_re = re.compile(r'\s+')
_trailing_brackets_re = re.compile(r'[\]\)\"\'\s]+$')
_leading_brackets_re = re.compile(r'^[\[\(\"\'\s]+')
_trailing_listnum_re = re.compile(r'(\s*\d+\.\s*$)')

# keywords to boost relevance for questions about weakness/vulnerability
KEYWORD_BOOST = ["weak", "vulnerable", "vulnerability", "appear", "appear", "strike", "attack", "bolt", "thunderbolt", "shun", "defend"]

def simple_sent_tokenize(text: str) -> List[str]:
    if not text:
        return []
    parts = _sent_end_re.split(text)
    sents = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        s = _bracket_footnote_re.sub("", s)
        s = _short_bracket_re.sub("", s)
        s = _multi_space_re.sub(" ", s)
        s = _leading_brackets_re.sub("", s)
        s = _trailing_brackets_re.sub("", s)
        s = s.strip(" ,;:")
        s = _trailing_listnum_re.sub("", s)
        if len(s) < 12:
            continue
        sents.append(s)
    return sents

def mmr_rerank(question: str, candidate_sents: List[str], vectorizer, lambda_param: float = 0.6, top_n: int = 3, keyword_boost: float = 0.35) -> List[Tuple[str,float]]:
    """
    Returns list of (sentence, score) selected by MMR with an additional keyword boost
    for sentences that contain any of KEYWORD_BOOST tokens.
    """
    if not candidate_sents:
        return []
    docs = [question] + candidate_sents
    vecs = vectorizer.transform(docs)
    qv = vecs[0]
    svecs = vecs[1:]
    sim_q = linear_kernel(qv, svecs)[0].astype(float)
    sim_docs = linear_kernel(svecs, svecs).astype(float)

    # apply keyword boost
    boost = np.zeros_like(sim_q)
    for i, s in enumerate(candidate_sents):
        sl = s.lower()
        for kw in KEYWORD_BOOST:
            if kw in sl:
                boost[i] += keyword_boost

    # incorporate boost into relevance before MMR
    sim_q_boosted = sim_q + boost

    selected_idxs = []
    candidates = list(range(len(candidate_sents)))
    while len(selected_idxs) < min(top_n, len(candidate_sents)):
        mmr_scores = []
        for idx in candidates:
            relevance = float(sim_q_boosted[idx])
            redundancy = 0.0 if not selected_idxs else float(max(sim_docs[idx][j] for j in selected_idxs))
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append((mmr_score, idx))
        mmr_scores.sort(reverse=True)
        chosen = mmr_scores[0][1]
        selected_idxs.append(chosen)
        candidates.remove(chosen)
    return [(candidate_sents[i], float(sim_q[i])) for i in selected_idxs]

def dedupe_and_order(sents_with_meta: List[Tuple[str,int,int]]) -> List[str]:
    out = []
    seen = set()
    for s, cidx, sidx in sorted(sents_with_meta, key=lambda x: (x[1], x[2])):
        key = re.sub(r'\s+',' ', s.strip().lower()).strip(" .,:;\"'")
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

def clean_and_capitalize(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    s = re.sub(r'[\[\]\"\'`]+', '', s)
    s = re.sub(r'\s+', ' ', s).strip(" ,;:")
    if not s:
        return s
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if s[-1] not in ".?!":
        s = s + "."
    return s

def build_answer(question: str, top_chunk_texts: List[str], vectorizer, top_k_sentences: int = 3) -> Tuple[str,str]:
    """
    Returns (short_extractive_summary, top_sentence_for_provenance)
    - Picks candidate sentences from top chunks
    - Uses MMR with keyword boosting to select best sentences
    - Orders them by chunk+position and returns a concise 1-3 sentence extractive summary
    """
    candidates = []
    for cidx, chunk in enumerate(top_chunk_texts):
        sents = simple_sent_tokenize(chunk)
        for si, s in enumerate(sents):
            candidates.append((s, cidx, si))
    if not candidates:
        return "(no relevant content found in the book)", ""

    s_texts = [t for t,_,_ in candidates]

    # select sentences with MMR + keyword boost
    selected = mmr_rerank(question, s_texts, vectorizer, top_n=min(len(s_texts), top_k_sentences))

    # mmr_rerank returned list of (sent, score_by_q). Map to metadata
    s_with_meta = []
    text_to_meta = {}
    for t,c,sidx in candidates:
        if t not in text_to_meta:
            text_to_meta[t] = (c,sidx)
    for s,score in selected:
        meta = text_to_meta.get(s,(999,999))
        s_with_meta.append((s, meta[0], meta[1], score))

    # dedupe & order by original position
    final = dedupe_and_order([(s,c,si) for s,c,si,sc in s_with_meta])

    # polish and join into short summary (limit to top_k_sentences)
    polished = [clean_and_capitalize(s) for s in final[:top_k_sentences]]
    polished = [p for p in polished if p]
    short_summary = " ".join(polished)

    # top sentence provenance: prefer first polished entry
    top_sentence = polished[0] if polished else ""

    return short_summary, top_sentence
