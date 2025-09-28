# src/nlp_utils.py
from typing import List, Tuple, Optional
import re

def safe_load_spacy():
    """Try to load spaCy en_core_web_sm. Return nlp or None if not available."""
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            # Model not present
            return None
    except Exception:
        return None

def top_nouns(text: str, n: int = 20) -> list:
    """Return up to n lemmatized nouns from text using spaCy if available; otherwise a regex fallback."""
    nlp = safe_load_spacy()
    if nlp:
        doc = nlp(text)
        nouns = [t.lemma_.lower() for t in doc if t.pos_ in {"NOUN", "PROPN"} and t.is_alpha]
        # dedupe preserving order
        seen, out = set(), []
        for w in nouns:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out[:n]
    # Fallback: crude keyword split
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:n]
