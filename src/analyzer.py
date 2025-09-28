# src/analyzer.py
from typing import Dict, List, Tuple, Set
import re
import math
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer

from .skills_catalog import ALL_SKILLS_CANONICAL, canonicalize
from .nlp_utils import top_nouns

WORD_RE = re.compile(r"[A-Za-z][A-Za-z+\-/]{1,}")

def _tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())

def _find_skills(text: str) -> Set[str]:
    tokens = " ".join(_tokenize(text))
    found = set()
    for sk in ALL_SKILLS_CANONICAL:
        # word boundary search for multi-words too
        pattern = r"(?<![A-Za-z])" + re.escape(sk) + r"(?![A-Za-z])"
        if re.search(pattern, tokens):
            found.add(sk)
    # fuzzy pass for common variants (edit distance/ token swap, etc.)
    # Use a small list to avoid false positives
    candidates, scores = [], []
    for sk in ALL_SKILLS_CANONICAL:
        match, score, _ = process.extractOne(sk, [tokens], scorer=fuzz.partial_ratio)
        if score >= 90:
            candidates.append(sk)
            scores.append(score)
    found.update(candidates)
    return found

def _tfidf_similarity(a: str, b: str) -> float:
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1, max_features=5000)
    X = vect.fit_transform([a, b])
    v1 = X[0].toarray()[0]
    v2 = X[1].toarray()[0]
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

def _missing_keywords(resume_text: str, jd_text: str, exclude: Set[str], top_k: int = 20) -> List[str]:
    # derive salient tokens from JD not present in resume, excluding skills
    jd_tokens = [t for t in _tokenize(jd_text) if t not in exclude]
    res_tokens = set(_tokenize(resume_text))
    counts = Counter(jd_tokens)
    missing = [w for w, _ in counts.most_common() if w not in res_tokens]
    # filter very short/ generic
    missing = [m for m in missing if len(m) >= 4]
    # de-duplicate nouns preference
    nouns = top_nouns(jd_text, n=200)
    noun_set = set(nouns)
    ordered = [m for m in missing if m in noun_set] + [m for m in missing if m not in noun_set]
    # keep unique order
    seen, out = set(), []
    for m in ordered:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out[:top_k]

def analyze_resume_vs_jd(resume_text: str, jd_text: str) -> Dict:
    resume_skills = _find_skills(resume_text)
    jd_skills = _find_skills(jd_text)

    sim = _tfidf_similarity(resume_text, jd_text)
    coverage = 0.0
    if jd_skills:
        coverage = len(resume_skills & jd_skills) / len(jd_skills)

    overall = 0.6 * sim + 0.4 * coverage
    overall_score = round(overall * 100, 1)

    # missing skills prioritized by jd order
    jd_order = [s for s in _tokenize(jd_text) if canonicalize(s) in jd_skills]
    # Keep unique order
    seen = set()
    missing_skills = []
    for s in jd_order:
        c = canonicalize(s)
        if c in jd_skills and c not in resume_skills and c not in seen:
            missing_skills.append(c)
            seen.add(c)

    # Missing keywords (beyond skills)
    exclude = set(list(jd_skills) + list(resume_skills))
    missing_kw = _missing_keywords(resume_text, jd_text, exclude=exclude, top_k=20)

    # Suggestions (templated + data-driven)
    suggestions = []
    for s in missing_skills[:8]:
        suggestions.append(f"Add evidence of **{s}** (e.g., a project or bullet that used {s} with measurable impact).")
    if sim < 0.5:
        suggestions.append("Align terminology with the job description — mirror key phrases where truthful (avoid keyword stuffing).")
    if coverage < 0.5:
        suggestions.append("Highlight relevant projects near the top and group bullets under skill headers to increase visibility.")
    if missing_kw:
        suggestions.append(f"Consider incorporating domain keywords like: {', '.join(missing_kw[:8])}.")

    result = {
        "resume_text": resume_text,
        "jd_text": jd_text,
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "missing_skills": missing_skills,
        "missing_keywords": missing_kw,
        "similarity_score": sim,
        "skills_coverage": coverage,
        "overall_score": overall_score,
        "suggestions": suggestions,
    }
    return result

def build_report_markdown(result: Dict, top_n_suggestions: int = 8) -> str:
    lines = []
    lines.append("# Resume ↔ Job Description Analysis Report")
    lines.append("")
    lines.append(f"**Overall Match Score:** {result['overall_score']:.1f}/100")
    lines.append(f"- TF‑IDF Similarity: {result['similarity_score']*100:.1f}/100")
    lines.append(f"- Skills Coverage: {result['skills_coverage']*100:.1f}/100")
    lines.append("")
    lines.append("## Skills")
    lines.append(f"**Resume skills:** {', '.join(result['resume_skills']) or '_None_'}")
    lines.append(f"**JD skills:** {', '.join(result['jd_skills']) or '_None_'}")
    lines.append(f"**Missing skills:** {', '.join(result['missing_skills']) or '_None_'}")
    lines.append("")
    lines.append("## Keyword Gaps")
    if result["missing_keywords"]:
        lines.append(", ".join(result["missing_keywords"]))
    else:
        lines.append("_None_")
    lines.append("")
    lines.append("## Suggestions")
    for tip in result["suggestions"][:top_n_suggestions]:
        lines.append(f"- {tip}")
    lines.append("")
    lines.append("_Generated by AI Resume Analyzer (Python + Streamlit)._")
    return "\n".join(lines)
