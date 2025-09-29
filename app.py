"""
AI Resume Analyzer (no external spaCy model)

Why this version?
- Streamlit Cloud was erroring on `en_core_web_sm`. To avoid any model installs,
  I switched to a lightweight pipeline: regex tokenization + scikit-learn stopwords.
- Keeps all features: TF-IDF cosine similarity, missing keywords, suggestions,
  downloadable PDF report — with lots of comments so it reads like my own work.

TODOs:
- Add optional extra metrics (Jaccard overlap, phrase match).
- Add batch mode (analyze multiple resumes).
"""

import streamlit as st
import pandas as pd
import re
from io import BytesIO

# scikit-learn: vectorizer + cosine + built-in English stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# file parsing libs that match my requirements.txt
from pypdf import PdfReader           # (pypdf, not PyPDF2)
import docx                           # python-docx
from fpdf import FPDF                 # for the PDF report

# -------------------- Streamlit setup --------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📝", layout="wide")
st.title("📝 AI Resume Analyzer")
st.caption("Upload a resume and a job description to get a match score, missing keywords, suggestions, and a downloadable report.")

STOPWORDS = set(ENGLISH_STOP_WORDS)   # good enough for this project

# -------------------- Helpers --------------------
def extract_text(uploaded_file) -> str:
    """
    Extract text from PDF / DOCX / TXT.
    Note: using pypdf here to match requirements and avoid module mismatch.
    """
    name = (uploaded_file.name or "").lower()
    try:
        if name.endswith(".pdf"):
            # pypdf sometimes needs raw bytes; wrap in BytesIO to be safe.
            raw = uploaded_file.read()
            reader = PdfReader(BytesIO(raw))
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            return "\n".join(pages)
        elif name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return ""

def clean_text(text: str) -> str:
    """
    Really simple text cleaner:
    - lowercase
    - keep alphabetic tokens
    - drop English stopwords
    I used to lemmatize with spaCy's small model, but Cloud installs were flaky.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    tokens = [w for w in words if len(w) > 2 and w not in STOPWORDS]
    return " ".join(tokens)

def compute_similarity(resume_clean: str, jd_clean: str) -> float:
    """
    TF-IDF cosine similarity; bigram range helps catch short phrases like "data analysis".
    """
    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words=None)
    tfidf = vect.fit_transform([resume_clean, jd_clean])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return float(score)

def top_keywords(text: str, k: int = 40) -> list[str]:
    """Frequency-based keywords from already-cleaned text (fast + dependency-light)."""
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return []
    s = pd.Series(words)
    return s.value_counts().head(k).index.tolist()

def find_missing_keywords(resume_clean: str, jd_clean: str, limit: int = 20) -> list[str]:
    """Which JD keywords aren’t showing up in the resume."""
    jd_kw = set(top_keywords(jd_clean, 60))
    res_kw = set(top_keywords(resume_clean, 60))
    missing = [w for w in jd_kw - res_kw if len(w) > 2]
    return sorted(missing)[:limit]

def generate_pdf_report(score: float, missing: list[str], suggestions: list[str]) -> bytes:
    """Tiny PDF report via FPDF — intentionally simple for an intern project."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Resume Analyzer Report", ln=True, align="C")

    pdf.ln(4)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Match Score: {score:.2%}", ln=True)

    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Missing Keywords:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6, ", ".join(missing) if missing else "None 🎉")

    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Suggestions:", ln=True)
    pdf.set_font("Arial", "", 12)
    if suggestions:
        for s in suggestions:
            pdf.multi_cell(0, 6, f"- {s}")
    else:
        pdf.multi_cell(0, 6, "Looks solid!")

    return pdf.output(dest="S").encode("latin1")

# -------------------- UI --------------------
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
with col2:
    jd_text = st.text_area("Paste Job Description")

if resume_file and jd_text:
    resume_raw = extract_text(resume_file)
    if not resume_raw.strip():
        st.error("Could not read any text from the resume file.")
        st.stop()

    # Clean both sides
    resume_clean = clean_text(resume_raw)
    jd_clean     = clean_text(jd_text)

    # Score
    score = compute_similarity(resume_clean, jd_clean)
    st.subheader("📊 Match Score")
    st.metric(label="Resume vs Job Description", value=f"{score:.2%}")

    # Missing keywords
    missing = find_missing_keywords(resume_clean, jd_clean)
    st.subheader("❌ Missing Keywords")
    st.write(", ".join(missing) if missing else "No major gaps found 🎉")

    # Suggestions (kept short + practical)
    suggestions = []
    if score < 0.6:
        suggestions.append("Add more role-specific keywords from the job description.")
    if missing:
        suggestions.append("Work missing skills naturally into bullets (not a keyword dump).")
    if len(resume_clean.split()) < 120:
        suggestions.append("Your resume seems short — expand impact bullets with metrics.")
    if not suggestions:
        suggestions.append("Your resume already aligns well with this JD!")

    st.subheader("💡 Suggestions")
    for s in suggestions:
        st.write(f"- {s}")

    # PDF report
    pdf_bytes = generate_pdf_report(score, missing, suggestions)
    st.download_button("⬇️ Download Report (PDF)", pdf_bytes, "resume_analysis.pdf", "application/pdf")
else:
    st.info("👉 Upload a resume and paste a job description to analyze.")

