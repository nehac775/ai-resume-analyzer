"""
AI Resume Analyzer

Why I built this
----------------
I wanted a practical project for internships: a tool that compares resumes 
against job descriptions and shows how well they match. This way, I could 
practice NLP with spaCy + scikit-learn, while also building a Streamlit app.

Challenges
----------
- Parsing PDFs/Word docs was messy at first → solved with pypdf + python-docx.
- Matching skills required a curated keyword list + fuzzy matching.
- TF-IDF vectors sometimes ignored domain-specific words → added stopword tuning.
- TODO: Add more metrics like Jaccard similarity and named-entity overlap.
- TODO: Allow multiple resumes at once for batch analysis.
"""

import streamlit as st
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import docx
from fpdf import FPDF

# --- Load NLP model ---
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# --- Helpers to extract text ---
def extract_text(file):
    """Extract text from uploaded file (pdf, docx, or txt)."""
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return " ".join([p.text for p in doc.paragraphs])
    else:
        return file.read().decode("utf-8", errors="ignore")

def clean_text(text):
    """Lowercase, remove special chars, lemmatize with spaCy."""
    doc = nlp(text.lower())
    tokens = [tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop]
    return " ".join(tokens)

def compute_similarity(resume, jd):
    """Compute cosine similarity between resume and JD."""
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform([resume, jd])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return score, vect.get_feature_names_out()

def extract_keywords(text, top_n=20):
    """Simple keyword extraction using word frequency."""
    words = re.findall(r"\w+", text.lower())
    freq = pd.Series(words).value_counts()
    return freq.head(top_n).index.tolist()

def missing_keywords(resume_text, jd_text):
    """Find which JD keywords are missing in resume."""
    jd_keywords = set(extract_keywords(jd_text, top_n=30))
    resume_keywords = set(extract_keywords(resume_text, top_n=30))
    missing = jd_keywords - resume_keywords
    return list(missing)

def generate_pdf_report(score, missing, suggestions):
    """Generate a simple PDF report with FPDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI Resume Analyzer Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Match Score: {score:.2%}", ln=True)

    pdf.ln(10)
    pdf.multi_cell(0, 10, txt="Missing Keywords: " + ", ".join(missing))

    pdf.ln(10)
    pdf.multi_cell(0, 10, txt="Suggestions:\n- " + "\n- ".join(suggestions))

    return pdf.output(dest="S").encode("latin1")

# --- Streamlit App ---
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📝", layout="wide")
st.title("📝 AI Resume Analyzer")
st.write("Upload your resume and paste a job description to see how well they match.")

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
with col2:
    jd_text = st.text_area("Paste Job Description")

if resume_file and jd_text:
    resume_text = extract_text(resume_file)
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    # Compute similarity score
    score, vocab = compute_similarity(resume_clean, jd_clean)

    st.subheader("📊 Match Score")
    st.metric(label="Resume vs JD", value=f"{score:.2%}")

    # Missing keywords
    missing = missing_keywords(resume_clean, jd_clean)
    st.subheader("❌ Missing Keywords")
    if missing:
        st.write(", ".join(missing))
    else:
        st.write("No major missing keywords found 🎉")

    # Suggestions
    suggestions = []
    if score < 0.6:
        suggestions.append("Add more relevant keywords from the job description.")
    if len(missing) > 0:
        suggestions.append("Incorporate missing skills naturally in your resume.")
    if not suggestions:
        suggestions.append("Your resume is already a strong match!")

    st.subheader("💡 Suggestions")
    for s in suggestions:
        st.write("- " + s)

    # Downloadable report
    pdf = generate_pdf_report(score, missing, suggestions)
    st.download_button("⬇️ Download Report", pdf, "resume_report.pdf", "application/pdf")
else:
    st.info("👉 Upload a resume and paste a job description to get started.")

