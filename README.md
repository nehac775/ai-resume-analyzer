# AI Resume Analyzer (Python + Streamlit)

Analyze how well a resume matches a job description. Extracts skills, finds missing keywords, and
computes an overall match score with actionable suggestions. Built to be internship‑ready and
portfolio‑friendly.

## ✨ Features
- Upload a **resume** (PDF/DOCX/TXT) and paste or upload a **job description**.
- Extract **tech skills** using a curated catalog + fuzzy matching.
- Compute **TF‑IDF cosine similarity** between resume and JD.
- Show **coverage metrics** (skills overlap, missing keywords).
- Generate **suggestions** and a **downloadable report** (Markdown).
- One‑click deploy to **Streamlit Cloud**.

## 🧰 Tech Stack
- Python 3.10+
- Streamlit, scikit-learn, spaCy, rapidfuzz, pandas, numpy, pypdf, python-docx, altair

## 🚀 Quickstart (Local)
```bash
# 1) Clone & enter
git clone <YOUR_FORK_URL>
cd ai-resume-analyzer

# 2) Create & activate a virtual env (choose one way)
python -m venv .venv && source .venv/bin/activate      # macOS/Linux
# OR
python -m venv .venv && .\.venv\Scripts\activate     # Windows (PowerShell)

# 3) Install deps
pip install -r requirements.txt

# 4) Download the small English spaCy model (first run only)
python -m spacy download en_core_web_sm

# 5) Run the app
streamlit run app.py
```

Open the printed local URL (e.g., http://localhost:8501).

## 🌐 Deploy to Streamlit Community Cloud
1. Push this repo to **GitHub** (see steps below).
2. Go to **https://streamlit.io/cloud** → *New app*.
3. Connect your GitHub repo, select `main` branch and `app.py` as the entrypoint.
4. Click **Deploy**. That’s it!

> Tip: If deployment fails due to spaCy model, add a file named `packages.txt` with the line
> `en-core-web-sm` and set `PYTHONPATH` accordingly, or keep the on‑startup downloader in `app.py`.

## 📁 Project Structure
```
ai-resume-analyzer/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
├── src/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── nlp_utils.py
│   ├── text_utils.py
│   └── skills_catalog.py
├── data/
│   ├── sample_resume.txt
│   └── sample_job_description.txt
└── tests/
    └── test_analyzer_smoke.py
```

## 🧪 Fast Smoke Test
```bash
pytest -q
```

## 🧵 How the Score Works
- **0.6 × TF‑IDF similarity** (resume vs JD) +
- **0.4 × skills coverage** (intersection of JD skills vs resume skills)

## 📝 License
MIT — see `LICENSE`.
