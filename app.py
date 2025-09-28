# app.py
import io
import pathlib
import streamlit as st
import pandas as pd
from src.text_utils import load_text_from_any
from src.analyzer import analyze_resume_vs_jd, build_report_markdown
from src.skills_catalog import ALL_SKILLS_CANONICAL

st.set_page_config(page_title="AI Resume Analyzer", page_icon="🧠", layout="wide")

st.title("🧠 AI Resume Analyzer")
st.write("Evaluate how well a resume matches a job description. Get a match score, skills coverage, and actionable suggestions.")

with st.expander("How it works", expanded=False):
    st.markdown("""
    - Upload your **resume** (PDF/DOCX/TXT).  
    - Paste or upload a **job description**.  
    - The app extracts skills, computes TF‑IDF similarity, and shows what's missing.  
    - Download a **Markdown report** you can keep or attach to applications.
    """)

colA, colB = st.columns(2, gap="large")

with colA:
    st.subheader("1) Resume")
    resume_file = st.file_uploader("Upload resume (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
    resume_text = ""
    if resume_file is not None:
        resume_text = load_text_from_any(resume_file, resume_file.name)

    sample_resume = st.checkbox("Use sample resume", value=False)
    if sample_resume:
        from pathlib import Path
        sample_path = Path("data/sample_resume.txt")
        resume_text = sample_path.read_text(encoding="utf-8")

    st.text_area("Resume text (preview / editable)", value=resume_text, height=260, key="resume_text_area")

with colB:
    st.subheader("2) Job Description")
    jd_source = st.radio("Provide JD as:", ["Paste text", "Upload file (.txt)"], horizontal=True)
    jd_text = ""
    if jd_source == "Paste text":
        jd_text = st.text_area("Paste job description here", height=260, key="jd_text_area")
    else:
        jd_file = st.file_uploader("Upload JD (.txt)", type=["txt"], key="jd_file")
        if jd_file is not None:
            jd_text = jd_file.read().decode("utf-8", errors="ignore")
    sample_jd = st.checkbox("Use sample JD", value=False, key="use_sample_jd")
    if sample_jd:
        from pathlib import Path
        jd_text = Path("data/sample_job_description.txt").read_text(encoding="utf-8")

run_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

if run_btn:
    resume_text_final = st.session_state.get("resume_text_area", "").strip()
    jd_text_final = st.session_state.get("jd_text_area", "").strip() if jd_source == "Paste text" else jd_text.strip()
    if not resume_text_final or not jd_text_final:
        st.error("Please provide both resume and job description.")
        st.stop()

    with st.spinner("Analyzing..."):
        result = analyze_resume_vs_jd(resume_text_final, jd_text_final)

    st.success("Analysis complete!")

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Match Score (0‑100)", f"{result['overall_score']:.1f}")
    c2.metric("TF‑IDF Similarity (0‑100)", f"{result['similarity_score']*100:.1f}")
    c3.metric("Skills Coverage (0‑100)", f"{result['skills_coverage']*100:.1f}")

    st.divider()

    st.subheader("Skills Overview")
    left, right = st.columns(2)
    with left:
        st.markdown("**Skills found in Resume**")
        st.write(", ".join(sorted(result["resume_skills"])) if result["resume_skills"] else "_None detected_")
    with right:
        st.markdown("**Skills expected from JD**")
        st.write(", ".join(sorted(result["jd_skills"])) if result["jd_skills"] else "_None detected_")

    st.markdown("**Missing (prioritized):** " + (", ".join(result["missing_skills"]) if result["missing_skills"] else "_No major gaps_"))

    st.subheader("Keyword Gaps (beyond skills)")
    if result["missing_keywords"]:
        df_kw = pd.DataFrame({"Missing Keyword": result["missing_keywords"]})
        st.dataframe(df_kw, use_container_width=True, hide_index=True)
    else:
        st.write("_No obvious keyword gaps._")

    st.subheader("Suggestions")
    for tip in result["suggestions"][:8]:
        st.markdown(f"- {tip}")

    st.subheader("Download Report")
    report_md = build_report_markdown(result, top_n_suggestions=8)
    st.download_button(
        label="⬇️ Download Markdown report",
        data=report_md.encode("utf-8"),
        file_name="resume_analysis_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    with st.expander("Debug / Raw Output"):
        st.json({k: v for k, v in result.items() if k not in ("resume_text", "jd_text")})

st.caption("Pro tip: Include this project link on LinkedIn and your resume.")
