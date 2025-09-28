# tests/test_analyzer_smoke.py
from src.analyzer import analyze_resume_vs_jd

def test_smoke():
    resume = "Python pandas numpy scikit-learn Flask SQL Docker Git"
    jd = "We want Python, SQL, REST APIs, pandas, numpy; CI/CD and Docker are a plus."
    out = analyze_resume_vs_jd(resume, jd)
    assert 0.0 <= out["similarity_score"] <= 1.0
    assert 0.0 <= out["skills_coverage"] <= 1.0
    assert "python" in out["resume_skills"]
    assert "sql" in out["jd_skills"]
