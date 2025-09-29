"""
Resume Builder (my version)

Why I built it:
---------------
I was tired of messing around with Word/Google Docs for resumes,
so I thought it'd be cool to just enter data and get a PDF out.
Also gave me an excuse to practice Streamlit + PDF generation.

Notes & Struggles:
------------------
- My first try used `cell` instead of `multi_cell` → text was overflowing off the page.
- I forgot to add page breaks and the PDF looked broken.
- Spacing/alignment was a nightmare at first; fixed by using consistent fonts/sizes.
- TODO: Add multiple template designs (modern vs simple).
- TODO: Add job description keyword highlighting (to match ATS systems).
"""

import streamlit as st
from fpdf import FPDF

st.set_page_config(page_title="Resume Builder", page_icon="📄", layout="centered")
st.title("📄 Resume Builder")

# --- Input fields ---
# I kept the form minimal for now (name, email, summary, skills, experience).
# In the future I might split sections with tabs, but this works.
name = st.text_input("Full Name")
email = st.text_input("Email")
summary = st.text_area("Summary / About Me", height=120, placeholder="Write a short intro here...")

# I almost used a multiselect for skills, but text input is faster.
skills = st.text_area("Skills (comma separated)", placeholder="Python, SQL, Streamlit, Pandas")

# Experience format is a bit strict → Company | Role | Dates | Bullets
# I considered a dynamic form with add buttons, but too complex for now.
experience = st.text_area(
    "Job Experience (one per line, format: Company | Role | Start-End | Bullet1; Bullet2)",
    height=160
)

def parse_experience(text):
    """
    Parse the raw job text input into a structured list.
    Example input line:
        Acme Corp | Data Intern | Jun 2024 - Aug 2024 | cleaned data; built dashboard
    """
    jobs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            # Splitting with maxsplit=3 → ensures bullets all stay in the last part.
            company, role, dates, bullets = [x.strip() for x in line.split("|", maxsplit=3)]
            jobs.append({
                "company": company,
                "role": role,
                "dates": dates,
                # Bullets are split by ";" → easy to type multiple ones
                "bullets": [b.strip() for b in bullets.split(";") if b.strip()]
            })
        except ValueError:
            # If the user types in wrong format, skip it but warn them.
            st.warning(f"Skipping badly formatted line: {line}")
    return jobs

def generate_pdf(name, email, summary, skills, jobs):
    """
    Build the actual PDF with fpdf.
    I experimented with different fonts and sizes until things looked decent.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)

    # --- Header ---
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, name, ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, email, ln=True, align="C")
    pdf.ln(5)

    # --- Summary ---
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Summary", ln=True)
    pdf.set_font("Arial", "", 11)
    # I had to switch to multi_cell or else long lines ran off the page.
    pdf.multi_cell(0, 6, summary)
    pdf.ln(4)

    # --- Skills ---
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Skills", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, skills)
    pdf.ln(4)

    # --- Experience ---
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Experience", ln=True)

    for job in jobs:
        pdf.set_font("Arial", "B", 11)
        # NOTE: I used em dash here for style (—)
        pdf.cell(0, 6, f"{job['role']} — {job['company']} ({job['dates']})", ln=True)
        pdf.set_font("Arial", "", 11)
        for bullet in job["bullets"]:
            # Using multi_cell again for safety (bullets can be long)
            pdf.multi_cell(0, 6, f"- {bullet}")
        pdf.ln(1)

    return pdf.output(dest="S").encode("latin1")

# --- Button actions ---
if st.button("Generate PDF"):
    if not name or not email:
        # A resume without name/email doesn’t make sense
        st.error("Name and Email are required!")
    else:
        jobs = parse_experience(experience)
        pdf_bytes = generate_pdf(name, email, summary, skills, jobs)
        # This lets me download the file straight from browser
        st.download_button("Download Resume PDF", pdf_bytes, file_name="resume.pdf", mime="application/pdf")
        st.success("PDF generated successfully ✔")

