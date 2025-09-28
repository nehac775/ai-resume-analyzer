# src/text_utils.py
from typing import Union
from pypdf import PdfReader
from docx import Document

import io, re

def _from_pdf(file_like: io.BytesIO) -> str:
    pdf = PdfReader(file_like)
    text_parts = []
    for page in pdf.pages:
        t = page.extract_text() or ""
        text_parts.append(t)
    return "\n".join(text_parts)

def _from_docx(file_like: io.BytesIO) -> str:
    # python-docx expects a path or a file-like object with seek/ tell support
    file_like.seek(0)
    doc = Document(file_like)
    return "\n".join([p.text for p in doc.paragraphs])

def load_text_from_any(uploaded_file: io.BytesIO, name: str) -> str:
    name = name.lower()
    data = uploaded_file.read()
    buf = io.BytesIO(data)
    if name.endswith(".pdf"):
        return _from_pdf(buf)
    elif name.endswith(".docx"):
        return _from_docx(buf)
    elif name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    else:
        # Try to decode as text anyway
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""
