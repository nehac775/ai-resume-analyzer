# src/skills_catalog.py
# Curated catalog of common CS/SE/Data skills with simple aliases.
# You can expand this over time or load from a JSON file.

from collections import defaultdict

SKILLS = {
    "Programming": [
        "python", "java", "c++", "c", "go", "rust", "javascript", "typescript",
    ],
    "Data & ML": [
        "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "keras",
        "pytorch", "matplotlib", "seaborn", "plotly", "statsmodels", "prophet",
        "jupyter", "notebook",
    ],
    "Databases": [
        "sql", "postgresql", "mysql", "sqlite", "mongodb", "redis",
    ],
    "Web & APIs": [
        "flask", "django", "fastapi", "rest", "graphql", "html", "css",
        "react", "node", "express",
    ],
    "DevOps": [
        "git", "github", "gitlab", "ci/cd", "cicd", "docker", "kubernetes",
        "aws", "gcp", "azure", "linux",
    ],
    "Testing": [
        "pytest", "unittest", "integration testing", "unit testing",
    ],
    "NLP": [
        "spacy", "nltk", "transformers", "hugging face",
    ],
    "Other": [
        "oop", "data structures", "algorithms", "design patterns",
    ],
}

ALIASES = {
    "sklearn": "scikit-learn",
    "ci cd": "ci/cd",
    "ci-cd": "ci/cd",
    "tf": "tensorflow",
    "np": "numpy",
    "py torch": "pytorch",
    "postgres": "postgresql",
    "js": "javascript",
}

def canonicalize(skill: str) -> str:
    s = skill.strip().lower()
    return ALIASES.get(s, s)

# Flat canonical set
ALL_SKILLS_CANONICAL = sorted({canonicalize(s) for cat in SKILLS.values() for s in cat})

# Category lookup for pretty grouping (optional)
CATEGORY_BY_SKILL = {}
for cat, items in SKILLS.items():
    for s in items:
        CATEGORY_BY_SKILL[canonicalize(s)] = cat
