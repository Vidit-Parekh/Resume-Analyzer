import re
import spacy
from spacy.matcher import PhraseMatcher

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# Define skill keywords
SKILL_TERMS = [
    "python", "java", "c++", "c#", "r", "sql", "javascript", "html", "css",
    "machine learning", "deep learning", "data analysis", "data visualization",
    "natural language processing", "computer vision", "feature engineering",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "git", "docker", "kubernetes", "jenkins", "aws", "azure", "google cloud",
    "tableau", "power bi", "excel", "api development", "object-oriented programming",
    "rest api", "microservices", "data structures", "algorithms",
    "communication", "leadership", "teamwork", "problem solving",
    "project management", "critical thinking", "collaboration", "adaptability"
]

patterns = [nlp.make_doc(skill) for skill in SKILL_TERMS]
matcher.add("SKILLS", patterns)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def lemmatize_text(text: str) -> str:
    if not text:
        return ""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def extract_skills_advanced(text: str):
    text = clean_text(text)
    doc = nlp(text)
    matches = matcher(doc)
    found_skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        found_skills.add(span.text.lower())
    return sorted(found_skills)
