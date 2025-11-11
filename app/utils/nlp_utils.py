# app/utils/nlp_utils.py
"""
NLPUtils - helper utilities used across the Resume Analyzer.

Provides:
- extract_skills_from_text(text)
- preprocess_text(text)
- compute_similarity(text1, text2)   (TF-IDF cosine -> returns 0-100)
- semantic_similarity(text1, text2, model_name='all-MiniLM-L6-v2') (SBERT -> 0-100)
- skill_match_percentage(resume_text, job_text, skills_list=None)
"""

from typing import List, Tuple
import re
import logging

# third-party imports
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# SentenceTransformers only imported when semantic_similarity is used (optional)
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
except Exception:
    SentenceTransformer = None
    sbert_util = None

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Ensure spaCy model available
def _load_spacy_model(name: str = "en_core_web_sm"):
    try:
        return spacy.load(name)
    except OSError:
        # try to download automatically (safe)
        from spacy.cli import download
        download(name)
        return spacy.load(name)


NLP = _load_spacy_model("en_core_web_sm")


# Default skill dictionary (expand as needed)
DEFAULT_SKILLS = [
    "python", "java", "c++", "c#", "r", "sql", "javascript", "html", "css",
    "machine learning", "deep learning", "natural language processing",
    "nlp", "transformers", "tensorflow", "pytorch", "scikit-learn",
    "pandas", "numpy", "docker", "kubernetes", "aws", "azure", "gcp",
    "git", "rest api", "fastapi", "streamlit", "flask", "django",
    "tableau", "spark", "hadoop", "feature engineering", "data preprocessing",
    "bert", "hugging face", "huggingface", "hugging-face"
]


class NLPUtils:
    def __init__(self):
        # TF-IDF vectorizer will be created on demand
        self._vectorizer = None

    # --------------------------
    # Preprocess helpers
    # --------------------------
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Simple cleaning: strip, normalize whitespace, remove weird chars, lower-case."""
        if not text:
            return ""
        txt = str(text)
        # remove emails, URLs
        txt = re.sub(r"\S+@\S+\.\S+", " ", txt)
        txt = re.sub(r"http\S+|www\.\S+", " ", txt)
        # keep letters, numbers, common punctuation
        txt = re.sub(r"[^A-Za-z0-9\.\,\-\+\#\/\s]", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt.lower()

    # --------------------------
    # Skill extraction (phrase matching)
    # --------------------------
    def extract_skills_from_text(self, text: str, skills_list: List[str] = None) -> List[str]:
        """
        Find skills in `text` using phrase matching against a skills list.
        Returns a sorted list of unique skills found.
        """
        if not text:
            return []

        skills_ref = skills_list if skills_list is not None else DEFAULT_SKILLS
        text_l = self.preprocess_text(text)

        found = set()
        # match longer phrases first (avoid partial matches)
        for skill in sorted(skills_ref, key=lambda s: -len(s)):
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"
            if re.search(pattern, text_l):
                found.add(skill)

        # additionally, include some noun-chunk heuristics (multi-word candidates)
        try:
            doc = NLP(text)
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip().lower()
                if 1 < len(phrase.split()) <= 4 and any(w in phrase for w in ["model", "pipeline", "deployment", "nlp", "transformer", "engineer", "analysis"]):
                    found.add(phrase)
        except Exception:
            # spaCy parsing is optional; ignore errors
            logger.debug("spaCy noun chunk parsing failed in extract_skills_from_text", exc_info=True)

        return sorted(found)

    # --------------------------
    # TF-IDF (fallback) similarity
    # --------------------------
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute TF-IDF cosine similarity between two texts and return [0, 100].
        This is a fast fallback if sentence-transformers isn't available.
        """
        t1 = self.preprocess_text(text1)
        t2 = self.preprocess_text(text2)
        if not t1 or not t2:
            return 0.0

        vec = TfidfVectorizer(stop_words="english")
        tfidf = vec.fit_transform([t1, t2])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return round(float(sim * 100.0), 2)

    # --------------------------
    # Optional: Sentence-BERT semantic similarity
    # --------------------------
    def semantic_similarity(self, text1: str, text2: str, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> float:
        """
        Compute semantic similarity using SentenceTransformers (if installed).
        Returns [0,100]. Falls back to TF-IDF compute_similarity if SBERT not available.
        """
        if not text1 or not text2:
            return 0.0

        if SentenceTransformer is None or sbert_util is None:
            # SBERT not installed — fallback
            logger.warning("SentenceTransformer not available; falling back to TF-IDF similarity.")
            return self.compute_similarity(text1, text2)

        try:
            model = SentenceTransformer(model_name, device=device)
            emb1 = model.encode(self.preprocess_text(text1), convert_to_tensor=True)
            emb2 = model.encode(self.preprocess_text(text2), convert_to_tensor=True)
            sim = sbert_util.cos_sim(emb1, emb2).item()
            sim = max(min(sim, 1.0), -1.0)
            return round(float(sim * 100.0), 2)
        except Exception as e:
            logger.exception("SBERT similarity failed, falling back to TF-IDF. Error: %s", e)
            return self.compute_similarity(text1, text2)

    # --------------------------
    # Skill match helper
    # --------------------------
    def skill_match_percentage(self, resume_text: str, job_text: str, skills_list: List[str] = None) -> Tuple[float, List[str], List[str]]:
        """
        Return (match_pct, matched_skills, job_skills_list)
        match_pct is percentage of job_skills that appear in resume.
        """
        job_skills = self.extract_skills_from_text(job_text, skills_list)
        resume_skills = self.extract_skills_from_text(resume_text, skills_list)

        if not job_skills:
            return 0.0, [], []

        matched = [s for s in job_skills if s in resume_skills]
        pct = round((len(matched) / len(job_skills)) * 100.0, 2) if job_skills else 0.0
        return pct, sorted(matched), sorted(job_skills)


# Module-level quick test (only if run directly)
if __name__ == "__main__":
    utils = NLPUtils()
    r = """
    Experienced ML Engineer. Skills: Python, TensorFlow, Docker, Kubernetes.
    Worked on model deployment pipelines and feature engineering.
    """
    jd = """
    Looking for ML Engineer with experience in Python, TensorFlow, Kubernetes, Docker, and feature engineering.
    """
    print("Extracted resume skills:", utils.extract_skills_from_text(r))
    print("Extracted job skills:", utils.extract_skills_from_text(jd))
    print("Skill match %:", utils.skill_match_percentage(r, jd))
    print("TF-IDF sim %:", utils.compute_similarity(r, jd))
    print("SBERT sim % (if available):", utils.semantic_similarity(r, jd))
