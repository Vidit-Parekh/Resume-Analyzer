import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from app.utils.nlp_utils import NLPUtils


class JobMatcher:
    """
    JobMatcher:
    Calculates resume–job description similarity using both
    semantic meaning (Sentence-BERT embeddings) and skill overlap.
    """

    def __init__(self, weight_semantic=0.6, weight_skills=0.4):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.weight_semantic = weight_semantic
        self.weight_skills = weight_skills
        self.nlp = NLPUtils()

    # -----------------------------------------------------------
    # ----------- TEXT CLEANING UTILITIES -----------------------
    # -----------------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove unwanted symbols, multiple spaces, and lowercase text."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^A-Za-z0-9\s\-\+\#]", "", text)
        return text.lower().strip()

    # -----------------------------------------------------------
    # ----------- SKILL EXTRACTION -------------------------------
    # -----------------------------------------------------------

    def _extract_skills(self, text: str) -> list:
        """Extract possible skill keywords using NLPUtils."""
        return self.nlp.extract_skills_from_text(text)

    # -----------------------------------------------------------
    # ----------- SEMANTIC SIMILARITY ----------------------------
    # -----------------------------------------------------------

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity using Sentence-BERT embeddings."""
        emb_a = self.model.encode(text_a, convert_to_tensor=True)
        emb_b = self.model.encode(text_b, convert_to_tensor=True)
        similarity = util.cos_sim(emb_a, emb_b).item()
        return round(similarity * 100, 2)

    # -----------------------------------------------------------
    # ----------- SKILL MATCHING LOGIC ---------------------------
    # -----------------------------------------------------------

    def _skill_match(self, resume_skills: list, job_skills: list) -> float:
        """Return skill match percentage."""
        if not resume_skills or not job_skills:
            return 0.0

        resume_skills = [s.lower() for s in resume_skills]
        job_skills = [s.lower() for s in job_skills]

        matched = [s for s in resume_skills if s in job_skills]
        match_score = len(matched) / len(job_skills)
        return round(match_score * 100, 2)

    # -----------------------------------------------------------
    # ----------- MAIN ANALYSIS FUNCTION ------------------------
    # -----------------------------------------------------------

    def analyze_from_text(self, resume_text: str, job_text: str,
                          weight_semantic=None, weight_skills=None):
        """Main method to analyze resume vs job description."""
        weight_semantic = weight_semantic or self.weight_semantic
        weight_skills = weight_skills or self.weight_skills

        # Clean input text
        resume_text = self.clean_text(resume_text)
        job_text = self.clean_text(job_text)

        # --- Semantic Similarity ---
        semantic_score = self._semantic_similarity(resume_text, job_text)

        # --- Skill Extraction & Matching ---
        resume_skills = self._extract_skills(resume_text)
        job_skills = self._extract_skills(job_text)
        skill_score = self._skill_match(resume_skills, job_skills)

        # --- Weighted Overall Score ---
        overall = (weight_semantic * semantic_score) + (weight_skills * skill_score)

        return {
            "overall_match_score": round(overall, 2),
            "semantic_similarity": round(semantic_score, 2),
            "skill_match": round(skill_score, 2),
            "matched_skills": [s for s in resume_skills if s.lower() in [x.lower() for x in job_skills]],
            "missing_skills": [s for s in job_skills if s.lower() not in [x.lower() for x in resume_skills]],
        }
