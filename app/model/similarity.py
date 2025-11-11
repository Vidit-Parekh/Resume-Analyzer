import numpy as np
from sentence_transformers import SentenceTransformer, util
from app.util.nlp_utils import extract_skills_advanced

model_cache = {}

def load_model(name="all-mpnet-base-v2"):
    if name not in model_cache:
        model_cache[name] = SentenceTransformer(name)
    return model_cache[name]


def text_to_embedding(text, model_name="all-mpnet-base-v2"):
    model = load_model(model_name)
    return model.encode(text, convert_to_tensor=True)


def compute_semantic_similarity(resume_text, job_desc, model_name="all-mpnet-base-v2"):
    if not resume_text or not job_desc:
        return 0.0
    emb_resume = text_to_embedding(resume_text, model_name)
    emb_job = text_to_embedding(job_desc, model_name)
    return util.cos_sim(emb_resume, emb_job).item()


def compute_skill_match(resume_text, job_desc):
    resume_skills = set(extract_skills_advanced(resume_text))
    job_skills = set(extract_skills_advanced(job_desc))

    if not job_skills:
        return 0.0, [], []

    matched = resume_skills.intersection(job_skills)
    ratio = len(matched) / len(job_skills)
    return round(ratio * 100, 2), sorted(matched), sorted(job_skills)


def compute_overall_match(resume_text, job_desc, model_name="all-mpnet-base-v2", weight_semantic=0.6, weight_skills=0.4):
    sem = compute_semantic_similarity(resume_text, job_desc, model_name)
    skill_score, matched_skills, job_skills = compute_skill_match(resume_text, job_desc)

    final = (weight_semantic * sem * 100) + (weight_skills * skill_score)
    return {
        "semantic_similarity": round(sem * 100, 2),
        "skill_match_percentage": skill_score,
        "matched_skills": matched_skills,
        "job_skills_found": job_skills,
        "overall_match_score": round(final, 2)
    }
