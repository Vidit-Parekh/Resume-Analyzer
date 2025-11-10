import re
import spacy
from collections import Counter

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# ----------------------------
# 1️⃣ Predefined Skill List
# ----------------------------
# You can expand this list over time or load from a CSV
TECH_SKILLS = {
    "python", "java", "c++", "c", "javascript", "typescript", "html", "css",
    "react", "angular", "nodejs", "express", "flask", "django",
    "sql", "mysql", "postgresql", "mongodb", "nosql",
    "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn",
    "tensorflow", "keras", "pytorch", "nlp", "transformers",
    "docker", "kubernetes", "aws", "azure", "gcp",
    "git", "linux", "bash", "api", "graphql", "rest",
    "ml", "ai", "data analysis", "data visualization", "deep learning"
}

NON_TECH_SKILLS = {
    "communication", "leadership", "teamwork", "problem solving", "creativity",
    "critical thinking", "project management", "time management",
    "adaptability", "collaboration", "organization", "attention to detail"
}

ALL_SKILLS = TECH_SKILLS.union(NON_TECH_SKILLS)

# ----------------------------
# 2️⃣ Extract Skills Function
# ----------------------------
def extract_skills(text: str):
    """
    Extracts known skills from text using NLP token matching.
    Returns a Counter (skill -> frequency).
    """
    text = text.lower()
    doc = nlp(text)
    found_skills = []

    for token in doc:
        word = token.text.strip().lower()
        if word in ALL_SKILLS:
            found_skills.append(word)

    # Also check for multi-word skills
    for phrase in ALL_SKILLS:
        if " " in phrase and phrase in text:
            found_skills.append(phrase)

    return Counter(found_skills)


# ----------------------------
# 3️⃣ Compare Resume vs JD
# ----------------------------
def compare_skills(resume_text: str, job_text: str):
    resume_skills = set(extract_skills(resume_text).keys())
    job_skills = set(extract_skills(job_text).keys())

    matched = resume_skills.intersection(job_skills)
    missing = job_skills - resume_skills

    if len(job_skills) == 0:
        skill_match_score = 0.0
    else:
        skill_match_score = round((len(matched) / len(job_skills)) * 100, 2)

    return {
        "resume_skills": sorted(list(resume_skills)),
        "job_skills": sorted(list(job_skills)),
        "matched_skills": sorted(list(matched)),
        "missing_skills": sorted(list(missing)),
        "skill_match_score": skill_match_score,
    }


# ----------------------------
# 4️⃣ Example Test
# ----------------------------
if __name__ == "__main__":
    resume = """Experienced ML Engineer skilled in Python, TensorFlow, and Docker.
                Worked with NLP models and AWS deployment."""
    job = """We are seeking a Machine Learning Engineer proficient in Python, TensorFlow,
             Docker, and Kubernetes. Experience with AWS is a plus."""
    
    result = compare_skills(resume, job)
    print(result)
