# app/model/resume_parser.py

import re
import spacy

class ResumeParser:
    def __init__(self):
        # Load English NLP model for Named Entity Recognition
        self.nlp = spacy.load("en_core_web_sm")
        
        # Education keywords to detect common degrees
        self.education_keywords = [
            "B.Tech", "B.E", "Bachelor", "M.Tech", "M.E", "Master",
            "B.Sc", "M.Sc", "PhD", "Doctorate", "Diploma", "MBA"
        ]
        
        # Experience pattern
        self.experience_pattern = re.compile(
            r'(\d+)\s+(?:years?|yrs?)\s+(?:of\s+)?experience', re.IGNORECASE
        )

        # Email pattern
        self.email_pattern = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
        
        # Skill keywords (you can expand this list later)
        self.common_skills = [
            "Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
            "SQL", "Java", "C++", "Docker", "Kubernetes", "Git", "NLP",
            "Communication", "Leadership", "Teamwork"
        ]

    def extract_name(self, doc):
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return None

    def extract_email(self, text):
        match = self.email_pattern.search(text)
        return match.group() if match else None

    def extract_education(self, text):
        education = []
        for keyword in self.education_keywords:
            if re.search(keyword, text, re.IGNORECASE):
                education.append(keyword)
        return list(set(education))

    def extract_experience(self, text):
        match = self.experience_pattern.search(text)
        return match.group() if match else None

    def extract_skills(self, text):
        found = []
        for skill in self.common_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
                found.append(skill)
        return list(set(found))

    def parse(self, text):
        """Main method to extract structured resume data."""
        doc = self.nlp(text)
        
        name = self.extract_name(doc)
        email = self.extract_email(text)
        education = self.extract_education(text)
        experience = self.extract_experience(text)
        skills = self.extract_skills(text)
        
        return {
            "name": name,
            "email": email,
            "education": education,
            "experience": experience,
            "skills": skills
        }

if __name__ == "__main__":
    sample_text = """
    Vidit Parekh
    Email: vidit@example.com
    B.Tech in Artificial Intelligence and Machine Learning
    2 years of experience as Machine Learning Engineer at XYZ Corp.
    Skills: Python, TensorFlow, Docker, Communication, Teamwork
    """
    parser = ResumeParser()
    result = parser.parse(sample_text)
    print(result)
