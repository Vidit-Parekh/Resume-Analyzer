# streamlit_app.py
import streamlit as st
from app.model.job_matcher import JobMatcher
import fitz  # PyMuPDF
import io

st.set_page_config(page_title="AI Resume Analyzer", page_icon="🤖", layout="wide")

# Title
st.title("🤖 AI Resume & Job Description Analyzer")
st.write("Upload your resume and job description to analyze skill and semantic match.")

# Cache the matcher to avoid reloading models on every rerun
@st.cache_resource
def get_job_matcher():
    return JobMatcher()

jm = get_job_matcher()


# Helper function to read PDF files
def extract_text_from_pdf(file):
    text = ""
    pdf_document = fitz.open(stream=io.BytesIO(file.read()), filetype="pdf")
    for page_num in range(pdf_document.page_count):
        text += pdf_document.load_page(page_num).get_text("text")
    return text.strip()


# File upload section
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
with col2:
    job_file = st.file_uploader("🧾 Upload Job Description (PDF)", type=["pdf"])

# Or manual text input
st.markdown("### Or Paste Text Below")
resume_text = st.text_area("Resume Text", height=180, placeholder="Paste your resume text here...")
job_desc = st.text_area("Job Description Text", height=180, placeholder="Paste job description here...")

# Extract text from uploaded files if available
if resume_file:
    resume_text = extract_text_from_pdf(resume_file)
if job_file:
    job_desc = extract_text_from_pdf(job_file)

# Run Analysis
if st.button("🔍 Analyze Match", use_container_width=True):
    if not resume_text or not job_desc:
        st.error("Please provide both resume and job description text.")
    else:
        with st.spinner("Analyzing... Please wait ⏳"):
            result = jm.analyze_from_text(resume_text, job_desc)

        st.success("✅ Analysis Complete!")

        # Display results
        st.markdown("## 📊 Results")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Overall Match", f"{result['overall_match_score']:.2f}%")
        col_b.metric("Semantic Similarity", f"{result['semantic_similarity']:.2f}%")
        col_c.metric("Skill Match", f"{result['skill_match']:.2f}%")

        st.progress(result["overall_match_score"] / 100)

        # Display skills
        st.markdown("### 💡 Extracted Skills")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Matched Skills")
            st.write(", ".join(result["matched_skills"]) or "No skills matched.")
        with col2:
            st.subheader("Missing Skills")
            st.write(", ".join(result["missing_skills"]) or "All skills matched!")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, spaCy, and scikit-learn")
