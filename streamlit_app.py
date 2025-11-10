import streamlit as st
from app.model.similarity import compute_overall_match
import spacy

# =========================
# Streamlit Page Setup
# =========================
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI Resume & Job Description Analyzer")
st.markdown("---")

# =========================
# File Upload + Input
# =========================
st.header("📄 Upload Your Resume")
uploaded_resume = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])

st.header("💼 Paste Job Description")
job_description = st.text_area("Enter or paste the job description here", height=200)

# =========================
# Process Button
# =========================
if st.button("🔍 Analyze Match"):
    if uploaded_resume is not None and job_description.strip():
        with st.spinner("Processing resume and analyzing match..."):
            # Read resume content
            if uploaded_resume.name.endswith(".pdf"):
                import fitz  # PyMuPDF
                doc = fitz.open(stream=uploaded_resume.read(), filetype="pdf")
                resume_text = " ".join([page.get_text("text") for page in doc])
            else:
                resume_text = uploaded_resume.read().decode("utf-8")

            # Compute similarity
            similarity_score = compute_overall_match(resume_text, job_description)

        st.markdown("---")
        st.subheader("🧩 Extracted Skills from Resume")

        if isinstance(similarity_score, dict):
            st.write(", ".join(similarity_score.get("resume_skills", [])))

        st.markdown("### ⚙️ Analyzing Match...")
        st.subheader("📊 Match Results")

        # Determine final display score
        if isinstance(similarity_score, dict):
            if "overall_match_score" in similarity_score:
                display_score = similarity_score["overall_match_score"]
            elif "overall_score" in similarity_score:
                display_score = similarity_score["overall_score"]
            else:
                display_score = 0
        else:
            display_score = similarity_score

        st.metric(label="Overall Match Score", value=f"{display_score:.2f}%")
        st.progress(display_score / 100)

        st.markdown("---")
        st.subheader("Detailed Breakdown")

        if isinstance(similarity_score, dict):
            st.markdown(f"- **Semantic Similarity:** {similarity_score.get('semantic_similarity', 0):.2f}")
            st.markdown(f"- **Skill Match Percentage:** {similarity_score.get('skill_match_percentage', 0):.2f}")
            st.markdown(f"- **Matched Skills:** {similarity_score.get('matched_skills', [])}")
            st.markdown(f"- **Job Skills Found:** {similarity_score.get('job_skills', [])}")
            st.markdown(f"- **Overall Match Score:** {similarity_score.get('overall_match_score', 0):.2f}")
        else:
            st.warning("Unable to retrieve detailed breakdown — check the similarity function output.")

    else:
        st.error("⚠️ Please upload a resume and provide a job description to proceed.")

# =========================
# Sidebar Information
# =========================
st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown(
    """
    This app uses advanced **NLP and embedding models** to analyze:
    - Skill match between your resume and a job description  
    - Semantic similarity of your experience and job requirements  
    - Overall compatibility score  

    Built using **SpaCy, Sentence Transformers, and Streamlit**.
    """
)
