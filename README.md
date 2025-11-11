# 🧠 Resume Analyzer using AI  

![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12+-yellow?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Linux%20|%20Streamlit%20Cloud-9cf?style=for-the-badge)

---

## 🌌 Overview  

**Resume Analyzer using AI** is an NLP-powered web application that evaluates resumes against job descriptions using semantic similarity and skill extraction.  
It helps candidates understand how well their resumes align with a target job role by providing an **Overall Match Score**, **Skill Match Percentage**, and **Semantic Similarity Index**.

The app leverages **Transformer-based models**, **cosine similarity**, and **keyword analysis** to generate accurate results.

---

## 🧭 Architecture Diagram  

<div align="center">
  <img src="A_flowchart_in_the_digital_image_illustrates_an_AI.png" alt="Architecture Diagram" width="85%">
</div>

---

## ⚙️ Tech Stack  

| Category | Technologies |
|-----------|---------------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.12, FastAPI (for future integration) |
| **NLP Models** | SpaCy, Sentence Transformers (BERT-based) |
| **Deployment** | Streamlit Cloud |
| **Containerization (Optional)** | Docker, Kubernetes |
| **Other Tools** | PyMuPDF (fitz), Scikit-learn, Pandas, NumPy |

---

## 🚀 Features  

- 📄 Extracts and parses resume content (PDF)  
- 🧠 Performs semantic similarity analysis between resume and job description  
- 🔍 Identifies matched and missing skills  
- 📊 Provides detailed match breakdown  
- ☁️ Deployable directly on **Streamlit Cloud**  

---

## 🧩 Folder Structure  
```bash
Resume-Analyzer/
│
├── app/
│ ├── model/
│ │ ├── job_matcher.py
│ ├── utils/
│ │ ├── nlp_utils.py
│
├── data/
│ ├── sample_resume.pdf
│ ├── job_description.txt
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── architecture_diagram.png

```

---

## 💻 Installation  

### 1️⃣ Clone the repository
```bash
git clone https://github.com/viditparekh/Resume-Analyzer.git
cd Resume-Analyzer
```
### 2️⃣ Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

🧠 Example Output Tested on:
```bash

Resume: Priya Mehta

Job Description: ML Engineer Intern

Metric	Score
Overall Match	28.75%
Semantic Similarity	23.91%
Skill Match	36.00%
```

☁️ Deployment on Streamlit Cloud
```bash
1. Push your repo to GitHub

2. Go to Streamlit Cloud

3. Connect your GitHub repository

4. Select streamlit_app.py as the entry point

5. Click Deploy
```
📘 Future Enhancements

  - Integration with FastAPI backend

  - Add LinkedIn job scraping

  - Auto-recommend missing skills

  - Multi-resume batch analysis

👨‍💻 Developed By

  Vidit Parekh
   |  University of Cincinnati
   |  AI & ML Enthusiast | Graduate Student | Aspiring ML Engineer



🪪 License
This project is licensed under the MIT License — feel free to use, modify, and share.

⭐ If you like this project, consider giving it a star on GitHub!

---