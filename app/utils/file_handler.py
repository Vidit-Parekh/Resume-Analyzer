import pdfplumber
import docx

def extract_text_from_file(uploaded_file):
    """
    Extracts text from PDF, DOCX, or TXT files.
    """
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif file_name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif file_name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    else:
        return ""


def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])
