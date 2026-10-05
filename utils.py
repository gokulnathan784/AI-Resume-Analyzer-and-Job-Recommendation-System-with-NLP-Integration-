import spacy
import re
import os
import tempfile
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# -------------------------
# NLP Preprocessing
# -------------------------
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def preprocess_text(text):
    """
    Clean text:
    - Lowercase
    - Remove stopwords/punctuation/numbers
    - Lemmatize
    Returns cleaned string.
    """
    if not text:
        return ""

    doc = nlp(text)
    tokens = [
        t.lemma_.lower()
        for t in doc
        if not t.is_stop and not t.is_punct and t.is_alpha
    ]
    return " ".join(tokens)


# -------------------------
# PDF Text Extraction
# -------------------------
def extract_text_from_pdf(path):
    text = extract_text(path)
    if text and len(text.strip()) > 50:
        return text
    # OCR fallback for scanned PDFs
    pages = convert_from_path(path, dpi=200)
    texts = []
    for page in pages:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            page.save(tmp.name, "PNG")
            t = pytesseract.image_to_string(Image.open(tmp.name))
            texts.append(t)
            os.unlink(tmp.name)
    return "\n".join(texts)


# -------------------------
# Email Extraction
# -------------------------
def extract_emails(text):
    """
    Extract email addresses from text.
    """
    if not text:
        return []
    pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
    emails = re.findall(pattern, text)
    return list(set(emails))


# -------------------------
# Phone Extraction (Improved)
# -------------------------
def extract_phones(text):
    """
    Extract and format phone numbers from text.
    Filters out invalid numbers (like years) and adds space after country code.
    """
    if not text:
        return []

    phone_pattern = re.compile(
        r"(\+?\d{1,3}[\s\-\(\)]*)?"      # Country code
        r"(\(?\d{2,4}\)?[\s\-\(\)]*)?"   # Area code
        r"(\d{6,10})"                     # Main number
    )

    matches = phone_pattern.findall(text)
    phones = []
    for match in matches:
        country_code = re.sub(r"\D", "", match[0]) if match[0] else ""
        area_code = re.sub(r"\D", "", match[1]) if match[1] else ""
        main_number = re.sub(r"\D", "", match[2]) if match[2] else ""

        digits = country_code + area_code + main_number

        # Filter unrealistic numbers (exclude if looks like year or ID)
        if len(digits) >= 7 and len(digits) <= 15:
            if not re.match(r"^(19|20)\d{2}$", digits):  # remove year-like numbers
                formatted = ""
                if country_code:
                    formatted = f"+{country_code} {area_code}{main_number}"
                else:
                    formatted = f"{area_code}{main_number}"
                phones.append(formatted.strip())

    return list(set(phones))



# -------------------------
# Skills Extraction
# -------------------------
def extract_skills(text, skills_list):
    """
    Extract skills present in text based on a provided list.
    Returns a sorted unique list of skills.
    """
    if not text or not skills_list:
        return []
    text_low = text.lower()
    found = []
    for skill in skills_list:
        if re.search(r"\b" + re.escape(skill.lower()) + r"\b", text_low):
            found.append(skill)
    return sorted(set(found))


# -------------------------
# Education Extraction
# -------------------------
def extract_education(text):
    edu_keywords = [
        "bachelor", "b.tech", "b.sc", "b.e", "msc",
        "m.tech", "mba", "phd", "diploma", "high school",
        "certificate", "degree", "graduation"
    ]
    education_lines = []
    lines = text.splitlines()
    for line in lines:
        for keyword in edu_keywords:
            if keyword in line.lower():
                education_lines.append(line.strip())
                break
    return education_lines


def extract_projects(text):
    """
    Extracts meaningful project and internship descriptions.
    Returns a list of unique project-related lines.
    """
    project_keywords = [
        "project", "internship", "capstone", "assignment"
    ]
    project_lines = set()  # use set to avoid duplicates
    lines = text.splitlines()

    for line in lines:
        line_lower = line.lower()
        for keyword in project_keywords:
            if keyword in line_lower and len(line.strip()) > 20:
                project_lines.add(line.strip())
                break

    return list(project_lines)

# -------------------------
# Debug Helper
# -------------------------
if __name__ == "__main__":
    sample_text = """
    John Doe
    Email: john.doe@example.com
    Phone: +1 (123) 456-7890
    Skills: Python, Java, Machine Learning
    Education: B.Tech Computer Science
    """
    print("Preprocessed:", preprocess_text(sample_text))
    print("Emails:", extract_emails(sample_text))
    print("Phones:", extract_phones(sample_text))
    print("Skills:", extract_skills(sample_text, ["python", "java", "ml", "deep learning"]))
    print("Education:", extract_education(sample_text))
