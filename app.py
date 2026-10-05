# app.py — AI Resume Analyzer (Multi-Page UI with Sidebar + Next/Back Navigation)
import os
import tempfile
import joblib
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import (
    extract_text_from_pdf,
    extract_emails,
    extract_phones,
    extract_skills,
    extract_education,
    extract_projects,

    preprocess_text,
)
import re

def compute_resume_score(text, skills_list=None):
    """
    Compute ATS-oriented resume score (0-100) and return a breakdown dict + suggestions.
    skills_list: list of known skills (your SKILLS).
    """
    lower = text.lower() if text else ""
    now_year = datetime.datetime.now().year

    # --- Contact info (10) ---
    emails = extract_emails(text)
    phones = extract_phones(text)
    contact_points = 0
    if emails and phones:
        contact_points = 10
    elif emails or phones:
        contact_points = 5
    contact_explain = []
    if emails:
        contact_explain.append("email found")
    else:
        contact_explain.append("no email")
    if phones:
        contact_explain.append("phone found")
    else:
        contact_explain.append("no phone")
    # --- Experience (20) ---
    def extract_experience_years(text):
        lower = text.lower()
        experience_keywords = ["experience", "worked", "internship", "role", "project"]

        # Only search sections mentioning experience
        exp_texts = []
        for line in lower.splitlines():
            if any(keyword in line for keyword in experience_keywords):
                exp_texts.append(line)

        exp_text = " ".join(exp_texts)

        # Search for explicit "X years" pattern
        m = re.search(r'(\d{1,2})\+?\s+years', exp_text)
        if m:
            return int(m.group(1))

        # Search for date ranges
        years_found = re.findall(r'\b(19\d{2}|20\d{2})\b', exp_text)
        if years_found and len(years_found) >= 2:
            years_found = sorted([int(y) for y in years_found])
            diff = years_found[-1] - years_found[0]
            if diff > 0 and diff < 50:  # filter unrealistic ranges
                return diff

        return 0  # default if no experience found


    years = extract_experience_years(text)

    exp_points = 0
    if years == 0:
        exp_points = 2   # fresher
        exp_note = "No work experience found"
    else:
        if years >= 10:
            exp_points = 20
        elif years >= 6:
            exp_points = 18
        elif years >= 3:
            exp_points = 14
        elif years >= 1:
            exp_points = 9
        else:
            exp_points = 4
        exp_note = f"estimated {years} years"

   
    # --- Skills (25) ---
    skills_found = extract_skills(text, skills_list or [])
    skills_count = len(set(skills_found))
    if skills_count >= 12:
        skills_points = 25
    elif skills_count >= 8:
        skills_points = 18
    elif skills_count >= 4:
        skills_points = 10
    elif skills_count >= 1:
        skills_points = 4
    else:
        skills_points = 0

    # --- Education (10) ---
    edu = extract_education(text)
    if edu:
        edu_points = 10
    else:
        edu_points = 0

    # --- Formatting / ATS friendliness (20) ---
    required_headings = ["experience", "education", "skills"]
    headings_found = sum(1 for h in required_headings if h in lower)
    # bullet points check
    bullet_pts = 0
    bullets = re.findall(r'^[\-\u2022\*\d\)]\s+', text, flags=re.M)
    bullet_count = len(bullets)
    if bullet_count >= 4:
        bullet_pts = 6
    elif 1 <= bullet_count < 4:
        bullet_pts = 3

    heading_score = int((headings_found / len(required_headings)) * 14)  # up to 14
    format_points = min(20, heading_score + bullet_pts)

    # ATS warnings: look for tables/columns characters which may break parsers
    warnings = []
    # check for many special table characters or images placeholder
    if re.search(r'\t|\|:|<img|www\.|http[s]?://', text.lower()):
        warnings.append("contains tables/links/HTML which may confuse ATS")
    # very long lines (possible PDF parsing issues)
    long_lines = [ln for ln in text.splitlines() if len(ln) > 200]
    if len(long_lines) > 3:
        warnings.append("many very long lines — PDF parsing may be messy")

    # --- Extras (15) : projects, certifications, internships ---
    cert_count = len(re.findall(r'certif', lower))
    proj_count = len(re.findall(r'\bproject\b', lower))
    intern_count = len(re.findall(r'\bintern', lower))
    extras_points = min(15, cert_count * 5 + min(5, proj_count * 2) + min(5, intern_count * 2))

    # --- Aggregate ---
    breakdown = {
        "Contact Info": (contact_points, 10, "; ".join(contact_explain)),
        "Experience": (exp_points, 20, exp_note),
        "Skills": (skills_points, 25, f"{skills_count} skill(s) found: {', '.join(skills_found[:10])}"),
        "Education": (edu_points, 10, f"{len(edu) if edu else 0} education entries"),
        "Formatting (ATS)": (format_points, 20, f"headings: {headings_found}/3, bullets: {bullet_count}"),
        "Extras (projects/certs)": (extras_points, 15, f"projects: {proj_count}, certs: {cert_count}, interns: {intern_count}")
    }

    total = sum(v[0] for v in breakdown.values())
    total = max(0, min(100, int(total)))

    # Suggestions
    suggestions = []
    if contact_points < 10:
        suggestions.append("Add a clear email and phone number at the top.")
    if exp_points < 10:
        suggestions.append("Clarify experience: add years, date ranges (e.g., 2019 - 2022) under roles.")
    if skills_points < 10:
        suggestions.append("List technical skills in a dedicated 'Skills' section; include keywords from job descriptions.")
    if edu_points == 0:
        suggestions.append("Add an Education section with degree and institute (or put 'Education' heading).")
    if format_points < 14:
        suggestions.append("Use clear section headings (Experience, Education, Skills) and bullet points per role.")
    if warnings:
        suggestions.append("Resolve ATS warnings: " + "; ".join(warnings))
    if extras_points < 5:
        suggestions.append("Add projects or certifications to increase credibility.")

    ats_friendly = format_points >= 12 and headings_found >= 2

    # prepare report dict
    report = {
        "total_score": total,
        "breakdown": breakdown,
        "suggestions": suggestions,
        "ats_friendly": ats_friendly,
        "warnings": warnings
    }
    return report


# 🔑 Admin credentials
ADMIN_USER = "admin"
ADMIN_PASS = "1234"

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

vec_path = "models/tfidf_vectorizer.joblib"
clf_path = "models/resume_classifier.joblib"

vec = joblib.load(vec_path) if os.path.exists(vec_path) else None
clf = joblib.load(clf_path) if os.path.exists(clf_path) else None

# Skills & course links
SKILLS = [
    "python","java","c++","c","c#","sql","nosql","machine learning","deep learning",
    "nlp","pandas","numpy","scikit-learn","tensorflow","pytorch","docker","kubernetes",
    "aws","azure","gcp","git","html","css","javascript","react","nodejs","spring",
    "flask","django","rest","api","devops","terraform","jenkins","prometheus","grafana"
]
COURSES = {
    "python": "YouTube: https://www.youtube.com/watch?v=_uQrJ0TkZlc | Coursera: https://www.coursera.org/courses?query=python | GeeksforGeeks: https://www.geeksforgeeks.org/python-programming-language/",
    "java": "YouTube: https://www.youtube.com/watch?v=grEKMHGYyns | Coursera: https://www.coursera.org/courses?query=java | GeeksforGeeks: https://www.geeksforgeeks.org/java/",
    "machine learning": "YouTube: https://www.youtube.com/watch?v=Gv9_4yMHFhI | Coursera: https://www.coursera.org/learn/machine-learning | GeeksforGeeks: https://www.geeksforgeeks.org/machine-learning/",
    "deep learning": "YouTube: https://www.youtube.com/watch?v=aircAruvnKk | Coursera: https://www.coursera.org/specializations/deep-learning | GeeksforGeeks: https://www.geeksforgeeks.org/deep-learning/",
    "nlp": "YouTube: https://www.youtube.com/watch?v=8rXD5-xhemo | Coursera: https://www.coursera.org/learn/natural-language-processing | GeeksforGeeks: https://www.geeksforgeeks.org/natural-language-processing-nlp/",
    "docker": "YouTube: https://www.youtube.com/watch?v=fqMOX6JJhGo | Coursera: https://www.coursera.org/learn/docker | GeeksforGeeks: https://www.geeksforgeeks.org/docker-introduction/",
    "kubernetes": "YouTube: https://www.youtube.com/watch?v=X48VuDVv0do | Coursera: https://www.coursera.org/learn/google-kubernetes-engine | GeeksforGeeks: https://www.geeksforgeeks.org/introduction-to-kubernetes/",
    "aws": "YouTube: https://www.youtube.com/watch?v=ulprqHHWlng | Coursera: https://www.coursera.org/learn/aws-fundamentals | GeeksforGeeks: https://www.geeksforgeeks.org/aws-introduction/",
    "azure": "YouTube: https://www.youtube.com/watch?v=juCNOJ1lQdc | Coursera: https://www.coursera.org/learn/azure-fundamentals | GeeksforGeeks: https://www.geeksforgeeks.org/microsoft-azure-introduction/",
    "gcp": "YouTube: https://www.youtube.com/watch?v=US4t2Ndtt4s | Coursera: https://www.coursera.org/learn/gcp-fundamentals | GeeksforGeeks: https://www.geeksforgeeks.org/google-cloud-platform-gcp-introduction/",
    "git": "YouTube: https://www.youtube.com/watch?v=SWYqp7iY_Tc | Coursera: https://www.coursera.org/learn/introduction-git-github | GeeksforGeeks: https://www.geeksforgeeks.org/git-introduction/",
    "html": "YouTube: https://www.youtube.com/watch?v=pQN-pnXPaVg | Coursera: https://www.coursera.org/learn/html | GeeksforGeeks: https://www.geeksforgeeks.org/html-tutorials/",
    "css": "YouTube: https://www.youtube.com/watch?v=1Rs2ND1ryYc | Coursera: https://www.coursera.org/learn/css | GeeksforGeeks: https://www.geeksforgeeks.org/css-tutorials/",
    "javascript": "YouTube: https://www.youtube.com/watch?v=W6NZfCO5SIk | Coursera: https://www.coursera.org/learn/javascript | GeeksforGeeks: https://www.geeksforgeeks.org/javascript-tutorial/",
    "react": "YouTube: https://www.youtube.com/watch?v=w7ejDZ8SWv8 | Coursera: https://www.coursera.org/learn/front-end-react | GeeksforGeeks: https://www.geeksforgeeks.org/reactjs-tutorial/",
    "nodejs": "YouTube: https://www.youtube.com/watch?v=TlB_eWDSMt4 | Coursera: https://www.coursera.org/learn/server-side-nodejs | GeeksforGeeks: https://www.geeksforgeeks.org/nodejs-introduction/",
    "spring": "YouTube: https://www.youtube.com/watch?v=5V9lRz2ch80 | Coursera: https://www.coursera.org/learn/spring-framework | GeeksforGeeks: https://www.geeksforgeeks.org/spring-framework-introduction/",
    "flask": "YouTube: https://www.youtube.com/watch?v=Z1RJmh_OqeA | Coursera: https://www.coursera.org/projects/rest-api-flask | GeeksforGeeks: https://www.geeksforgeeks.org/flask-introduction/",
    "django": "YouTube: https://www.youtube.com/watch?v=F5mRW0jo-U4 | Coursera: https://www.coursera.org/projects/django-web-app | GeeksforGeeks: https://www.geeksforgeeks.org/django-tutorial/",
    "rest": "YouTube: https://www.youtube.com/watch?v=lsMQRaeKNDk | Coursera: https://www.coursera.org/learn/rest-api | GeeksforGeeks: https://www.geeksforgeeks.org/restful-web-services/",
    "api": "YouTube: https://www.youtube.com/watch?v=GZvSYJDk-us | Coursera: https://www.coursera.org/learn/api-development | GeeksforGeeks: https://www.geeksforgeeks.org/api-introduction/",
    "devops": "YouTube: https://www.youtube.com/watch?v=0yWAtQ6wYNM | Coursera: https://www.coursera.org/learn/devops | GeeksforGeeks: https://www.geeksforgeeks.org/devops/",
    "terraform": "YouTube: https://www.youtube.com/watch?v=SLB_c_ayRMo | Coursera: https://www.coursera.org/projects/terraform | GeeksforGeeks: https://www.geeksforgeeks.org/introduction-to-terraform/",
    "jenkins": "YouTube: https://www.youtube.com/watch?v=FX322RVNGj4 | Coursera: https://www.coursera.org/projects/jenkins-pipeline | GeeksforGeeks: https://www.geeksforgeeks.org/jenkins/",
    "prometheus": "YouTube: https://www.youtube.com/watch?v=h4Sl21AKiDg | Coursera: https://www.coursera.org/projects/prometheus-monitoring | GeeksforGeeks: https://www.geeksforgeeks.org/prometheus-monitoring/",
    "grafana": "YouTube: https://www.youtube.com/watch?v=3mJ3KC4d7MM | Coursera: https://www.coursera.org/projects/grafana-dashboard | GeeksforGeeks: https://www.geeksforgeeks.org/grafana-introduction/",
    "sql": "YouTube: https://www.youtube.com/watch?v=HXV3zeQKqGY | Coursera: https://www.coursera.org/courses?query=sql | GeeksforGeeks: https://www.geeksforgeeks.org/sql-tutorial/",
    "nosql": "YouTube: https://www.youtube.com/watch?v=qI_g07C_Q5I | Coursera: https://www.coursera.org/courses?query=nosql | GeeksforGeeks: https://www.geeksforgeeks.org/nosql/",
    "c++": "YouTube: https://www.youtube.com/watch?v=vLnPwxZdW4Y | Coursera: https://www.coursera.org/courses?query=c%2B%2B | GeeksforGeeks: https://www.geeksforgeeks.org/c-plus-plus/",
    "c": "YouTube: https://www.youtube.com/watch?v=KJgsSFOSQv0 | Coursera: https://www.coursera.org/courses?query=c | GeeksforGeeks: https://www.geeksforgeeks.org/c-programming-language/",
    "c#": "YouTube: https://www.youtube.com/watch?v=GhQdlIFylQ8 | Coursera: https://www.coursera.org/courses?query=c%23 | GeeksforGeeks: https://www.geeksforgeeks.org/c-sharp-programming-language/",
    "pandas": "YouTube: https://www.youtube.com/watch?v=vmEHCJofslg | Coursera: https://www.coursera.org/courses?query=pandas | GeeksforGeeks: https://www.geeksforgeeks.org/python-pandas-tutorial/",
    "numpy": "YouTube: https://www.youtube.com/watch?v=QUT1VHiLmmI | Coursera: https://www.coursera.org/courses?query=numpy | GeeksforGeeks: https://www.geeksforgeeks.org/numpy-tutorials/",
    "scikit-learn": "YouTube: https://www.youtube.com/watch?v=0Lt9w-BxKFQ | Coursera: https://www.coursera.org/courses?query=scikit-learn | GeeksforGeeks: https://www.geeksforgeeks.org/scikit-learn/",
    "tensorflow": "YouTube: https://www.youtube.com/watch?v=tPYj3fFJGjk | Coursera: https://www.coursera.org/courses?query=tensorflow | GeeksforGeeks: https://www.geeksforgeeks.org/tensorflow/",
    "pytorch": "YouTube: https://www.youtube.com/watch?v=GIsg-ZUy0MY | Coursera: https://www.coursera.org/courses?query=pytorch | GeeksforGeeks: https://www.geeksforgeeks.org/pytorch/"
}

# ----------------------------
# PAGE NAVIGATION
# ----------------------------
PAGES = [
    "🏠 Home",
    "📑 Resume Analysis",
    "📋 JD Match",
    "📊 Resume Score",
    "✨ Resume Optimizer",
    "📈 Skill Gap Dashboard",
    "🎯 Career Recommendations",
    "📝 Feedback",
    "👨‍💻 Admin"
]


# Session state initialization
if "page_index" not in st.session_state:
    st.session_state.page_index = 0
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "skills_found" not in st.session_state:
    st.session_state.skills_found = []

# Sidebar navigation (only one!)
choice = st.sidebar.radio(
    "📂 Choose Section",
    PAGES,
    index=st.session_state.page_index,
    key="sidebar_page_selector"
)
st.session_state.page_index = PAGES.index(choice)

# Navigation helpers
def go_next():
    if st.session_state.page_index < len(PAGES) - 1:
        st.session_state.page_index += 1
        st.rerun()

def go_back():
    if st.session_state.page_index > 0:
        st.session_state.page_index -= 1
        st.rerun()

# Current page
page = PAGES[st.session_state.page_index]

# ----------------------------
# PAGES
# ----------------------------
if page == "🏠 Home":
    st.title("AI Resume Analyzer")
    st.write("Upload your resume and get smart recommendations!")

    name = st.text_input("Name")
    email = st.text_input("Email")
    phone = st.text_input("Mobile Number")

    uploaded = st.file_uploader("Upload PDF Resume", type=["pdf"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        st.session_state.resume_text = extract_text_from_pdf(tmp_path)
        st.success("Resume uploaded successfully!")

elif page == "📑 Resume Analysis":
    st.title("Resume Analysis 🔍")
    if not st.session_state.resume_text:
        st.warning("Please upload a resume in Home section first.")
    else:
        resume_text = st.session_state.resume_text
        st.subheader("Resume Preview")
        st.text(resume_text[:400])

        emails = extract_emails(resume_text)
        phones = extract_phones(resume_text)
        st.session_state.skills_found = extract_skills(resume_text, SKILLS)
        education = extract_education(resume_text)
        projects = extract_projects(resume_text)


        st.markdown("📧 Emails: " + (", ".join(emails) if emails else "—"))
        st.markdown("📱 Phones: " + (", ".join(phones) if phones else "—"))
        st.markdown("🛠 Skills Found: " + (", ".join(st.session_state.skills_found) if st.session_state.skills_found else "—"))
        st.markdown("🎓 Education:")
        if education:
            for e in education:
                st.text("- " + e)
        else:
            st.text("—")

        st.markdown("📂 Projects / Internships:")
        if projects:
            for p in projects:
                st.text("- " + p)
        else:
            st.text("—")


        if vec and clf:
            clean = preprocess_text(resume_text)
            pred = clf.predict(vec.transform([clean]))[0]
            st.success(f"🔮 Predicted Category: {pred}")

            # Save to log
            log_path = "data/analysis_log.csv"
            os.makedirs("data", exist_ok=True)
            new_entry = pd.DataFrame([{
                "timestamp": datetime.datetime.now(),
                "prediction": pred,
                "skills_found": ", ".join(st.session_state.skills_found) if st.session_state.skills_found else "",
            }])
            if os.path.exists(log_path):
                new_entry.to_csv(log_path, mode="a", header=False, index=False)
            else:
                new_entry.to_csv(log_path, index=False)
elif page == "📋 JD Match":
    st.title("Job Description Match 📋")

    if not st.session_state.resume_text:
        st.warning("Please upload a resume in Home section first.")
    else:
        job_desc = st.text_area("Paste job description / job ad", height=250)

        if st.button("Match Now"):
            resume_text = st.session_state.resume_text

            # --------------------------
            # 1) Clean + preprocess text
            # --------------------------
            def simple_preprocess(s):
                if not s:
                    return ""
                s = s.lower()
                s = s.replace('\n', ' ')
                s = s.replace('/', ' ')
                s = s.replace('-', ' ')
                # remove multiple spaces
                s = " ".join(s.split())
                return s

            jd_clean = simple_preprocess(job_desc)
            resume_clean = simple_preprocess(resume_text)

            # --------------------------
            # 2) TF-IDF text similarity
            # --------------------------
            # prefer trained vec when available, otherwise create local TF-IDF
            try:
                if vec:
                    q_vec = vec.transform([jd_clean])
                    r_vec = vec.transform([resume_clean])
                    text_sim = float(cosine_similarity(q_vec, r_vec)[0][0])
                else:
                    tv = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=10000)
                    mat = tv.fit_transform([jd_clean, resume_clean])
                    text_sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
            except Exception as e:
                st.warning("TF-IDF similarity error: " + str(e))
                text_sim = 0.0

            # --------------------------
            # 3) Skill-based similarity (token + fuzzy)
            # --------------------------
            # extract skills from JD and resume using your existing function
            jd_skills = extract_skills(job_desc, SKILLS)
            res_skills = extract_skills(resume_text, SKILLS)

            # lower-case sets
            jd_skills_set = set([s.lower() for s in jd_skills])
            res_skills_set = set([s.lower() for s in res_skills])

            # direct matches
            matched = jd_skills_set & res_skills_set
            missing = sorted(list(jd_skills_set - res_skills_set))

            # fuzzy near matches (helpful when terms differ slightly)
            from difflib import SequenceMatcher
            fuzzy_matches = []
            for js in list(jd_skills_set):
                if js in matched:
                    continue
                best = ("", 0.0)
                for rs in res_skills_set:
                    score = SequenceMatcher(None, js, rs).ratio()
                    if score > best[1]:
                        best = (rs, score)
                if best[1] >= 0.75:  # threshold for "close" match
                    fuzzy_matches.append((js, best[0], round(best[1], 2)))
                    # treat fuzzy match as partial match (remove from missing)
                    if js in missing:
                        missing.remove(js)

            # skill similarity score: direct matches count as 1, fuzzy as 0.6, missing 0
            direct_count = len(matched)
            fuzzy_count = len(fuzzy_matches)
            jd_total = max(1, len(jd_skills_set))  # avoid division by zero
            skill_score = (direct_count * 1.0 + fuzzy_count * 0.6) / jd_total  # 0..1

            # --------------------------
            # 4) Blend text_sim and skill_score
            #    (weights can be tuned)
            # --------------------------
            # If JD lists many explicit skills, we rely more on skill_score.
            if jd_total >= 6:
                # skills-heavy JD: weight skills more
                final_sim = 0.55 * skill_score + 0.45 * text_sim
            else:
                final_sim = 0.35 * skill_score + 0.65 * text_sim

            match_percent = round(final_sim * 100, 1)
            text_percent = round(text_sim * 100, 1)
            skill_percent = round(skill_score * 100, 1)

            # --------------------------
            # 5) Display results & breakdown
            # --------------------------
            st.metric("Resume ↔ JD Match (blended)", f"{match_percent}%")
            st.write(f"• Text similarity (TF-IDF): **{text_percent}%**")
            st.write(f"• Skill match score: **{skill_percent}%** (direct: {len(matched)}, fuzzy: {len(fuzzy_matches)})")

            # Skill overlap / details
            st.subheader("Skill Overlap")
            st.write(f"JD Skills detected: {', '.join(jd_skills) if jd_skills else 'None detected'}")
            st.write(f"Resume Skills detected: {', '.join(res_skills) if res_skills else 'None detected'}")
            st.write(f"Direct matches: {', '.join(sorted(matched)) if matched else 'None'}")
            if fuzzy_matches:
                st.write("Fuzzy matches (near):")
                for js, rs, score in fuzzy_matches:
                    st.write(f"- JD `{js}` ~ Resume `{rs}` (similarity {score})")
            st.write(f"Missing skills from JD: {', '.join(missing) if missing else 'None — good!'}")

            # --------------------------
            # 6) Smart suggestions (actionable)
            # --------------------------
            st.markdown("---")
            st.subheader("💡 Suggestions to Improve Match Score")

            # Suggest phrase mirroring when text_sim is low but skills present
            if text_percent < 60 and skill_percent >= 60:
                st.info("Your resume contains many required skills but the phrasing/context differs from the JD.")
                st.write("- Mirror key JD phrases in your **Summary** and **Experience** (use exact wording).")
                st.write("- Add a one-line 'Technical Summary' at the top listing the JD keywords.")
                st.write("- In Experience bullets mention the tool + result: e.g., 'Developed RESTful APIs using Flask, reducing latency by 20%.'")

            # Suggest adding skills when missing
            if missing:
                st.warning("You are missing some skills asked in the JD.")
                st.write("- Add these missing skills to your Skills section or mention them in project bullets.")
                for skill in missing:
                    if skill.lower() in COURSES:
                        st.markdown(f"  - **{skill}** → Learn here: [link]({COURSES[skill.lower()]})")
                    else:
                        st.markdown(f"  - **{skill}** → Search tutorials on YouTube / Coursera / Udemy")

            # Suggest expanding context if both low
            if text_percent < 50 and skill_percent < 50:
                st.warning("Both your resume phrasing and skill coverage are low for this JD.")
                st.write("- Expand experience bullets with more technical details relevant to the JD.")
                st.write("- Add 2–3 short project lines that demonstrate the required skills.")
                st.write("- Consider adding a short measurable achievement per project (performance, users, percentage).")

            # If nearly perfect, congratulate
            if match_percent >= 90:
                st.success("🌟 Excellent match! Your resume strongly aligns with this JD.")

            # --------------------------
            # 7) Optional: show example rewrites for missing skills (quick helper)
            # --------------------------
            st.markdown("---")
            st.subheader("✍️ Quick rewrite hints (paste into Resume Optimizer)")

            # For each missing skill propose a short line to add (simple templates)
            for skill in missing[:6]:  # show up to 6
                hint = f"• Implemented features using {skill} in project X to achieve <result>."
                st.code(hint, language="text")

            

elif page == "📊 Resume Score":
    st.title("Resume Score 📊 (ATS-oriented)")

    if not st.session_state.resume_text:
        st.warning("Please upload a resume in Home section first.")
    else:
        resume_text = st.session_state.resume_text

    # Compute score
    report = compute_resume_score(resume_text, SKILLS)
    total = report["total_score"]
    breakdown = report["breakdown"]
    suggestions = report["suggestions"]
    ats_friendly = report["ats_friendly"]
    warnings = report["warnings"]

    # Visual overall
    st.subheader("Overall Score")
    st.progress(total / 100)
    st.metric("ATS Resume Score", f"{total} / 100", delta=None)

    # Detailed breakdown table
    st.subheader("Score breakdown (points per category)")
    rows = []
    for k, (pts, mx, note) in breakdown.items():
        rows.append({"Category": k, "Points": pts, "Max": mx, "Notes": note})
    df_break = pd.DataFrame(rows)
    st.dataframe(df_break, use_container_width=True)

    # Explain each category with actionable tips
    st.subheader("Actionable suggestions")
    if suggestions:
        for s in suggestions:
            st.write("• " + s)
    else:
        st.write("Great — resume looks solid!")

    # ATS friendliness
    st.subheader("ATS compatibility")
    st.write("ATS friendly:" , "✅" if ats_friendly else "❌")
    if warnings:
        st.warning("Warnings that may affect ATS parsing:")
        for w in warnings:
            st.write("- " + w)

    # Show top skills found
    st.subheader("Top skills detected")
    skills_found = extract_skills(resume_text, SKILLS)
    if skills_found:
        st.write(", ".join(skills_found))
    else:
        st.write("No known skills detected. Consider adding a dedicated 'Skills' section with keywords.")

    # Show projects detected
    st.subheader("Projects Detected")
    project_lines = re.findall(r'(project[s]?|portfolio)', resume_text.lower())
    proj_count = len(project_lines)
    if proj_count > 0:
        st.write(f"{proj_count} project(s) found in resume.")
    else:
        st.write("No projects detected. Consider adding relevant projects with detailed descriptions.")

    # Downloadable short report
    csv_report = pd.DataFrame([{
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "score": total,
        "ats_friendly": ats_friendly,
        "warnings": "; ".join(warnings),
        "top_skills": ", ".join(skills_found[:10]),
        "projects_found": proj_count,
        "suggestions": " | ".join(suggestions)
    }])
    st.download_button("Download score report (CSV)", csv_report.to_csv(index=False).encode('utf-8'),
                       "resume_score_report.csv", "text/csv")

            
elif page == "✨ Resume Optimizer":
    st.title("✨ AI Resume Optimizer")
    st.write("Improve your resume sections with professional, ATS-friendly phrasing.")

    input_text = st.text_area("Paste a section of your resume (e.g., Experience or Summary):")

    if st.button("Enhance Text"):
        if input_text.strip():
            import re
            import random

            # --- Step 1: Basic cleanup
            text = input_text.strip()
            text = re.sub(r'\s+', ' ', text)

            # --- Step 2: Enhancement keywords
            replacements = {
                r"\b(i|we)\b": "Successfully",
                r"\b(made|did|worked|helped|handled|managed|created)\b": random.choice(["Developed", "Implemented", "Executed", "Engineered", "Optimized"]),
                r"\b(good|nice|great|excellent)\b": "strong",
                r"\b(using|with)\b": "leveraging",
                r"\b(team|group)\b": "cross-functional team",
                r"\b(project|task)\b": random.choice(["initiative", "solution", "application"]),
            }

            for pattern, repl in replacements.items():
                text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

            # --- Step 3: Add bullet formatting for readability
            sentences = re.split(r'(?<=[.!?])\s+', text)
            improved = "\n• " + "\n• ".join(sentences)

            # --- Step 4: Add a small “final polish” header
            st.subheader("📝 Enhanced Version:")
            st.success(improved)

            st.markdown("✅ *Your text was improved with action verbs, professional tone, and clean bullet points.*")

        else:
            st.warning("Please paste some text to improve.")

elif page == "📈 Skill Gap Dashboard":
    st.title("📈 Skill Gap Visualization")

    if not st.session_state.resume_text:
        st.warning("Please upload a resume in the Home section first.")
    else:
        job_desc = st.text_area("Paste Job Description", height=200)

        if st.button("Analyze Skill Gap"):
            jd_skills = extract_skills(job_desc, SKILLS)
            resume_skills = extract_skills(st.session_state.resume_text, SKILLS)

            have = sorted(list(set(jd_skills) & set(resume_skills)))
            missing = sorted(list(set(jd_skills) - set(resume_skills)))

            st.write(f"✅ Skills You Have: {', '.join(have) if have else 'None'}")
            st.write(f"⚠️ Missing Skills: {', '.join(missing) if missing else 'None'}")

            # Visualization
            import matplotlib.pyplot as plt

            jd_counts = [1 for s in jd_skills]
            res_counts = [1 if s in have else 0 for s in jd_skills]

            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(jd_skills, jd_counts, label='Required (JD)', alpha=0.4)
            ax.bar(jd_skills, res_counts, label='Present (Resume)', alpha=0.8)
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
elif page == "🎯 Career Recommendations":
    st.title("🎯 AI Career Recommendation System")

    if not st.session_state.resume_text:
        st.warning("Please upload a resume in Home section first.")
    else:
        skills_found = extract_skills(st.session_state.resume_text, SKILLS)
        st.write("🛠 Detected Skills:", ", ".join(skills_found))

        # Weighted mapping of careers
        career_map = {
            "Machine Learning Engineer / Data Scientist": ["python", "machine learning", "pandas", "tensorflow", "pytorch", "nlp"],
            "Backend Developer / Software Engineer": ["java", "spring", "sql", "api", "flask", "django"],
            "Frontend / Full Stack Developer": ["html", "css", "javascript", "react", "nodejs"],
            "DevOps / Cloud Engineer": ["aws", "docker", "kubernetes", "jenkins", "terraform"],
            "Cybersecurity Analyst": ["security", "firewall", "encryption", "network", "malware"]
        }

        # Count how many matching skills per career
        career_scores = {}
        for career, skillset in career_map.items():
            match_count = len(set(skills_found).intersection(set(skillset)))
            if match_count > 0:
                career_scores[career] = match_count

        # Sort by score
        if career_scores:
            sorted_careers = sorted(career_scores.items(), key=lambda x: x[1], reverse=True)
            top_careers = [c[0] for c in sorted_careers[:2]]  # top 2 roles
            st.subheader("👩‍💻 Recommended Career Paths:")
            for c in top_careers:
                st.write(f"• {c} ({career_scores[c]} relevant skills matched)")
        else:
            st.info("No clear career path detected. Try adding more technical keywords to your resume.")

elif page == "📝 Feedback":
    st.title("Feedback 📝")
    name = st.text_input("Your Name")
    rating = st.slider("Rate the app (1-5)", 1, 5, 3)
    comments = st.text_area("Comments")

    if st.button("Submit Feedback"):
        os.makedirs("data", exist_ok=True)
        fb_path = "data/feedback.csv"
        entry = pd.DataFrame([{
            "timestamp": datetime.datetime.now(),
            "name": name,
            "rating": rating,
            "comments": comments
        }])
        if os.path.exists(fb_path):
            entry.to_csv(fb_path, mode="a", header=False, index=False)
        else:
            entry.to_csv(fb_path, index=False)
        st.success("Thanks for your feedback!")

elif page == "👨‍💻 Admin":
    st.title("Admin Dashboard 👨‍💻")
    user = st.text_input("User ID")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == ADMIN_USER and pwd == ADMIN_PASS:
            st.success("✅ Login successful! Welcome Admin.")

            log_path = "data/analysis_log.csv"
            feedback_path = "data/feedback.csv"

            # --- Resume Analysis Logs ---
            if os.path.exists(log_path):
                st.subheader("📄 Resume Analysis Log")
                df_log = pd.read_csv(log_path)
                st.dataframe(df_log)

                # Plot distribution of predicted categories
                if "prediction" in df_log.columns:
                    st.subheader("📊 Category Distribution")
                    counts = df_log["prediction"].value_counts()

                    # Bar chart
                    st.bar_chart(counts)

                    # Pie chart
                    import matplotlib.pyplot as plt
                    fig1, ax1 = plt.subplots()
                    ax1.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
                    ax1.axis("equal")  # Equal aspect ratio
                    st.pyplot(fig1)

                # Plot skills frequency
                if "skills_found" in df_log.columns:
                    st.subheader("🛠 Top 10 Skills Frequency")
                    skills_series = df_log["skills_found"].dropna().str.split(", ").explode()
                    skill_counts = skills_series.value_counts().head(10)

                    # Bar chart
                    st.bar_chart(skill_counts)

                    # Pie chart
                    fig2, ax2 = plt.subplots()
                    ax2.pie(skill_counts, labels=skill_counts.index, autopct="%1.1f%%", startangle=90)
                    ax2.axis("equal")
                    st.pyplot(fig2)

            else:
                st.info("No resume logs found.")

            # --- Feedback Logs ---
            if os.path.exists(feedback_path):
                st.subheader("📝 Feedback Data")
                df_fb = pd.read_csv(feedback_path)
                st.dataframe(df_fb)

                if "rating" in df_fb.columns:
                    st.subheader("⭐ Feedback Ratings")
                    avg_rating = df_fb["rating"].mean()
                    st.metric("Average Rating", f"{avg_rating:.2f} / 5")

                    # Bar chart of ratings
                    st.bar_chart(df_fb["rating"].value_counts().sort_index())

                    # Pie chart of ratings
                    fig3, ax3 = plt.subplots()
                    rating_counts = df_fb["rating"].value_counts().sort_index()
                    ax3.pie(rating_counts, labels=rating_counts.index, autopct="%1.1f%%", startangle=90)
                    ax3.axis("equal")
                    st.pyplot(fig3)

            else:
                st.info("No feedback found.")
        else:
            st.error("❌ Invalid User ID or Password")

# ----------------------------
# NAVIGATION BUTTONS (BOTTOM)
# ----------------------------
st.markdown("---")
col_nav1, col_nav2 = st.columns([1,1])
with col_nav1:
    if st.button("⬅️ Back", key=f"back_{page}"):
        go_back()
with col_nav2:
    if st.button("Next ➡️", key=f"next_{page}"):
        
        go_next()

