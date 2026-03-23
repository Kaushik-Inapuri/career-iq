"""
CareerIQ — AI Career Intelligence Platform
===========================================
Full-stack implementation with:
  • Pandas + Scikit-learn  — TF-IDF vectorization & cosine similarity scoring
  • SQLite                 — user profile and session persistence
  • RAG                    — career knowledge base retrieval for AI context
  • pdfplumber             — resume parsing & NLP keyword extraction
  • Flask                  — web framework & API layer
  • Groq API               — LLM enrichment via 4-agent pipeline
"""

import json
import os
import re
import sqlite3
import datetime
import urllib.error
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import pdfplumber
from flask import Flask, request, jsonify, render_template, g

# ── Load .env for local dev ──
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

ENV_API_KEY  = os.environ.get("GROQ_API_KEY", "").strip()
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
DB_PATH      = os.path.join(BASE_DIR, "careeriq.db")
KB_PATH      = os.path.join(BASE_DIR, "career_knowledge.json")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# ═══════════════════════════════════════════════════════
#  KNOWLEDGE BASE — RAG
# ═══════════════════════════════════════════════════════

def load_knowledge_base():
    """Load career knowledge base JSON for RAG retrieval."""
    if not os.path.exists(KB_PATH):
        return {"careers": [], "market_insights": {}}
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

KB = load_knowledge_base()

# Build a Pandas DataFrame from the knowledge base for vectorized retrieval
def build_kb_dataframe():
    """Convert knowledge base to a Pandas DataFrame with combined text field for TF-IDF."""
    rows = []
    for career in KB.get("careers", []):
        combined_text = " ".join([
            career.get("title", ""),
            career.get("overview", ""),
            career.get("responsibilities", ""),
            career.get("required_skills", ""),
            career.get("soft_skills", ""),
            career.get("future_scope", ""),
        ])
        rows.append({
            "id":            career.get("id", ""),
            "title":         career.get("title", ""),
            "domain":        career.get("domain", ""),
            "combined_text": combined_text,
            "required_skills": career.get("required_skills", ""),
            "overview":      career.get("overview", ""),
            "learning_path": career.get("learning_path", ""),
            "certifications":career.get("certifications", ""),
            "top_companies": career.get("top_companies", ""),
            "salary_india":  career.get("salary_india", ""),
            "salary_global": career.get("salary_global", ""),
            "market_demand": career.get("market_demand", 70),
            "growth_outlook":career.get("growth_outlook", "Medium"),
        })
    return pd.DataFrame(rows)

KB_DF = build_kb_dataframe()

# Fit TF-IDF vectorizer on knowledge base
_tfidf_vectorizer = None
_kb_tfidf_matrix  = None

def get_tfidf():
    """Lazy-init TF-IDF vectorizer fitted on career knowledge base."""
    global _tfidf_vectorizer, _kb_tfidf_matrix
    if _tfidf_vectorizer is None and not KB_DF.empty:
        _tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        _kb_tfidf_matrix = _tfidf_vectorizer.fit_transform(KB_DF["combined_text"])
    return _tfidf_vectorizer, _kb_tfidf_matrix


def rag_retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    RAG retrieval: given a user query string, find the most relevant
    career documents from the knowledge base using TF-IDF cosine similarity.
    Returns a list of career dicts with their similarity scores.
    """
    vectorizer, kb_matrix = get_tfidf()
    if vectorizer is None or kb_matrix is None:
        return []

    query_vec   = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, kb_matrix).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.01:  # relevance threshold
            career_row = KB_DF.iloc[idx].to_dict()
            career_row["similarity_score"] = round(float(similarities[idx]), 4)
            results.append(career_row)
    return results


def build_rag_context(user_skills: list, domains: list, top_careers: list) -> str:
    """
    Build enriched RAG context for AI prompts by retrieving relevant
    career knowledge documents based on the user's profile.
    """
    if KB_DF.empty:
        return ""

    # Build query from user skills and top careers
    query = " ".join(user_skills[:15] + domains[:3] + top_careers[:3])
    retrieved = rag_retrieve(query, top_k=min(len(top_careers), 5))

    if not retrieved:
        return ""

    context_parts = ["=== RETRIEVED CAREER KNOWLEDGE (RAG) ==="]
    for doc in retrieved:
        context_parts.append(f"""
Career: {doc['title']}
Overview: {doc['overview']}
Required Skills: {doc['required_skills']}
Learning Path: {doc['learning_path']}
Top Companies: {doc['top_companies']}
Certifications: {doc['certifications']}
Similarity Score: {doc['similarity_score']}""")

    return "\n".join(context_parts)


# ═══════════════════════════════════════════════════════
#  SQLITE DATABASE
# ═══════════════════════════════════════════════════════

def get_db():
    """Get or create a thread-local database connection."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    """Create database tables if they don't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS assessments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            name        TEXT,
            age         TEXT,
            education   TEXT,
            experience  TEXT,
            location    TEXT,
            domains     TEXT,
            skills      TEXT,
            soft_skills TEXT,
            tools       TEXT,
            languages   TEXT,
            personality TEXT,
            work_mode   TEXT,
            goals       TEXT,
            salary_exp  TEXT,
            fit_score   INTEGER,
            top_career  TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS career_results (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id  INTEGER REFERENCES assessments(id),
            career_title   TEXT,
            match_percent  INTEGER,
            skill_score    INTEGER,
            domain_score   INTEGER,
            market_demand  INTEGER,
            salary_range   TEXT,
            is_it          INTEGER,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_assessments_session
            ON assessments(session_id);
        CREATE INDEX IF NOT EXISTS idx_assessments_created
            ON assessments(created_at);
        """)

def save_assessment(profile: dict, result: dict) -> int:
    """Persist a user assessment and results to SQLite. Returns assessment ID."""
    try:
        db = get_db()
        top_careers = result.get("topCareers", [])
        top_career  = top_careers[0]["title"] if top_careers else ""

        cur = db.execute("""
            INSERT INTO assessments
              (session_id, name, age, education, experience, location,
               domains, skills, soft_skills, tools, languages,
               personality, work_mode, goals, salary_exp, fit_score, top_career)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            profile.get("session_id", "anon"),
            profile.get("name", ""),
            profile.get("age", ""),
            profile.get("education", ""),
            profile.get("experience", ""),
            profile.get("location", ""),
            json.dumps(profile.get("domains", [])),
            json.dumps(profile.get("skills", [])),
            json.dumps(profile.get("softSkills", [])),
            json.dumps(profile.get("tools", [])),
            json.dumps(profile.get("languages", [])),
            json.dumps(profile.get("personality", [])),
            profile.get("workMode", ""),
            json.dumps(profile.get("goals", [])),
            profile.get("salary", ""),
            result.get("fitScore", 0),
            top_career,
        ))
        assessment_id = cur.lastrowid

        # Save individual career results
        for career in top_careers[:5]:
            db.execute("""
                INSERT INTO career_results
                  (assessment_id, career_title, match_percent, skill_score,
                   domain_score, market_demand, salary_range, is_it)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                assessment_id,
                career.get("title", ""),
                career.get("matchPercent", 0),
                career.get("skillScore", 0),
                career.get("domainScore", 0),
                career.get("marketDemand", 0),
                career.get("salaryRange", ""),
                1 if career.get("isIT") else 0,
            ))

        db.commit()
        return assessment_id
    except Exception as e:
        print(f"  DB save error: {e}")
        return -1

def get_analytics() -> dict:
    """
    Query SQLite using Pandas for aggregate analytics.
    Returns stats useful for dashboard insights.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Load into Pandas DataFrames
            assessments_df = pd.read_sql_query(
                "SELECT * FROM assessments ORDER BY created_at DESC LIMIT 500",
                conn
            )
            results_df = pd.read_sql_query(
                "SELECT * FROM career_results",
                conn
            )

        if assessments_df.empty:
            return {"total": 0}

        # Pandas aggregations
        avg_score   = round(float(assessments_df["fit_score"].mean()), 1)
        score_dist  = assessments_df["fit_score"].value_counts(bins=5).to_dict()
        top_careers = (
            results_df["career_title"].value_counts().head(5).to_dict()
            if not results_df.empty else {}
        )
        edu_dist    = assessments_df["education"].value_counts().head(6).to_dict()
        recent_names= assessments_df["name"].head(5).tolist()

        return {
            "total":        len(assessments_df),
            "avg_score":    avg_score,
            "top_careers":  top_careers,
            "edu_dist":     edu_dist,
            "recent_names": recent_names,
        }
    except Exception as e:
        return {"total": 0, "error": str(e)}


# ═══════════════════════════════════════════════════════
#  SKILL KNOWLEDGE BASE (hardcoded for scoring)
# ═══════════════════════════════════════════════════════

CAREER_SKILL_MAP = {
    "Machine Learning Engineer":      ["Python","Machine Learning","Deep Learning","TensorFlow","PyTorch","Docker","SQL","Statistics","NumPy","Scikit-learn"],
    "Data Scientist":                  ["Python","SQL","Statistics","Machine Learning","Data Visualization","Pandas","NumPy","Tableau","R","Excel"],
    "Full Stack Developer":            ["JavaScript","React","Node.js","SQL","Docker","HTML","CSS","REST APIs","TypeScript","Git"],
    "Backend Developer":               ["Python","Java","C++","C#","Node.js","SQL","REST APIs","Docker","Git","Linux"],
    "Software Engineer":               ["Python","Java","C","C++","C#","Data Structures & Algorithms","OOP","Git","Linux","SQL"],
    "DevOps Engineer":                 ["Docker","Kubernetes","AWS","Linux","CI/CD","Python","Terraform","Jenkins","Git","Bash Scripting"],
    "Cloud Architect":                 ["AWS","Azure","GCP","Docker","Kubernetes","Terraform","Networking","Linux","Python"],
    "Competitive Programmer":          ["C","C++","Java","Python","Data Structures & Algorithms","Problem Solving","Mathematics"],
    "UI/UX Designer":                  ["Figma","Adobe XD","User Research","Prototyping","CSS","HTML","Design Thinking","Canva"],
    "Data Analyst":                    ["SQL","Excel","Python","Tableau","Power BI","Statistics","Data Visualization","R"],
    "Product Manager":                 ["Project Management","Market Research","Agile","Communication","SQL","Analytics","Leadership"],
    "Cybersecurity Analyst":           ["Network Security","Ethical Hacking","Python","Linux","Firewalls","SIEM","C","Cryptography"],
    "Embedded Systems Engineer":       ["C","C++","Embedded Systems","Arduino","Raspberry Pi","IoT","Circuit Design","MATLAB"],
    "Mechanical Engineer":             ["AutoCAD","SolidWorks","MATLAB","Thermodynamics","CAD/CAM","Six Sigma"],
    "Civil Engineer":                  ["AutoCAD","Structural Analysis","STAAD Pro","Surveying","Construction Management"],
    "Financial Analyst":               ["Financial Modeling","Excel","Accounting","Valuation","SQL","Bloomberg"],
    "Marketing Manager":               ["Digital Marketing","SEO","Content Marketing","Analytics","Brand Management","CRM"],
    "Doctor / Physician":              ["Clinical Diagnosis","Patient Care","Pharmacology","Medical Research","Anatomy & Physiology","Biochemistry"],
    "HR Manager":                      ["HR Management","Talent Acquisition","Labor Law","Communication","Excel","HRMS"],
    "Content Creator":                 ["Content Writing","Video Editing","Social Media","SEO","Photography","Copywriting"],
    "Lawyer":                          ["Legal Research","Contract Drafting","Litigation","Communication","Critical Thinking"],
    "Teacher / Educator":              ["Teaching","Curriculum Development","Communication","Patience","Subject Expertise"],
    "Graphic Designer":                ["Graphic Design","Photoshop","Illustrator","Typography","Figma","Canva"],
    "Entrepreneur / Startup Founder":  ["Business Strategy","Marketing","Sales","Financial Analysis","Leadership","Product Management"],
}

DOMAIN_EXCLUSIVE_SKILLS = {
    "Doctor / Physician":   ["Clinical Diagnosis","Patient Care","Pharmacology","Anatomy & Physiology","Biochemistry","Medical Research","First Aid & CPR","Nursing Care"],
    "Lawyer":               ["Legal Research","Contract Drafting","Litigation","Corporate Law","Intellectual Property","Labor Law","Compliance"],
    "Civil Engineer":       ["AutoCAD","Structural Analysis","STAAD Pro","Surveying","Construction Management","Geotechnical Engineering"],
    "Mechanical Engineer":  ["AutoCAD","SolidWorks","CATIA","ANSYS","Thermodynamics","CAD/CAM","Six Sigma","CNC Machining"],
    "Graphic Designer":     ["Graphic Design","Photoshop","Illustrator","Typography","Canva","CorelDRAW","InDesign"],
    "HR Manager":           ["HR Management","Talent Acquisition","Payroll","HRMS","Labor Law","Learning & Development"],
    "Financial Analyst":    ["Financial Modeling","Accounting","Valuation","Bloomberg","Investment Analysis","Portfolio Management","Auditing"],
    "Marketing Manager":    ["Digital Marketing","SEO","Content Marketing","Brand Management","CRM","Social Media Marketing"],
    "Content Creator":      ["Content Writing","Video Editing","Copywriting","Photography","Podcast Production","Blogging"],
    "Teacher / Educator":   ["Teaching","Curriculum Development","E-learning Design","Tutoring","Special Education"],
}

MARKET_DEMAND = {
    "Machine Learning Engineer": 92, "Data Scientist": 89, "Full Stack Developer": 91,
    "Backend Developer": 90, "Software Engineer": 92, "DevOps Engineer": 88,
    "Cloud Architect": 85, "Competitive Programmer": 78, "UI/UX Designer": 80,
    "Data Analyst": 87, "Product Manager": 83, "Cybersecurity Analyst": 90,
    "Embedded Systems Engineer": 82, "Mechanical Engineer": 72, "Civil Engineer": 70,
    "Financial Analyst": 78, "Marketing Manager": 75, "Doctor / Physician": 88,
    "HR Manager": 71, "Content Creator": 74, "Lawyer": 73,
    "Teacher / Educator": 76, "Graphic Designer": 72, "Entrepreneur / Startup Founder": 80,
}

SALARY_RANGES = {
    "Machine Learning Engineer": "₹12–40 LPA", "Data Scientist": "₹10–35 LPA",
    "Full Stack Developer": "₹8–30 LPA", "Backend Developer": "₹8–28 LPA",
    "Software Engineer": "₹6–25 LPA", "DevOps Engineer": "₹10–32 LPA",
    "Cloud Architect": "₹15–45 LPA", "Competitive Programmer": "₹8–30 LPA",
    "UI/UX Designer": "₹6–22 LPA", "Data Analyst": "₹6–20 LPA",
    "Product Manager": "₹12–40 LPA", "Cybersecurity Analyst": "₹8–28 LPA",
    "Embedded Systems Engineer": "₹5–20 LPA", "Mechanical Engineer": "₹4–16 LPA",
    "Civil Engineer": "₹4–14 LPA", "Financial Analyst": "₹6–24 LPA",
    "Marketing Manager": "₹6–22 LPA", "Doctor / Physician": "₹8–60 LPA",
    "HR Manager": "₹5–18 LPA", "Content Creator": "₹3–15 LPA",
    "Lawyer": "₹5–30 LPA", "Teacher / Educator": "₹3–12 LPA",
    "Graphic Designer": "₹3–14 LPA", "Entrepreneur / Startup Founder": "Variable",
}

DOMAIN_MAP = {
    "Machine Learning Engineer":     ("Technology & IT", True),
    "Data Scientist":                 ("Technology & IT", True),
    "Full Stack Developer":           ("Technology & IT", True),
    "Backend Developer":              ("Technology & IT", True),
    "Software Engineer":              ("Technology & IT", True),
    "DevOps Engineer":                ("Technology & IT", True),
    "Cloud Architect":                ("Technology & IT", True),
    "Competitive Programmer":         ("Technology & IT", True),
    "UI/UX Designer":                 ("Creative & Design", True),
    "Data Analyst":                   ("Technology & IT", True),
    "Product Manager":                ("Business & Management", False),
    "Cybersecurity Analyst":          ("Technology & IT", True),
    "Embedded Systems Engineer":      ("Engineering & Manufacturing", True),
    "Mechanical Engineer":            ("Engineering & Manufacturing", False),
    "Civil Engineer":                 ("Engineering & Manufacturing", False),
    "Financial Analyst":              ("Finance & Economics", False),
    "Marketing Manager":              ("Business & Management", False),
    "Doctor / Physician":             ("Healthcare & Medicine", False),
    "HR Manager":                     ("Business & Management", False),
    "Content Creator":                ("Media & Communications", False),
    "Lawyer":                         ("Law & Public Services", False),
    "Teacher / Educator":             ("Education & Training", False),
    "Graphic Designer":               ("Creative & Design", False),
    "Entrepreneur / Startup Founder": ("Business & Management", False),
}


# ═══════════════════════════════════════════════════════
#  ML SCORING ENGINE (Scikit-learn + Pandas)
# ═══════════════════════════════════════════════════════

def skill_matches(user_skill: str, career_skill: str) -> bool:
    """Word-boundary skill matching to prevent false positives."""
    u = user_skill.lower().strip()
    c = career_skill.lower().strip()
    if u == c:
        return True
    if len(u) >= 3:
        if re.search(r'(?<![a-z])' + re.escape(u) + r'(?![a-z])', c):
            return True
        if re.search(r'(?<![a-z])' + re.escape(c) + r'(?![a-z])', u):
            return True
    return False


def compute_ml_scores(user_skills: list, domains: list, personality: list, goals: list) -> pd.DataFrame:
    """
    Scikit-learn + Pandas scoring pipeline.

    Uses TF-IDF cosine similarity between the user's skill vector and each
    career's required skill text, combined with rule-based domain, personality,
    and market demand components. Returns a sorted Pandas DataFrame.
    """
    vectorizer, kb_matrix = get_tfidf()

    # ── Build user skill query string for TF-IDF
    user_skill_text = " ".join(user_skills)

    rows = []
    for career_title, career_skills in CAREER_SKILL_MAP.items():
        # ── 1. TF-IDF cosine similarity (if KB available) ──────────────────
        tfidf_sim = 0.0
        if vectorizer is not None and not KB_DF.empty:
            try:
                career_idx = KB_DF[KB_DF["title"] == career_title].index
                if len(career_idx) > 0:
                    user_vec  = vectorizer.transform([user_skill_text])
                    car_vec   = kb_matrix[career_idx[0]]
                    tfidf_sim = float(cosine_similarity(user_vec, car_vec)[0][0])
            except Exception:
                tfidf_sim = 0.0

        # ── 2. Exact + partial skill overlap (rule-based) ──────────────────
        matched = 0
        for cs in career_skills:
            for us in user_skills:
                if us.lower().strip() == cs.lower().strip():
                    matched += 1.0; break
                elif skill_matches(us, cs):
                    matched += 0.6; break
        skill_overlap = min(matched / max(len(career_skills), 1), 1.0)

        # ── 3. Combined skill score (TF-IDF + overlap) ─────────────────────
        skill_score = (tfidf_sim * 0.4 + skill_overlap * 0.6) * 100

        # ── 4. Hard disqualification for domain-exclusive careers ──────────
        exclusive = DOMAIN_EXCLUSIVE_SKILLS.get(career_title, [])
        if exclusive:
            has_any = any(
                skill_matches(us, ex) or us.lower() == ex.lower()
                for ex in exclusive for us in user_skills
            )
            if not has_any and skill_score < 25:
                skill_score = max(skill_score * 0.3, 5)

        # ── 5. Domain interest score ───────────────────────────────────────
        career_domain, _ = DOMAIN_MAP.get(career_title, ("", False))
        domain_score = 85 if career_domain in domains else (60 if "Not Sure — Explore All" in domains else 20)

        # ── 6. Personality fit ─────────────────────────────────────────────
        personality_map = {
            "Machine Learning Engineer":      ["Analytical thinker","Data-driven","Detail-oriented","Independent worker"],
            "Data Scientist":                  ["Analytical thinker","Data-driven","Creative problem solver"],
            "Full Stack Developer":            ["Detail-oriented","Independent worker","Creative problem solver"],
            "Backend Developer":               ["Analytical thinker","Detail-oriented","Independent worker"],
            "Software Engineer":               ["Analytical thinker","Detail-oriented","Methodical planner"],
            "DevOps Engineer":                 ["Methodical planner","Detail-oriented","Analytical thinker"],
            "Competitive Programmer":          ["Analytical thinker","Data-driven","Detail-oriented","Methodical planner"],
            "Cybersecurity Analyst":           ["Analytical thinker","Detail-oriented","Risk-taker"],
            "Embedded Systems Engineer":       ["Detail-oriented","Methodical planner","Analytical thinker"],
            "Cloud Architect":                 ["Big-picture visionary","Methodical planner","Analytical thinker"],
            "UI/UX Designer":                  ["Creative problem solver","People-oriented","Detail-oriented"],
            "Product Manager":                 ["Natural leader","Big-picture visionary","People-oriented"],
            "Doctor / Physician":              ["Empathetic listener","Detail-oriented","Methodical planner"],
            "Lawyer":                          ["Analytical thinker","Detail-oriented","Methodical planner"],
            "Teacher / Educator":              ["Empathetic listener","People-oriented"],
            "Entrepreneur / Startup Founder":  ["Risk-taker","Big-picture visionary","Natural leader"],
            "Marketing Manager":               ["Creative problem solver","People-oriented","Big-picture visionary"],
            "Financial Analyst":               ["Analytical thinker","Detail-oriented","Data-driven"],
            "HR Manager":                      ["Empathetic listener","People-oriented","Team collaborator"],
            "Content Creator":                 ["Creative problem solver","Independent worker","Big-picture visionary"],
            "Graphic Designer":                ["Creative problem solver","Detail-oriented","Independent worker"],
        }
        ideal = personality_map.get(career_title, [])
        p_hit = sum(1 for p in personality if p in ideal)
        personality_score = min(p_hit / max(len(ideal), 1), 1.0) * 100 if ideal else 50

        # ── 7. Market demand ───────────────────────────────────────────────
        demand = MARKET_DEMAND.get(career_title, 70)

        # ── 8. Weighted final score ────────────────────────────────────────
        final = (
            skill_score      * 0.45 +
            domain_score     * 0.20 +
            personality_score* 0.15 +
            demand           * 0.20
        )

        domain, is_it = DOMAIN_MAP.get(career_title, ("General", False))
        rows.append({
            "title":            career_title,
            "algoScore":        round(min(final, 99)),
            "skillScore":       round(skill_score),
            "domainScore":      round(domain_score),
            "personalityScore": round(personality_score),
            "marketDemand":     demand,
            "domain":           domain,
            "isIT":             is_it,
            "salaryRange":      SALARY_RANGES.get(career_title, "—"),
            "tfidfSimilarity":  round(tfidf_sim, 4),
        })

    # ── Build Pandas DataFrame and sort ──────────────────────────────────
    df = pd.DataFrame(rows)
    df = df.sort_values("algoScore", ascending=False).reset_index(drop=True)
    return df


def get_skill_gaps(user_skills: list, career_title: str) -> list:
    return [s for s in CAREER_SKILL_MAP.get(career_title, [])
            if not any(us.lower().strip() == s.lower().strip() or skill_matches(us, s)
                       for us in user_skills)][:5]

def get_matched_skills(user_skills: list, career_title: str) -> list:
    return [s for s in CAREER_SKILL_MAP.get(career_title, [])
            if any(us.lower().strip() == s.lower().strip() or skill_matches(us, s)
                   for us in user_skills)]


# ═══════════════════════════════════════════════════════
#  NLP — RESUME PARSER
# ═══════════════════════════════════════════════════════

ALL_SKILLS = list(set(list(CAREER_SKILL_MAP.keys()) + [
    "Python","Java","C++","C#","JavaScript","TypeScript","Go","Rust","PHP","Ruby","Swift","Kotlin",
    "HTML","CSS","React","Angular","Vue.js","Node.js","Django","Flask","FastAPI","Spring Boot",
    "Machine Learning","Deep Learning","NLP","Computer Vision","TensorFlow","PyTorch","Scikit-learn",
    "SQL","MySQL","PostgreSQL","MongoDB","Redis","Cassandra","Oracle","SQLite",
    "Docker","Kubernetes","AWS","Azure","GCP","Terraform","Jenkins","Git","Linux",
    "Pandas","NumPy","Matplotlib","Seaborn","Tableau","Power BI","Excel","MATLAB",
    "Figma","Photoshop","Illustrator","Adobe XD","Canva","AutoCAD","SolidWorks","ANSYS",
    "Accounting","Financial Analysis","Marketing","Sales","Project Management","HR Management",
    "Content Writing","Video Editing","Graphic Design","Photography","SEO","Social Media",
    "Clinical Diagnosis","Patient Care","Pharmacology","Legal Research","Teaching",
    "Communication","Leadership","Teamwork","Problem Solving","Critical Thinking",
    "Data Analysis","Statistics","Research","Agile","Scrum","REST APIs","GraphQL",
]))

# NLP: Build a Pandas Series for fast vectorized skill matching
_skills_series = pd.Series([s.lower() for s in ALL_SKILLS], name="skill")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes); tmp_path = tmp.name
    text = ""
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: text += t + "\n"
    finally:
        os.unlink(tmp_path)
    return text


def parse_resume_nlp(text: str) -> dict:
    """
    NLP-based resume parsing using:
    - Regex for structured fields (email, phone, dates)
    - Pandas vectorized string matching for skills
    - Keyword extraction for education level detection
    """
    text_lower = text.lower()

    # ── Pandas vectorized skill matching (NLP keyword extraction)
    found_skills = [
        ALL_SKILLS[i] for i, s in enumerate(_skills_series)
        if s in text_lower and len(s) > 2
    ]
    found_skills = list(set(found_skills))

    # ── Regex NLP extraction ──────────────────────────────────────────────
    lines  = [l.strip() for l in text.split("\n") if l.strip()]
    name   = lines[0] if lines else "Candidate"

    email_m = re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', text)
    email   = email_m.group() if email_m else ""

    phone_m = re.search(r'[\+]?[\d\s\-\(\)]{10,15}', text)
    phone   = phone_m.group().strip() if phone_m else ""

    # ── Education level detection via NLP keyword presence ───────────────
    education = "Not specified"
    edu_keywords = [
        ("PhD", ["phd","p.h.d","doctorate","doctoral"]),
        ("MBA / MA / M.Com", ["mba","m.b.a","master of business"]),
        ("M.Tech / ME / MCA", ["m.tech","m.e.","mca","m.sc","m.c.a","masters"]),
        ("MBBS / MD", ["mbbs","m.d.","m.b.b.s"]),
        ("B.Tech / B.E.", ["b.tech","b.e.","bachelor of technology","bachelor of engineering"]),
        ("BCA / MCA", ["bca","b.c.a"]),
        ("B.Sc / B.Com / B.A.", ["b.sc","b.com","b.a.","bachelor"]),
        ("12th / Intermediate", ["12th","intermediate","hsc","higher secondary"]),
        ("10th / High School", ["10th","high school","ssc","matriculation"]),
    ]
    for level, keywords in edu_keywords:
        if any(kw in text_lower for kw in keywords):
            education = level; break

    # ── Experience years detection ────────────────────────────────────────
    experience = ""
    exp_m = re.search(r'(\d+)\+?\s*years?\s*(of\s*)?(experience|exp)', text_lower)
    if exp_m:
        yrs = int(exp_m.group(1))
        if yrs == 0:    experience = "Student / Fresher (0 years)"
        elif yrs == 1:  experience = "Less than 1 year"
        elif yrs <= 2:  experience = "1–2 years"
        elif yrs <= 5:  experience = "3–5 years"
        elif yrs <= 10: experience = "5–10 years"
        else:           experience = "10+ years"

    return {
        "name":       name,
        "email":      email,
        "phone":      phone.strip(),
        "education":  education,
        "experience": experience,
        "skills":     found_skills,
        "raw_text":   text[:2000],
        "skill_count":len(found_skills),
    }


# ═══════════════════════════════════════════════════════
#  GROQ API
# ═══════════════════════════════════════════════════════

def call_groq(api_key: str, model: str, messages: list, json_mode: bool = True) -> dict:
    import requests as req_lib
    payload = {"model": model, "max_tokens": 4000, "temperature": 0.7, "messages": messages}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    resp = req_lib.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}",
                 "User-Agent": "Mozilla/5.0 (compatible; CareerIQ/1.0)"},
        json=payload, timeout=60,
    )
    if not resp.ok:
        raise urllib.error.HTTPError(url=resp.url, code=resp.status_code,
                                     msg=resp.text, hdrs=None, fp=None)
    raw = resp.json()["choices"][0]["message"]["content"]
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw.strip()).strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    return json.loads(m.group() if m else raw)


# ═══════════════════════════════════════════════════════
#  4-AGENT PIPELINE
# ═══════════════════════════════════════════════════════

def run_agents(profile: dict, api_key: str, model: str) -> dict:
    """
    4-Agent pipeline:
    Agent 1 — ML Scoring (Pandas + Scikit-learn TF-IDF cosine similarity)
    Agent 2 — AI Enrichment (Groq LLM + RAG context injection)
    Agent 3 — Skill Gap Analysis (Groq LLM)
    Agent 4 — Result Assembly & DB persistence
    """
    user_skills = profile.get("skills", [])
    domains     = profile.get("domains", [])
    personality = profile.get("personality", [])
    goals       = profile.get("goals", [])

    # ── AGENT 1: Scikit-learn + Pandas ML scoring ──────────────────────
    scores_df = compute_ml_scores(user_skills, domains, personality, goals)
    top5_df   = scores_df.head(5)
    top5_titles = top5_df["title"].tolist()

    # Build top5_algo list for merging later
    top5_algo = []
    for _, row in top5_df.iterrows():
        title = row["title"]
        top5_algo.append({
            "title":            title,
            "algoScore":        int(row["algoScore"]),
            "domain":           row["domain"],
            "isIT":             bool(row["isIT"]),
            "matchedSkills":    get_matched_skills(user_skills, title),
            "gapSkills":        get_skill_gaps(user_skills, title),
            "salaryRange":      row["salaryRange"],
            "marketDemand":     int(row["marketDemand"]),
            "tfidfSimilarity":  float(row["tfidfSimilarity"]),
        })

    # ── AGENT 2: RAG retrieval + AI enrichment ──────────────────────────
    rag_context = build_rag_context(user_skills, domains, top5_titles)

    agent2_prompt = f"""You are a Career Intelligence Agent with access to a retrieved knowledge base.
Use the RAG context below to give highly specific, grounded career advice.

USER PROFILE:
Name: {profile.get('name','')}, Age: {profile.get('age','')}, Education: {profile.get('education','')}
Skills: {', '.join(user_skills)}
Soft Skills: {', '.join(profile.get('softSkills',[]))}
Personality: {', '.join(personality)}
Goals: {', '.join(goals)}
Preferred Domains: {', '.join(domains)}
Work Mode: {profile.get('workMode','')}
Salary Expectation: {profile.get('salary','')}

TOP 5 CAREERS (ML-scored): {json.dumps(top5_titles)}

{rag_context}

Return ONLY valid JSON:
{{
  "overallSummary": "3 sentence personalized summary referencing their actual skills and the retrieved knowledge",
  "personalityInsight": "2 sentences about personality-career fit",
  "strengthsIdentified": ["s1","s2","s3","s4"],
  "immediateActions": ["action1","action2","action3"],
  "alternativeCareers": ["c1","c2","c3"],
  "careers": [
    {{
      "title": "exact title from list",
      "whyFit": "2 sentences referencing their specific skills and retrieved knowledge",
      "dailyWork": "1 sentence typical day",
      "growthOutlook": "High|Medium|Stable",
      "roadmap": ["Step 1: ...","Step 2: ...","Step 3: ...","Step 4: ...","Step 5: ...","Step 6: ..."],
      "courses": ["Course - Platform","Course - Platform","Course - Platform"],
      "certifications": ["Cert 1","Cert 2"],
      "jobRoles": ["Role A","Role B","Role C"],
      "companies": ["Company A","Company B","Company C","Company D"],
      "portfolioProjects": ["Project 1","Project 2","Project 3"]
    }}
  ]
}}"""

    ai_result = call_groq(api_key, model, [
        {"role": "system", "content": "You are an expert career guidance AI with access to a career knowledge base. Respond ONLY with valid JSON."},
        {"role": "user",   "content": agent2_prompt},
    ])

    # ── AGENT 3: Skill Gap Analysis ─────────────────────────────────────
    gap_result = call_groq(api_key, model, [
        {"role": "system", "content": "You are a skill gap analysis expert. Respond ONLY with valid JSON."},
        {"role": "user", "content": f"""Analyze skill gaps for this user targeting {top5_titles[0]}.
User Skills: {', '.join(user_skills)}
Target Career: {top5_titles[0]}
Return ONLY valid JSON:
{{
  "skillMatchPercent": 75,
  "userSkillStrengths": [{{"skill":"Python","level":85}},{{"skill":"SQL","level":70}}],
  "criticalGaps": [{{"skill":"Docker","demand":88,"priority":"High"}},{{"skill":"Kubernetes","demand":82,"priority":"High"}}],
  "learningPath": [{{"week":"Week 1-2","focus":"...","resource":"..."}},{{"week":"Week 3-4","focus":"...","resource":"..."}}],
  "marketTrends": [{{"skill":"GenAI","growth":"+145% YoY"}},{{"skill":"LLMs","growth":"+120% YoY"}}]
}}"""},
    ])

    # ── AGENT 4: Merge & assemble ────────────────────────────────────────
    ai_careers  = {c["title"]: c for c in ai_result.get("careers", [])}
    top5_merged = []
    for ac in top5_algo:
        merged = {**ac, **ai_careers.get(ac["title"], {})}
        merged["matchPercent"]     = ac["algoScore"]
        merged["skillScore"]       = round(len(ac["matchedSkills"]) / max(len(CAREER_SKILL_MAP.get(ac["title"], ["x"])), 1) * 100)
        merged["domainScore"]      = 85 if ac["domain"] in domains else 45
        merged["personalityScore"] = round(merged["matchPercent"] * 0.9)
        merged["marketScore"]      = ac["marketDemand"]
        top5_merged.append(merged)

    result = {
        "fitScore":            top5_merged[0]["matchPercent"] if top5_merged else 0,
        "overallSummary":      ai_result.get("overallSummary", ""),
        "personalityInsight":  ai_result.get("personalityInsight", ""),
        "strengthsIdentified": ai_result.get("strengthsIdentified", []),
        "immediateActions":    ai_result.get("immediateActions", []),
        "alternativeCareers":  ai_result.get("alternativeCareers", []),
        "topCareers":          top5_merged,
        "skillGapAnalysis":    gap_result,
        "allScored":           scores_df.head(24).to_dict(orient="records"),
        "ragUsed":             len(rag_retrieve(" ".join(user_skills[:5]), top_k=1)) > 0,
    }

    # Save to SQLite (Agent 4 persistence step)
    save_assessment(profile, result)

    return result


# ═══════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════

@app.route("/")
def index():
    if not os.path.exists(os.path.join(TEMPLATE_DIR, "index.html")):
        return "<h2>index.html not found — put it in the templates/ folder</h2>", 404
    return render_template("index.html", key_configured=bool(ENV_API_KEY))


@app.route("/api/key-status")
def key_status():
    return jsonify({"configured": bool(ENV_API_KEY)})


@app.route("/api/parse-resume", methods=["POST"])
def parse_resume_route():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["resume"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files supported"}), 400
    text = extract_text_from_pdf(f.read())
    if not text.strip():
        return jsonify({"error": "Could not extract text from PDF"}), 400
    return jsonify(parse_resume_nlp(text))


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data    = request.get_json()
    if not data: return jsonify({"error": "No data provided"}), 400
    api_key = ENV_API_KEY or data.get("apiKey", "").strip()
    model   = data.get("model", "llama-3.3-70b-versatile")
    profile = data.get("profile", {})
    if not api_key:                return jsonify({"error": "No API key. Set GROQ_API_KEY in environment."}), 400
    if not api_key.startswith("gsk_"): return jsonify({"error": "Invalid Groq API key."}), 400
    if not profile.get("skills"):  return jsonify({"error": "Skills are required."}), 400
    try:
        return jsonify(run_agents(profile, api_key, model))
    except urllib.error.HTTPError as e:
        codes = {429: "Rate limit hit. Wait 30s and retry.",
                 401: "Invalid API key.", 403: "Access denied. Get a fresh key at console.groq.com/keys"}
        return jsonify({"error": codes.get(e.code, f"Groq error {e.code}")}), e.code
    except Exception as e:
        err = str(e)
        for code, msg in [("429","Rate limit hit."),("401","Invalid API key."),("403","Access denied.")]:
            if code in err: return jsonify({"error": msg}), int(code)
        return jsonify({"error": err}), 500


@app.route("/api/score-preview", methods=["POST"])
def score_preview():
    data        = request.get_json()
    skills      = data.get("skills", [])
    domains     = data.get("domains", [])
    personality = data.get("personality", [])
    goals       = data.get("goals", [])
    df = compute_ml_scores(skills, domains, personality, goals)
    return jsonify(df.head(8)[["title","algoScore","domain","isIT"]].rename(
        columns={"algoScore":"score"}).to_dict(orient="records"))


@app.route("/api/chat", methods=["POST"])
def chat():
    data    = request.get_json()
    if not data: return jsonify({"error": "No data"}), 400
    api_key  = ENV_API_KEY or data.get("apiKey", "").strip()
    model    = data.get("model", "llama-3.3-70b-versatile")
    messages = data.get("messages", [])
    if not api_key:  return jsonify({"error": "No API key configured."}), 400
    if not messages: return jsonify({"error": "No messages provided."}), 400
    try:
        import requests as req_lib
        resp = req_lib.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}",
                     "User-Agent": "Mozilla/5.0 (compatible; CareerIQ/1.0)"},
            json={"model": model, "messages": messages,
                  "temperature": 0.75, "max_tokens": 512},
            timeout=30,
        )
        if not resp.ok:
            return jsonify({"error": f"Groq error {resp.status_code}"}), resp.status_code
        return jsonify({"reply": resp.json()["choices"][0]["message"]["content"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analytics")
def analytics():
    """Returns aggregate stats from SQLite via Pandas for dashboard use."""
    return jsonify(get_analytics())


@app.route("/api/rag-test")
def rag_test():
    """Debug endpoint — test RAG retrieval with a query."""
    query = request.args.get("q", "machine learning python")
    results = rag_retrieve(query, top_k=3)
    return jsonify({
        "query":    query,
        "results":  [{"title": r["title"], "similarity": r["similarity_score"]} for r in results],
        "kb_size":  len(KB_DF),
    })


# ═══════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    print()
    print("=" * 62)
    print("   CareerIQ  —  AI Career Intelligence Platform")
    print(f"   http://localhost:{port}")
    print("=" * 62)
    print(f"   Groq Key   : {'SET ✓' if ENV_API_KEY else 'NOT SET — edit .env or set GROQ_API_KEY'}")
    print(f"   Database   : {DB_PATH}")
    print(f"   Knowledge  : {len(KB_DF)} careers in RAG knowledge base")
    print(f"   ML Engine  : Pandas {pd.__version__} + Scikit-learn {__import__('sklearn').__version__}")
    print()
    print("   Technologies active:")
    print("   ✓  Pandas + Scikit-learn  — TF-IDF cosine similarity scoring")
    print("   ✓  SQLite                 — user profile & result persistence")
    print("   ✓  RAG                    — career knowledge base retrieval")
    print("   ✓  NLP                    — resume parsing & keyword extraction")
    print("   ✓  Groq API               — 4-agent LLM pipeline")
    print("   ✓  Chart.js               — analytics visualizations")
    print("=" * 62)
    print()
    app.run(host="0.0.0.0", port=port, debug=False)