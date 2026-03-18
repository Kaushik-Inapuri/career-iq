"""
CareerIQ — AI Career Intelligence Platform
Flask Backend
"""

import json
import os
import re
import math
import urllib.request
import urllib.error
from flask import Flask, request, jsonify, render_template, Response
import pdfplumber
import tempfile

# ── Load .env file if it exists (python-dotenv) ──
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass  # dotenv not installed — key must come from frontend or system env

# ── Read API key from environment (optional) ──
# If set, the frontend input box is hidden and this key is used automatically.
# If not set, the user must paste their key in the browser.
ENV_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

# Always find templates relative to THIS file, not the working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload

# ─────────────────────────────────────────
#  SKILL KNOWLEDGE BASE
# ─────────────────────────────────────────

SKILL_DOMAINS = {
    "Machine Learning": "Data & AI", "Deep Learning": "Data & AI", "Python": "Programming",
    "TensorFlow": "Data & AI", "PyTorch": "Data & AI", "Data Science": "Data & AI",
    "NLP": "Data & AI", "Computer Vision": "Data & AI", "Scikit-learn": "Data & AI",
    "Java": "Programming", "C++": "Programming", "JavaScript": "Programming",
    "TypeScript": "Programming", "Go": "Programming", "Rust": "Programming",
    "React": "Web", "Node.js": "Web", "Django": "Web", "Flask": "Web",
    "Docker": "DevOps", "Kubernetes": "DevOps", "AWS": "Cloud", "Azure": "Cloud",
    "SQL": "Data", "PostgreSQL": "Data", "MongoDB": "Data", "Redis": "Data",
    "Figma": "Design", "Photoshop": "Design", "Illustrator": "Design",
    "AutoCAD": "Engineering", "SolidWorks": "Engineering", "MATLAB": "Engineering",
    "Accounting": "Finance", "Financial Analysis": "Finance", "Excel": "Business",
    "Marketing": "Business", "Sales": "Business", "Project Management": "Business",
    "Content Writing": "Creative", "Graphic Design": "Creative", "Video Editing": "Creative",
    "Clinical Diagnosis": "Medical", "Patient Care": "Medical", "Pharmacology": "Medical",
    "Legal Research": "Law", "Contract Drafting": "Law", "Litigation": "Law",
    "Teaching": "Education", "Curriculum Development": "Education",
}

CAREER_SKILL_MAP = {
    "Machine Learning Engineer":     ["Python","Machine Learning","Deep Learning","TensorFlow","PyTorch","Docker","SQL","Statistics","NumPy","Scikit-learn"],
    "Data Scientist":                 ["Python","SQL","Statistics","Machine Learning","Data Visualization","Pandas","NumPy","Tableau","R","Excel"],
    "Full Stack Developer":           ["JavaScript","React","Node.js","SQL","Docker","HTML","CSS","REST APIs","TypeScript","Git"],
    "Backend Developer":              ["Python","Java","C++","C#","Node.js","SQL","REST APIs","Docker","Git","Linux"],
    "Software Engineer":              ["Python","Java","C","C++","C#","Data Structures & Algorithms","OOP","Git","Linux","SQL"],
    "DevOps Engineer":                ["Docker","Kubernetes","AWS","Linux","CI/CD","Python","Terraform","Jenkins","Git","Bash Scripting"],
    "Cloud Architect":                ["AWS","Azure","GCP","Docker","Kubernetes","Terraform","Networking","Linux","Python"],
    "Competitive Programmer":         ["C","C++","Java","Python","Data Structures & Algorithms","Problem Solving","Mathematics"],
    "UI/UX Designer":                 ["Figma","Adobe XD","User Research","Prototyping","CSS","HTML","Design Thinking","Canva"],
    "Data Analyst":                   ["SQL","Excel","Python","Tableau","Power BI","Statistics","Data Visualization","R"],
    "Product Manager":                ["Project Management","Market Research","Agile","Communication","SQL","Analytics","Leadership"],
    "Cybersecurity Analyst":          ["Network Security","Ethical Hacking","Python","Linux","Firewalls","SIEM","C","Cryptography"],
    "Embedded Systems Engineer":      ["C","C++","Embedded Systems","Arduino","Raspberry Pi","IoT","Circuit Design","MATLAB"],
    "Mechanical Engineer":            ["AutoCAD","SolidWorks","MATLAB","Thermodynamics","CAD/CAM","Six Sigma"],
    "Civil Engineer":                 ["AutoCAD","Structural Analysis","STAAD Pro","Surveying","Construction Management"],
    "Financial Analyst":              ["Financial Modeling","Excel","Accounting","Valuation","SQL","Bloomberg"],
    "Marketing Manager":              ["Digital Marketing","SEO","Content Marketing","Analytics","Brand Management","CRM"],
    "Doctor / Physician":             ["Clinical Diagnosis","Patient Care","Pharmacology","Medical Research","Anatomy & Physiology","Biochemistry"],
    "HR Manager":                     ["HR Management","Talent Acquisition","Labor Law","Communication","Excel","HRMS"],
    "Content Creator":                ["Content Writing","Video Editing","Social Media","SEO","Photography","Copywriting"],
    "Lawyer":                         ["Legal Research","Contract Drafting","Litigation","Communication","Critical Thinking"],
    "Teacher / Educator":             ["Teaching","Curriculum Development","Communication","Patience","Subject Expertise"],
    "Graphic Designer":               ["Graphic Design","Photoshop","Illustrator","Typography","Figma","Canva"],
    "Entrepreneur / Startup Founder": ["Business Strategy","Marketing","Sales","Financial Analysis","Leadership","Product Management"],
}

# Skills that are EXCLUSIVELY associated with certain domain families.
# If a user has NONE of these domain-exclusive skills, that career is heavily penalised.
DOMAIN_EXCLUSIVE_SKILLS = {
    "Doctor / Physician":   ["Clinical Diagnosis","Patient Care","Pharmacology","Anatomy & Physiology","Biochemistry","Medical Research","First Aid & CPR","Nursing Care","Radiology","Surgery"],
    "Lawyer":               ["Legal Research","Contract Drafting","Litigation","Corporate Law","Intellectual Property","Labor Law","Compliance","GDPR"],
    "Civil Engineer":       ["AutoCAD","Structural Analysis","STAAD Pro","Surveying","Construction Management","Geotechnical Engineering","Highway Engineering"],
    "Mechanical Engineer":  ["AutoCAD","SolidWorks","CATIA","ANSYS","Thermodynamics","CAD/CAM","Six Sigma","CNC Machining","3D Printing"],
    "Graphic Designer":     ["Graphic Design","Photoshop","Illustrator","Typography","Canva","CorelDRAW","InDesign","Figma"],
    "HR Manager":           ["HR Management","Talent Acquisition","Payroll","HRMS","Labor Law","Learning & Development"],
    "Financial Analyst":    ["Financial Modeling","Accounting","Valuation","Bloomberg","Investment Analysis","Portfolio Management","Auditing"],
    "Marketing Manager":    ["Digital Marketing","SEO","Content Marketing","Brand Management","CRM","Social Media Marketing","Google Ads"],
    "Content Creator":      ["Content Writing","Video Editing","Copywriting","Photography","Podcast Production","Blogging","Scriptwriting"],
    "Teacher / Educator":   ["Teaching","Curriculum Development","E-learning Design","Tutoring","Special Education"],
}

MARKET_DEMAND = {
    "Machine Learning Engineer": 92, "Data Scientist": 89, "Full Stack Developer": 91,
    "Backend Developer": 90, "Software Engineer": 92, "DevOps Engineer": 88,
    "Cloud Architect": 85, "Competitive Programmer": 78,
    "UI/UX Designer": 80, "Data Analyst": 87, "Product Manager": 83,
    "Cybersecurity Analyst": 90, "Embedded Systems Engineer": 82,
    "Mechanical Engineer": 72, "Civil Engineer": 70, "Financial Analyst": 78,
    "Marketing Manager": 75, "Doctor / Physician": 88, "HR Manager": 71,
    "Content Creator": 74, "Lawyer": 73, "Teacher / Educator": 76,
    "Graphic Designer": 72, "Entrepreneur / Startup Founder": 80,
}

SALARY_RANGES = {
    "Machine Learning Engineer":  "₹12–40 LPA", "Data Scientist": "₹10–35 LPA",
    "Full Stack Developer":        "₹8–30 LPA",  "Backend Developer": "₹8–28 LPA",
    "Software Engineer":           "₹6–25 LPA",  "DevOps Engineer": "₹10–32 LPA",
    "Cloud Architect":             "₹15–45 LPA", "Competitive Programmer": "₹8–30 LPA",
    "UI/UX Designer":              "₹6–22 LPA",  "Data Analyst": "₹6–20 LPA",
    "Product Manager":             "₹12–40 LPA", "Cybersecurity Analyst": "₹8–28 LPA",
    "Embedded Systems Engineer":   "₹5–20 LPA",  "Mechanical Engineer": "₹4–16 LPA",
    "Civil Engineer":              "₹4–14 LPA",  "Financial Analyst": "₹6–24 LPA",
    "Marketing Manager":           "₹6–22 LPA",  "Doctor / Physician": "₹8–60 LPA",
    "HR Manager":                  "₹5–18 LPA",  "Content Creator": "₹3–15 LPA",
    "Lawyer":                      "₹5–30 LPA",  "Teacher / Educator": "₹3–12 LPA",
    "Graphic Designer":            "₹3–14 LPA",  "Entrepreneur / Startup Founder": "Variable",
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

# ─────────────────────────────────────────
#  SCORING ALGORITHM
# ─────────────────────────────────────────

def skill_matches(user_skill, career_skill):
    """
    Smart skill matching — avoids false positives like 'C' matching 'Clinical Diagnosis'.
    Requires whole-word or full-string match.
    """
    u = user_skill.lower().strip()
    c = career_skill.lower().strip()
    if u == c:
        return True
    # Only allow substring match if user skill is 3+ characters
    # AND it appears as a whole word (surrounded by spaces or at start/end)
    if len(u) >= 3:
        import re as _re
        pattern = r'(?<![a-z])' + _re.escape(u) + r'(?![a-z])'
        if _re.search(pattern, c):
            return True
        pattern2 = r'(?<![a-z])' + _re.escape(c) + r'(?![a-z])'
        if _re.search(pattern2, u):
            return True
    return False

def compute_match_score(user_skills, career_title, domain_prefs, personality, goals):
    """
    Improved scoring formula with hard disqualification:
    - Uses word-boundary matching (prevents 'C' matching 'Clinical Diagnosis')
    - If zero skills match AND career has domain-exclusive skills → score capped at 15
    - match = (skill_match * 0.45) + (domain_interest * 0.20) +
              (personality_fit * 0.15) + (market_demand * 0.20)
    """
    career_skills = CAREER_SKILL_MAP.get(career_title, [])
    if not career_skills:
        return 0

    # 1. Skill match (45%)
    matched = 0
    for cs in career_skills:
        for us in user_skills:
            if us.lower().strip() == cs.lower().strip():  # exact match
                matched += 1.0; break
            elif skill_matches(us, cs):                    # word-boundary partial
                matched += 0.6; break
    skill_score = min((matched / len(career_skills)) * 100, 100)

    # ── HARD DISQUALIFICATION ──
    # If the career has domain-exclusive skills and user has NONE of them → hard cap
    exclusive = DOMAIN_EXCLUSIVE_SKILLS.get(career_title, [])
    if exclusive:
        has_any_exclusive = any(skill_matches(us, ex) or us.lower() == ex.lower()
                                for ex in exclusive for us in user_skills)
        if not has_any_exclusive and skill_score < 25:
            return max(round(skill_score * 0.3), 5)

    # 2. Domain interest (20%)
    career_domain, _ = DOMAIN_MAP.get(career_title, ("", False))
    if career_domain in domain_prefs:
        domain_score = 85
    elif "Not Sure — Explore All" in domain_prefs:
        domain_score = 60
    else:
        domain_score = 20

    # 3. Personality fit (15%)
    personality_career_map = {
        "Machine Learning Engineer":     ["Analytical thinker","Data-driven","Detail-oriented","Independent worker"],
        "Data Scientist":                ["Analytical thinker","Data-driven","Creative problem solver"],
        "Full Stack Developer":          ["Detail-oriented","Independent worker","Creative problem solver"],
        "Backend Developer":             ["Analytical thinker","Detail-oriented","Independent worker"],
        "Software Engineer":             ["Analytical thinker","Detail-oriented","Methodical planner"],
        "DevOps Engineer":               ["Methodical planner","Detail-oriented","Analytical thinker"],
        "Competitive Programmer":        ["Analytical thinker","Data-driven","Detail-oriented","Methodical planner"],
        "Cybersecurity Analyst":         ["Analytical thinker","Detail-oriented","Risk-taker"],
        "Embedded Systems Engineer":     ["Detail-oriented","Methodical planner","Analytical thinker"],
        "Cloud Architect":               ["Big-picture visionary","Methodical planner","Analytical thinker"],
        "UI/UX Designer":                ["Creative problem solver","People-oriented","Detail-oriented"],
        "Product Manager":               ["Natural leader","Big-picture visionary","People-oriented"],
        "Doctor / Physician":            ["Empathetic listener","Detail-oriented","Methodical planner"],
        "Lawyer":                        ["Analytical thinker","Detail-oriented","Methodical planner"],
        "Teacher / Educator":            ["Empathetic listener","People-oriented"],
        "Entrepreneur / Startup Founder":["Risk-taker","Big-picture visionary","Natural leader"],
        "Marketing Manager":             ["Creative problem solver","People-oriented","Big-picture visionary"],
        "Financial Analyst":             ["Analytical thinker","Detail-oriented","Data-driven"],
        "HR Manager":                    ["Empathetic listener","People-oriented","Team collaborator"],
        "Content Creator":               ["Creative problem solver","Independent worker","Big-picture visionary"],
        "Graphic Designer":              ["Creative problem solver","Detail-oriented","Independent worker"],
    }
    ideal = personality_career_map.get(career_title, [])
    p_matched = sum(1 for p in personality if p in ideal)
    personality_score = min((p_matched / max(len(ideal), 1)) * 100, 100) if ideal else 50

    # 4. Market demand (20%)
    demand = MARKET_DEMAND.get(career_title, 70)

    # Weighted formula
    final = (skill_score * 0.45) + (domain_score * 0.20) + (personality_score * 0.15) + (demand * 0.20)
    return round(min(final, 99))

def find_skill_gaps(user_skills, career_title):
    career_skills = CAREER_SKILL_MAP.get(career_title, [])
    gaps = [s for s in career_skills
            if not any(us.lower().strip() == s.lower().strip() or skill_matches(us, s)
                       for us in user_skills)]
    return gaps[:5]

def find_matched_skills(user_skills, career_title):
    career_skills = CAREER_SKILL_MAP.get(career_title, [])
    matched = [s for s in career_skills
               if any(us.lower().strip() == s.lower().strip() or skill_matches(us, s)
                      for us in user_skills)]
    return matched

# ─────────────────────────────────────────
#  RESUME PARSER
# ─────────────────────────────────────────

ALL_SKILLS = list(CAREER_SKILL_MAP.keys()) + [
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
]

def extract_text_from_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    text = ""
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    finally:
        os.unlink(tmp_path)
    return text

def parse_resume(text):
    text_lower = text.lower()
    found_skills = []
    for skill in ALL_SKILLS:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    # Extract name (first non-empty line usually)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    name = lines[0] if lines else "Candidate"
    # Extract email
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', text)
    email = email_match.group() if email_match else ""
    # Extract phone
    phone_match = re.search(r'[\+]?[\d\s\-\(\)]{10,14}', text)
    phone = phone_match.group().strip() if phone_match else ""
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": list(set(found_skills)),
        "raw_text": text[:2000],
    }

# ─────────────────────────────────────────
#  GROQ API CALL
# ─────────────────────────────────────────

def call_groq(api_key, model, messages, json_mode=True):
    payload = {
        "model": model,
        "max_tokens": 4000,
        "temperature": 0.7,
        "messages": messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    import requests as req_lib
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0 (compatible; CareerIQ/1.0)",
    }
    resp = req_lib.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    if not resp.ok:
        raise urllib.error.HTTPError(
            url=resp.url, code=resp.status_code,
            msg=resp.text, hdrs=None, fp=None
        )
    result = resp.json()
    raw = result["choices"][0]["message"]["content"]
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw.strip()).strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        raw = m.group()
    return json.loads(raw)

# ─────────────────────────────────────────
#  MULTI-AGENT ORCHESTRATOR
# ─────────────────────────────────────────

def run_agents(profile, api_key, model):
    """
    4-agent pipeline:
    Agent 1 — Profile Analyzer
    Agent 2 — Career Matcher (algorithmic + AI)
    Agent 3 — Skill Gap Detector
    Agent 4 — Roadmap & Resource Generator
    """
    user_skills = profile.get("skills", [])
    domains     = profile.get("domains", [])
    personality = profile.get("personality", [])
    goals       = profile.get("goals", [])

    # ── AGENT 1: Compute algorithmic scores for ALL careers
    scored = []
    for career in CAREER_SKILL_MAP:
        score = compute_match_score(user_skills, career, domains, personality, goals)
        gaps  = find_skill_gaps(user_skills, career)
        matched = find_matched_skills(user_skills, career)
        domain, is_it = DOMAIN_MAP.get(career, ("General", False))
        scored.append({
            "title": career,
            "algoScore": score,
            "domain": domain,
            "isIT": is_it,
            "matchedSkills": matched,
            "gapSkills": gaps,
            "salaryRange": SALARY_RANGES.get(career, "—"),
            "marketDemand": MARKET_DEMAND.get(career, 70),
        })

    scored.sort(key=lambda x: x["algoScore"], reverse=True)
    top5_algo = scored[:5]
    top5_titles = [c["title"] for c in top5_algo]

    # ── AGENT 2: AI enrichment for top 5 careers
    agent2_prompt = f"""You are a Career Intelligence Agent. Given this user profile and top 5 algorithmically matched careers, enrich each career with personalized insights.

USER PROFILE:
Name: {profile.get('name','')}, Age: {profile.get('age','')}, Education: {profile.get('education','')}
Skills: {', '.join(user_skills)}
Soft Skills: {', '.join(profile.get('softSkills',[]))}
Personality: {', '.join(personality)}
Goals: {', '.join(goals)}
Preferred Domains: {', '.join(domains)}
Work Mode: {profile.get('workMode','')}
Salary Expectation: {profile.get('salary','')}

TOP 5 CAREERS (already scored algorithmically):
{json.dumps(top5_titles)}

Return ONLY valid JSON:
{{
  "overallSummary": "3 sentence personalized summary referencing their actual skills",
  "personalityInsight": "2 sentences about personality-career fit",
  "strengthsIdentified": ["s1","s2","s3","s4"],
  "immediateActions": ["action1","action2","action3"],
  "alternativeCareers": ["c1","c2","c3"],
  "careers": [
    {{
      "title": "exact title from list above",
      "whyFit": "2 sentences referencing their specific skills",
      "dailyWork": "1 sentence typical day",
      "growthOutlook": "High|Medium|Stable",
      "roadmap": ["Step 1: ...","Step 2: ...","Step 3: ...","Step 4: ...","Step 5: ...","Step 6: ..."],
      "courses": ["Course – Platform","Course – Platform","Course – Platform"],
      "certifications": ["Cert 1","Cert 2"],
      "jobRoles": ["Role A","Role B","Role C"],
      "companies": ["Company A","Company B","Company C","Company D"],
      "portfolioProjects": ["Project 1","Project 2","Project 3"]
    }}
  ]
}}"""

    ai_result = call_groq(api_key, model, [
        {"role": "system", "content": "You are an expert career guidance AI. Respond ONLY with valid JSON."},
        {"role": "user",   "content": agent2_prompt},
    ])

    # ── AGENT 3: Skill Gap Analysis
    agent3_prompt = f"""You are a Skill Gap Analysis Agent. Analyze this user's skills against the job market.

User Skills: {', '.join(user_skills)}
Target Career: {top5_titles[0]}

Return ONLY valid JSON:
{{
  "skillMatchPercent": 75,
  "userSkillStrengths": [{{"skill":"Python","level":85}},{{"skill":"SQL","level":70}}],
  "criticalGaps": [{{"skill":"Docker","demand":88,"priority":"High"}},{{"skill":"Kubernetes","demand":82,"priority":"High"}}],
  "learningPath": [{{"week":"Week 1-2","focus":"...","resource":"..."}},{{"week":"Week 3-4","focus":"...","resource":"..."}}],
  "marketTrends": [{{"skill":"GenAI","growth":"+145% YoY"}},{{"skill":"LLMs","growth":"+120% YoY"}}]
}}"""

    gap_result = call_groq(api_key, model, [
        {"role": "system", "content": "You are a skill gap analysis expert. Respond ONLY with valid JSON."},
        {"role": "user",   "content": agent3_prompt},
    ])

    # ── MERGE results
    ai_careers = {c["title"]: c for c in ai_result.get("careers", [])}

    top5_merged = []
    for algo_career in top5_algo:
        title = algo_career["title"]
        ai_enrichment = ai_careers.get(title, {})
        merged = {**algo_career, **ai_enrichment}
        # Keep algo score as matchPercent
        merged["matchPercent"] = algo_career["algoScore"]
        # Compute weighted breakdown scores
        merged["skillScore"]       = round(len(algo_career["matchedSkills"]) / max(len(CAREER_SKILL_MAP.get(title,["x"])),1) * 100)
        merged["domainScore"]      = 85 if algo_career["domain"] in domains else 45
        merged["personalityScore"] = round(merged["matchPercent"] * 0.9)
        merged["marketScore"]      = algo_career["marketDemand"]
        top5_merged.append(merged)

    return {
        "fitScore": top5_merged[0]["matchPercent"] if top5_merged else 0,
        "overallSummary": ai_result.get("overallSummary",""),
        "personalityInsight": ai_result.get("personalityInsight",""),
        "strengthsIdentified": ai_result.get("strengthsIdentified",[]),
        "immediateActions": ai_result.get("immediateActions",[]),
        "alternativeCareers": ai_result.get("alternativeCareers",[]),
        "topCareers": top5_merged,
        "skillGapAnalysis": gap_result,
        "allScored": scored,
    }

# ─────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────

@app.route("/")
def index():
    html_path = os.path.join(TEMPLATE_DIR, "index.html")
    if not os.path.exists(html_path):
        return f"""
        <h2 style="font-family:sans-serif;color:#c00;padding:40px">
            ❌ index.html not found!<br><br>
            <span style="font-size:16px;color:#333">
            Expected location:<br>
            <code>{html_path}</code><br><br>
            Please move index.html into a folder called <b>templates</b>
            next to app.py and restart.
            </span>
        </h2>""", 404
    # Tell the frontend if a server-side key is already configured
    return render_template("index.html", key_configured=bool(ENV_API_KEY))

@app.route("/api/key-status", methods=["GET"])
def key_status():
    """Frontend can check if a server-side key is configured."""
    return jsonify({"configured": bool(ENV_API_KEY)})

@app.route("/api/parse-resume", methods=["POST"])
def parse_resume_route():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["resume"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files supported"}), 400
    file_bytes = f.read()
    text = extract_text_from_pdf(file_bytes)
    if not text.strip():
        return jsonify({"error": "Could not extract text from PDF"}), 400
    parsed = parse_resume(text)
    return jsonify(parsed)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Use server-side key if configured, otherwise use key from frontend
    api_key = ENV_API_KEY or data.get("apiKey", "").strip()
    model   = data.get("model", "llama-3.3-70b-versatile")
    profile = data.get("profile", {})

    if not api_key:
        return jsonify({"error": "No API key found. Either set GROQ_API_KEY in your .env file or enter it in the browser."}), 400
    if not api_key.startswith("gsk_"):
        return jsonify({"error": "Invalid Groq API key — must start with gsk_"}), 400
    if not profile.get("skills"):
        return jsonify({"error": "Skills are required"}), 400

    try:
        result = run_agents(profile, api_key, model)
        return jsonify(result)
    except urllib.error.HTTPError as e:
        body = e.msg if hasattr(e, 'msg') else str(e)
        if e.code == 429:
            return jsonify({"error": "Rate limit hit. Please wait 30 seconds and try again."}), 429
        if e.code == 401:
            return jsonify({"error": "Invalid API key. Please check your Groq key at console.groq.com"}), 401
        if e.code == 403:
            return jsonify({"error": "Access denied (403). Your API key may be invalid or expired. Get a fresh key at console.groq.com/keys"}), 403
        return jsonify({"error": f"Groq API error {e.code}: {str(body)[:300]}"}), 500
    except Exception as e:
        err = str(e)
        if "401" in err:
            return jsonify({"error": "Invalid API key. Get a fresh key at console.groq.com/keys"}), 401
        if "403" in err:
            return jsonify({"error": "Access denied (403). Please get a fresh API key at console.groq.com/keys"}), 403
        if "429" in err:
            return jsonify({"error": "Rate limit hit. Please wait 30 seconds and retry."}), 429
        return jsonify({"error": err}), 500

@app.route("/api/score-preview", methods=["POST"])
def score_preview():
    """Quick algorithmic scoring without AI — for instant feedback"""
    data = request.get_json()
    skills     = data.get("skills", [])
    domains    = data.get("domains", [])
    personality= data.get("personality", [])
    goals      = data.get("goals", [])

    scored = []
    for career in CAREER_SKILL_MAP:
        score = compute_match_score(skills, career, domains, personality, goals)
        scored.append({"title": career, "score": score,
                       "domain": DOMAIN_MAP.get(career,("",False))[0],
                       "isIT": DOMAIN_MAP.get(career,("",False))[1]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return jsonify(scored[:8])

if __name__ == "__main__":
    print("\n  ╔══════════════════════════════════════╗")
    print("  ║   CareerIQ — AI Career Platform      ║")
    print("  ║   Running at http://localhost:5000    ║")
    print("  ╚══════════════════════════════════════╝\n")
    app.run(debug=True, port=5000)
