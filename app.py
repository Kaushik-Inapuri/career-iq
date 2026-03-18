"""
CareerIQ — AI Career Intelligence Platform
Flask Backend — Production Ready for Render
"""

import json
import os
import re
import urllib.request
import urllib.error
from flask import Flask, request, jsonify, render_template
import pdfplumber
import tempfile

# ── Load .env for local dev ──
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

ENV_API_KEY  = os.environ.get("GROQ_API_KEY", "").strip()
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# ─────────────────────────────────────────
#  KNOWLEDGE BASE
# ─────────────────────────────────────────

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

# ─────────────────────────────────────────
#  SCORING
# ─────────────────────────────────────────

def skill_matches(user_skill, career_skill):
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

def compute_match_score(user_skills, career_title, domain_prefs, personality, goals):
    career_skills = CAREER_SKILL_MAP.get(career_title, [])
    if not career_skills:
        return 0
    matched = 0
    for cs in career_skills:
        for us in user_skills:
            if us.lower().strip() == cs.lower().strip():
                matched += 1.0; break
            elif skill_matches(us, cs):
                matched += 0.6; break
    skill_score = min((matched / len(career_skills)) * 100, 100)

    exclusive = DOMAIN_EXCLUSIVE_SKILLS.get(career_title, [])
    if exclusive:
        has_any = any(skill_matches(us, ex) or us.lower() == ex.lower()
                      for ex in exclusive for us in user_skills)
        if not has_any and skill_score < 25:
            return max(round(skill_score * 0.3), 5)

    career_domain, _ = DOMAIN_MAP.get(career_title, ("", False))
    domain_score = 85 if career_domain in domain_prefs else (60 if "Not Sure — Explore All" in domain_prefs else 20)

    personality_career_map = {
        "Machine Learning Engineer":     ["Analytical thinker","Data-driven","Detail-oriented","Independent worker"],
        "Data Scientist":                 ["Analytical thinker","Data-driven","Creative problem solver"],
        "Full Stack Developer":           ["Detail-oriented","Independent worker","Creative problem solver"],
        "Backend Developer":              ["Analytical thinker","Detail-oriented","Independent worker"],
        "Software Engineer":              ["Analytical thinker","Detail-oriented","Methodical planner"],
        "DevOps Engineer":                ["Methodical planner","Detail-oriented","Analytical thinker"],
        "Competitive Programmer":         ["Analytical thinker","Data-driven","Detail-oriented","Methodical planner"],
        "Cybersecurity Analyst":          ["Analytical thinker","Detail-oriented","Risk-taker"],
        "Embedded Systems Engineer":      ["Detail-oriented","Methodical planner","Analytical thinker"],
        "Cloud Architect":                ["Big-picture visionary","Methodical planner","Analytical thinker"],
        "UI/UX Designer":                 ["Creative problem solver","People-oriented","Detail-oriented"],
        "Product Manager":                ["Natural leader","Big-picture visionary","People-oriented"],
        "Doctor / Physician":             ["Empathetic listener","Detail-oriented","Methodical planner"],
        "Lawyer":                         ["Analytical thinker","Detail-oriented","Methodical planner"],
        "Teacher / Educator":             ["Empathetic listener","People-oriented"],
        "Entrepreneur / Startup Founder": ["Risk-taker","Big-picture visionary","Natural leader"],
        "Marketing Manager":              ["Creative problem solver","People-oriented","Big-picture visionary"],
        "Financial Analyst":              ["Analytical thinker","Detail-oriented","Data-driven"],
        "HR Manager":                     ["Empathetic listener","People-oriented","Team collaborator"],
        "Content Creator":                ["Creative problem solver","Independent worker","Big-picture visionary"],
        "Graphic Designer":               ["Creative problem solver","Detail-oriented","Independent worker"],
    }
    ideal = personality_career_map.get(career_title, [])
    p_matched = sum(1 for p in personality if p in ideal)
    personality_score = min((p_matched / max(len(ideal), 1)) * 100, 100) if ideal else 50
    demand = MARKET_DEMAND.get(career_title, 70)

    return round(min((skill_score * 0.45) + (domain_score * 0.20) + (personality_score * 0.15) + (demand * 0.20), 99))

def find_skill_gaps(user_skills, career_title):
    return [s for s in CAREER_SKILL_MAP.get(career_title, [])
            if not any(us.lower().strip() == s.lower().strip() or skill_matches(us, s)
                       for us in user_skills)][:5]

def find_matched_skills(user_skills, career_title):
    return [s for s in CAREER_SKILL_MAP.get(career_title, [])
            if any(us.lower().strip() == s.lower().strip() or skill_matches(us, s)
                   for us in user_skills)]

# ─────────────────────────────────────────
#  RESUME PARSER
# ─────────────────────────────────────────

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

def extract_text_from_pdf(file_bytes):
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

def parse_resume(text):
    text_lower = text.lower()
    found = list(set(s for s in ALL_SKILLS if s.lower() in text_lower))
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    email = (re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', text) or type('',(),{'group':lambda s:''})()).group()
    phone = (re.search(r'[\+]?[\d\s\-\(\)]{10,14}', text) or type('',(),{'group':lambda s:''})()).group()
    return {"name": lines[0] if lines else "Candidate",
            "email": email, "phone": phone.strip(),
            "skills": found, "raw_text": text[:2000]}

# ─────────────────────────────────────────
#  GROQ API
# ─────────────────────────────────────────

def call_groq(api_key, model, messages, json_mode=True):
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

# ─────────────────────────────────────────
#  4-AGENT PIPELINE
# ─────────────────────────────────────────

def run_agents(profile, api_key, model):
    user_skills = profile.get("skills", [])
    domains     = profile.get("domains", [])
    personality = profile.get("personality", [])
    goals       = profile.get("goals", [])

    # Agent 1 — Algorithmic scoring
    scored = []
    for career in CAREER_SKILL_MAP:
        score   = compute_match_score(user_skills, career, domains, personality, goals)
        domain, is_it = DOMAIN_MAP.get(career, ("General", False))
        scored.append({"title": career, "algoScore": score, "domain": domain, "isIT": is_it,
                       "matchedSkills": find_matched_skills(user_skills, career),
                       "gapSkills": find_skill_gaps(user_skills, career),
                       "salaryRange": SALARY_RANGES.get(career, "—"),
                       "marketDemand": MARKET_DEMAND.get(career, 70)})
    scored.sort(key=lambda x: x["algoScore"], reverse=True)
    top5_algo   = scored[:5]
    top5_titles = [c["title"] for c in top5_algo]

    # Agent 2 — AI enrichment
    ai_result = call_groq(api_key, model, [
        {"role": "system", "content": "You are an expert career guidance AI. Respond ONLY with valid JSON."},
        {"role": "user", "content": f"""Enrich these top 5 careers for this user.

USER PROFILE:
Name: {profile.get('name','')}, Education: {profile.get('education','')}
Skills: {', '.join(user_skills)}
Personality: {', '.join(personality)}
Goals: {', '.join(goals)}
TOP 5 CAREERS: {json.dumps(top5_titles)}

Return ONLY valid JSON:
{{
  "overallSummary": "3 sentence personalized summary",
  "personalityInsight": "2 sentences about personality fit",
  "strengthsIdentified": ["s1","s2","s3","s4"],
  "immediateActions": ["action1","action2","action3"],
  "alternativeCareers": ["c1","c2","c3"],
  "careers": [
    {{
      "title": "exact title from list",
      "whyFit": "2 sentences referencing their skills",
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
}}"""}
    ])

    # Agent 3 — Skill Gap Analysis
    gap_result = call_groq(api_key, model, [
        {"role": "system", "content": "You are a skill gap analysis expert. Respond ONLY with valid JSON."},
        {"role": "user", "content": f"""Analyze skill gaps.
User Skills: {', '.join(user_skills)}
Target Career: {top5_titles[0]}
Return ONLY valid JSON:
{{
  "skillMatchPercent": 75,
  "userSkillStrengths": [{{"skill":"Python","level":85}}],
  "criticalGaps": [{{"skill":"Docker","demand":88,"priority":"High"}}],
  "learningPath": [{{"week":"Week 1-2","focus":"...","resource":"..."}}],
  "marketTrends": [{{"skill":"GenAI","growth":"+145% YoY"}}]
}}"""}
    ])

    # Merge
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

    return {
        "fitScore":           top5_merged[0]["matchPercent"] if top5_merged else 0,
        "overallSummary":     ai_result.get("overallSummary", ""),
        "personalityInsight": ai_result.get("personalityInsight", ""),
        "strengthsIdentified":ai_result.get("strengthsIdentified", []),
        "immediateActions":   ai_result.get("immediateActions", []),
        "alternativeCareers": ai_result.get("alternativeCareers", []),
        "topCareers":         top5_merged,
        "skillGapAnalysis":   gap_result,
        "allScored":          scored,
    }

# ─────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────

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
    return jsonify(parse_resume(extract_text_from_pdf(f.read())))

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data    = request.get_json()
    if not data: return jsonify({"error": "No data provided"}), 400
    api_key = ENV_API_KEY or data.get("apiKey", "").strip()
    model   = data.get("model", "llama-3.3-70b-versatile")
    profile = data.get("profile", {})
    if not api_key:      return jsonify({"error": "No API key. Set GROQ_API_KEY in environment."}), 400
    if not api_key.startswith("gsk_"): return jsonify({"error": "Invalid Groq API key."}), 400
    if not profile.get("skills"):      return jsonify({"error": "Skills are required."}), 400
    try:
        return jsonify(run_agents(profile, api_key, model))
    except urllib.error.HTTPError as e:
        codes = {429: "Rate limit hit. Wait 30s and retry.",
                 401: "Invalid API key.",
                 403: "Access denied (403). Get a fresh key at console.groq.com/keys"}
        return jsonify({"error": codes.get(e.code, f"Groq error {e.code}")}), e.code
    except Exception as e:
        err = str(e)
        for code, msg in [("429","Rate limit hit."),("401","Invalid API key."),("403","Access denied.")]:
            if code in err: return jsonify({"error": msg}), int(code)
        return jsonify({"error": err}), 500

@app.route("/api/score-preview", methods=["POST"])
def score_preview():
    data = request.get_json()
    scored = sorted([
        {"title": c, "score": compute_match_score(data.get("skills",[]), c,
                                                   data.get("domains",[]),
                                                   data.get("personality",[]),
                                                   data.get("goals",[])),
         "domain": DOMAIN_MAP.get(c,("",False))[0],
         "isIT":   DOMAIN_MAP.get(c,("",False))[1]}
        for c in CAREER_SKILL_MAP
    ], key=lambda x: x["score"], reverse=True)
    return jsonify(scored[:8])

# ─────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  CareerIQ running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)