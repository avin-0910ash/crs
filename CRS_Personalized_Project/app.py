"""CRS: Streamlit login + live course discovery + Groq/Grok-powered personalized analysis.
Run: python -m streamlit run app.py
"""
import os, json, sqlite3, hashlib, hmac, secrets, re
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv
from online_search import search_online_courses, PROVIDERS

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DB_PATH = BASE_DIR / "crs_users.db"
st.set_page_config(page_title="CRS | Personalized Course Recommender", page_icon="🎓", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#f7faf8; }
[data-testid="stSidebar"] { background:#eef6f0; }
.crs-hero { padding:1.2rem 1.4rem; border-radius:18px; background:linear-gradient(120deg,#e4f5e9,#f2f8ff); border:1px solid #d6eadc; }
</style>
""", unsafe_allow_html=True)


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, salt TEXT NOT NULL, password_hash TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("CREATE TABLE IF NOT EXISTS saved_courses (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, course_title TEXT NOT NULL, platform TEXT, url TEXT, saved_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES users(id))")
    conn.commit(); return conn


def password_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 220_000).hex()


def create_user(name, email, password):
    conn = connect_db(); salt = secrets.token_hex(16)
    try:
        conn.execute("INSERT INTO users(name,email,salt,password_hash) VALUES(?,?,?,?)", (name.strip(), email.strip().lower(), salt, password_hash(password, salt)))
        conn.commit(); return True, "Account created. Please log in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    finally: conn.close()


def authenticate(email, password):
    conn = connect_db(); row = conn.execute("SELECT id,name,email,salt,password_hash FROM users WHERE email=?", (email.strip().lower(),)).fetchone(); conn.close()
    if not row or not hmac.compare_digest(password_hash(password, row[3]), row[4]): return None
    return {"id": row[0], "name": row[1], "email": row[2]}


def show_auth():
    st.markdown('<div class="crs-hero"><h1>🎓 CRS</h1><p>Personalized Course Recommendation System</p><span>Search live course listings across 30 learning-provider websites and get AI-personalized explanations.</span></div>', unsafe_allow_html=True)
    login_tab, register_tab = st.tabs(["Login", "Create account"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email address", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
        if submitted:
            user = authenticate(email, password)
            if user: st.session_state.user = user; st.rerun()
            else: st.error("Email or password is incorrect.")
    with register_tab:
        with st.form("register_form"):
            name = st.text_input("Full name")
            email = st.text_input("Email address", key="register_email")
            password = st.text_input("Create password (minimum 8 characters)", type="password", key="register_password")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
        if submitted:
            if not name.strip() or not email.strip() or len(password) < 8: st.error("Enter your name and email, and use a password with at least 8 characters.")
            elif password != confirm: st.error("Passwords do not match.")
            else:
                ok, message = create_user(name, email, password); (st.success if ok else st.error)(message)


def _api_provider_and_key():
    """Detect Groq keys separately from xAI/Grok keys."""
    groq_key = (
        os.getenv("GROQ_API_KEY") or ""
    ).strip()

    xai_key = (
        os.getenv("XAI_API_KEY")
        or os.getenv("GROK_API_KEY")
        or ""
    ).strip()

    # Use the configured Groq key, or the fallback key.
    if not groq_key and not xai_key.startswith("gsk_"):
        groq_key = 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

    # Detect Groq keys even if placed in an xAI variable.
    if groq_key or xai_key.startswith("gsk_"):
        return "groq", groq_key or xai_key

    if xai_key:
        return "xai", xai_key

    return None, None


def _available_model(api_key, provider, requested):
    """Return a model supported by the configured provider/key when possible."""
    base = "https://api.groq.com/openai/v1" if provider == "groq" else "https://api.x.ai/v1"
    try:
        r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        if not r.ok:
            # Keep the requested model; the completion response will show the actual error.
            return requested
        ids = [item.get("id", "") for item in r.json().get("data", []) if item.get("id")]
        if requested in ids:
            return requested
        candidates = ( ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"]
                       if provider == "groq" else ["grok-4.7", "grok-4.6", "grok-4.1-fast"] )
        for candidate in candidates:
            if candidate in ids:
                return candidate
        return ids[0] if ids else requested
    except requests.RequestException:
        return requested


def grok_analyze(profile, courses):
    provider, api_key = _api_provider_and_key()
    if not api_key:
        return None, "AI API key is missing. For a gsk_ key, set GROQ_API_KEY in .env. For an xAI key, set XAI_API_KEY."
    records = []
    for _, row in courses.head(30).iterrows():
        records.append({"title": str(row.get("course_title", "")), "platform": str(row.get("platform", "")),
                        "url": str(row.get("url", "")), "search_snippet": str(row.get("description", "")),
                        "price": row.get("price") if pd.notna(row.get("price")) else "Not found in search result",
                        "level": str(row.get("level", "Not listed")), "duration": str(row.get("content_duration", "Not listed")),
                        "search_match_score": round(float(row.get("relevance_score", 0)) * 100, 1)})
    prompt = {"student_profile": profile, "live_search_results": records,
      "instructions": "You are a careful course advisor. Recommend the best 5 courses ONLY from the provided live web search results. Do not invent course facts, prices, ratings, duration, prerequisites, accreditation or links. Use snippets and profile to explain match. If details are missing, say verify on provider page. Never claim you searched a provider unless its result is supplied. Return valid JSON with keys overview, recommendations (array of title, platform, url, match_percent, why_best, tradeoffs, next_step), learning_roadmap (array of strings). Copy title/platform/url exactly from source records. Match percent is an estimated profile fit, not verified course quality."}
    requested = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile") if provider == "groq" else os.getenv("GROK_MODEL", "grok-4.7")
    base = "https://api.groq.com/openai/v1" if provider == "groq" else "https://api.x.ai/v1"
    model = _available_model(api_key, provider, requested)
    try:
        response = requests.post(f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [
                {"role": "system", "content": "You are a careful personalized learning advisor. Return valid JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}], "temperature": 0.2}, timeout=120)
        if not response.ok:
            try:
                body = response.json()
                detail = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("error")
                detail = detail or body.get("message") or response.text[:600]
            except Exception:
                detail = response.text[:600]
            provider_label = "Groq" if provider == "groq" else "xAI/Grok"
            if response.status_code in (400, 401) and provider == "xai" and api_key.startswith("gsk_"):
                detail = "This key starts with gsk_ and belongs to Groq, not xAI. Put it in GROQ_API_KEY, or create an xAI key at https://console.x.ai."
            return None, f"{provider_label} API error HTTP {response.status_code}: {detail}"
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(content)
        return parsed, None
    except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
        return None, f"AI analysis failed: {exc}. Live web search results are still available below."


def save_course(user_id, row):
    conn = connect_db()
    conn.execute("INSERT INTO saved_courses(user_id,course_title,platform,url) VALUES(?,?,?,?)", (user_id, str(row.get("course_title", "")), str(row.get("platform", "")), str(row.get("url", ""))))
    conn.commit(); conn.close()


def main_app():
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"### Hello, {user['name']} 👋"); st.caption(user["email"])
        if st.button("Log out", use_container_width=True): st.session_state.pop("user", None); st.rerun()
        st.divider(); st.markdown("**Live search coverage**"); st.write(f"{len(PROVIDERS)} course-provider websites will be queried when you recommend courses.")
        st.caption("Search results depend on search-engine availability and each provider's public indexing. Course price and duration must be verified on the provider page.")
    st.markdown('<div class="crs-hero"><h1>Find the right course for your future</h1><p>CRS searches live online listings and uses Grok to explain which courses best match your goals.</p></div>', unsafe_allow_html=True)
    with st.form("student_profile_form"):
        st.subheader("Your learning profile")
        c1, c2 = st.columns(2)
        with c1:
            education = st.selectbox("Education", ["High school", "Diploma", "Bachelor's degree", "Master's degree", "Working professional", "Other"])
            current = st.text_input("Skills you already have", "Python, HTML, CSS")
            desired = st.text_input("Skills you want to learn", "Machine Learning, Data Analysis")
            interests = st.text_area("Interests / topics", "Artificial Intelligence, Data Science", height=90)
        with c2:
            goal = st.text_input("Career goal", "Data Scientist")
            level = st.selectbox("Experience level", ["Beginner", "Intermediate", "Advanced", "Any"])
            budget_enabled = st.checkbox("Set maximum course budget", value=True)
            budget = st.number_input("Maximum budget (INR; search results may not show prices)", min_value=0.0, value=5000.0, step=500.0, disabled=not budget_enabled)
            hours_week = st.slider("Available learning hours per week", 1, 40, 5)
            duration = st.selectbox("Preferred duration", ["Any", "Under 1 month", "1–3 months", "3–6 months", "6+ months"])
        top_k = st.select_slider("Number of results to display", options=[5, 10, 15, 20, 30], value=15)
        submitted = st.form_submit_button("🌐 Search Online & Recommend Courses", type="primary", use_container_width=True)
    if submitted:
        profile = {"name": user["name"], "education": education, "skills_you_have": current, "skills_to_learn": desired,
                   "interests": interests, "career_goal": goal, "experience_level": level, "budget": budget if budget_enabled else None,
                   "available_hours_per_week": hours_week, "preferred_duration": duration}
        with st.spinner(f"Searching live course listings across {len(PROVIDERS)} provider websites… this may take a minute."):
            try: courses, searched_domains, search_errors = search_online_courses(profile)
            except Exception as exc:
                st.error(f"Online search could not start: {exc}"); st.stop()
        st.session_state.live_courses = courses
        st.session_state.live_profile = profile
        st.session_state.searched_domains = searched_domains
        st.session_state.search_errors = search_errors
        if courses.empty:
            st.error("The online search returned no course pages. Check your internet connection, then try again. No CSV fallback is used.")
            with st.expander("Search diagnostics"): st.write(search_errors[:10])
            st.stop()
        st.success(f"Found {len(courses)} unique course pages from {courses['platform'].nunique()} provider websites. No local CSV was used.")
        with st.spinner("AI is comparing the live course results against your profile…"):
            ai_result, ai_error = grok_analyze(profile, courses.head(30))
        if ai_error: st.warning(ai_error)
        if ai_result:
            st.subheader("✨ AI personalized recommendation")
            st.write(ai_result.get("overview", "Recommendations based on your profile and the live search results."))
            recs = ai_result.get("recommendations", [])
            if recs:
                chart_rows = []
                for i, item in enumerate(recs, 1):
                    with st.container(border=True):
                        st.markdown(f"### {i}. {item.get('title', 'Course')}")
                        st.caption(f"{item.get('platform', 'Provider not listed')} · Estimated match: {item.get('match_percent', '—')}%")
                        st.write(item.get("why_best", ""))
                        if item.get("tradeoffs"): st.caption("Things to check: " + str(item["tradeoffs"]))
                        if item.get("next_step"): st.write("**Next step:** " + str(item["next_step"]))
                        url = item.get("url", "")
                        if url.startswith("https://") or url.startswith("http://"): st.markdown(f"[Open provider course page]({url})")
                        try:
                            score_value = float(str(item.get("match_percent", 0)).replace("%", "").strip() or 0)
                        except (ValueError, TypeError):
                            score_value = 0.0
                        chart_rows.append({"Course": str(item.get("title", f"Course {i}"))[:45], "Estimated match (%)": score_value})
                if chart_rows:
                    st.subheader("Visual match comparison"); st.bar_chart(pd.DataFrame(chart_rows).set_index("Course"))
            roadmap = ai_result.get("learning_roadmap", [])
            if roadmap:
                st.subheader("Your learning roadmap")
                for i, step in enumerate(roadmap, 1): st.markdown(f"**Step {i}.** {step}")
        st.subheader("Live course pages found")
        display = courses.head(top_k).copy()
        display["Estimated match (%)"] = (display["relevance_score"] * 100).round().astype(int)
        st.dataframe(display[["rank", "course_title", "platform", "Estimated match (%)", "url", "description"]], use_container_width=True, hide_index=True, column_config={"url": st.column_config.LinkColumn("Course link", display_text="Open course"), "description": st.column_config.TextColumn("Search snippet", width="large")})
        with st.expander("Provider coverage and search diagnostics"):
            found_domains = sorted(set(courses["source_domain"].tolist()))
            st.write(f"Providers with results: {len(found_domains)} / {len(PROVIDERS)}")
            st.write(", ".join(found_domains))
            if search_errors: st.warning("Some providers could not be searched in this run."); st.write(search_errors[:12])
        st.caption("Search match scores are keyword-based estimates. AI match percentages are estimates, not ratings of course quality. Search snippets may be incomplete; verify fees, curriculum, duration, and prerequisites on each official provider page.")
    elif st.session_state.get("live_courses") is not None:
        st.caption("Your last live search is still available in this session. Submit the form to search again.")
        st.dataframe(st.session_state.live_courses.head(20), use_container_width=True, hide_index=True)


if "user" not in st.session_state: show_auth()
else: main_app()
