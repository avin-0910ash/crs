# CRS Personalized Course Recommendation System — Live Search + Groq/Grok Edition

This Streamlit application searches live web listings on 30 course-provider domains. It does not use the bundled CSV for recommendations. Results are ranked against the student's profile and an AI provider (Groq or xAI/Grok) explains the best matches, tradeoffs, and a learning roadmap.

## Run on Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set one provider key. A key beginning with `gsk_` is a **Groq** key, not an xAI/Grok key: use `GROQ_API_KEY` and `GROQ_MODEL=llama-3.3-70b-versatile`. Create Groq keys at https://console.groq.com/keys. For an xAI key, use `XAI_API_KEY` and `GROK_MODEL=grok-4.7`; create keys at https://console.x.ai/. Do not paste keys into chat, screenshots, source files, or GitHub.

Run the app:

```powershell
python -m streamlit run app.py
```

## Live search behavior

- The app attempts web searches against 30 course-provider domains, including Coursera, Udemy, edX, DataCamp, freeCodeCamp, SWAYAM, NPTEL, Microsoft Learn, Google Skills, IBM SkillsBuild, AWS Skill Builder, Cisco Networking Academy, and others.
- Results depend on web search provider availability, indexing, and provider restrictions. The diagnostics panel shows which domains returned results and errors. It does not claim all 30 sites returned results.
- It does not scrape behind logins/paywalls and does not fall back to the CSV.
- Course details such as price, duration, ratings, and prerequisites may be missing from snippets; verify them on the official course page.
- AI match percentages are estimates of fit to the submitted profile, not objective course quality scores.
