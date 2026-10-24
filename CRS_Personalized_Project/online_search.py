"""Live course discovery across multiple course-provider domains using web search."""
from __future__ import annotations
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import pandas as pd

PROVIDERS = [
    ("Coursera", "coursera.org"), ("Udemy", "udemy.com"), ("edX", "edx.org"),
    ("Udacity", "udacity.com"), ("FutureLearn", "futurelearn.com"),
    ("Pluralsight", "pluralsight.com"), ("DataCamp", "datacamp.com"),
    ("Codecademy", "codecademy.com"), ("freeCodeCamp", "freecodecamp.org"),
    ("Khan Academy", "khanacademy.org"), ("Simplilearn", "simplilearn.com"),
    ("Great Learning", "greatlearning.in"), ("SWAYAM", "swayam.gov.in"),
    ("NPTEL", "nptel.ac.in"), ("Skill India Digital", "skillindiadigital.gov.in"),
    ("Google Skills", "skills.google"), ("Microsoft Learn", "learn.microsoft.com"),
    ("IBM SkillsBuild", "skillsbuild.org"), ("AWS Skill Builder", "skillbuilder.aws"),
    ("Google Cloud Skills Boost", "cloudskillsboost.google"), ("Cisco Networking Academy", "netacad.com"),
    ("Oracle University", "education.oracle.com"), ("LinkedIn Learning", "linkedin.com/learning"),
    ("Skillshare", "skillshare.com"), ("Alison", "alison.com"),
    ("OpenLearning", "openlearning.com"), ("GeeksforGeeks", "geeksforgeeks.org"),
    ("Educative", "educative.io"), ("Frontend Masters", "frontendmasters.com"),
    ("Databricks Academy", "databricks.com/learn"),
]


def _terms(value):
    if isinstance(value, (list, tuple, set)):
        value = " ".join(map(str, value))
    return re.sub(r"[^\w+#. -]", " ", str(value or "")).strip()


def _score(row, profile):
    title = str(row.get("course_title", "")).lower()
    snippet = str(row.get("description", "")).lower()
    haystack = f"{title} {snippet}"
    goal = _terms(profile.get("career_goal", "")).lower()
    desired = _terms(profile.get("skills_to_learn", "")).lower()
    interests = _terms(profile.get("interests", "")).lower()
    current = _terms(profile.get("skills_you_have", "")).lower()
    level = _terms(profile.get("experience_level", "")).lower()
    budget = profile.get("budget")
    terms = []
    for source in [goal, desired, interests, current]:
        terms += [t for t in re.split(r"[,;\s]+", source) if len(t) > 2]
    unique = list(dict.fromkeys(terms))
    if not unique:
        lexical = 0.25
    else:
        lexical = sum(1 for t in unique if t in haystack) / len(unique)
    level_bonus = 0.08 if level and level in haystack else 0
    # Search results generally do not provide dependable price metadata.
    return round(min(0.98, 0.15 + lexical * 0.72 + level_bonus), 3)


def search_online_courses(profile: dict, max_per_provider: int = 2) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Search 30 provider domains. Returns courses, searched domains, provider domains with results."""
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("Missing package 'ddgs'. Run: python -m pip install -r requirements.txt") from exc

    topic = " ".join(filter(None, [
        _terms(profile.get("career_goal")),
        _terms(profile.get("skills_to_learn")),
        _terms(profile.get("interests")),
        _terms(profile.get("experience_level")),
        "online course training certification",
    ]))
    if not topic.strip():
        topic = "beginner technology online courses"

    def one_provider(provider):
        name, domain = provider
        query = f"site:{domain} {topic} course training"
        try:
            # A fresh client per worker avoids sharing a session across threads.
            with DDGS(timeout=18) as ddgs:
                hits = list(ddgs.text(query, max_results=max_per_provider, backend="auto"))
            rows = []
            for hit in hits:
                url = str(hit.get("href") or hit.get("url") or "").strip()
                title = re.sub(r"\s+", " ", str(hit.get("title") or "")).strip()
                snippet = re.sub(r"\s+", " ", str(hit.get("body") or hit.get("snippet") or "")).strip()
                if not url or not title or not url.startswith(("http://", "https://")):
                    continue
                if any(x in url.lower() for x in ["/login", "/signin", "/search?", "/tag/"]):
                    continue
                rows.append({"course_title": title, "platform": name, "url": url,
                             "description": snippet, "price": None, "level": "Not listed",
                             "content_duration": "Not listed", "source_domain": domain})
            return name, domain, rows, None
        except Exception as exc:
            return name, domain, [], str(exc)[:180]

    found, domains, errors = [], [], []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(one_provider, provider) for provider in PROVIDERS]
        for future in as_completed(futures):
            name, domain, rows, error = future.result()
            domains.append(domain)
            if rows:
                found.extend(rows)
            elif error:
                errors.append(f"{name}: {error}")

    # De-duplicate exact URLs and obvious generic pages.
    seen, unique = set(), []
    for row in found:
        key = row["url"].split("?")[0].rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        row["relevance_score"] = _score(row, profile)
        row["why"] = "Matched search terms from your career goal, desired skills, interests, and experience level; verify course details on the provider page."
        unique.append(row)
    unique.sort(key=lambda r: r["relevance_score"], reverse=True)
    for i, row in enumerate(unique, 1):
        row["rank"] = i
    return pd.DataFrame(unique), domains, errors
