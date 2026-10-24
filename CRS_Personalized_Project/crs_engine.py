# %% config
"""
crs_engine.py - Personalized Course Recommendation Engine (CRS enhancement)

This file is the engine code that also appears, cell by cell, in
Course_Recommendation_System_Enhanced.ipynb. The notebook is self-contained;
this module exists so the optional Streamlit app (app.py) can reuse the same code.
"""
import re
import difflib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# CONFIGURATION (everything a student may want to tune lives here)
# ----------------------------------------------------------------------------
# Weights are DESIGN CHOICES made by the developer, not values learned from data
# or scientifically validated. Change them and re-run to see the effect.
DEFAULT_WEIGHTS = {
    "interest_match":      0.25,  # title/subject text similarity to the student's interests
    "desired_skill_match": 0.30,  # similarity to skills the student WANTS to learn (strongest signal)
    "career_match":        0.10,  # similarity to keywords associated with the career goal
    "subject_match":       0.08,  # course subject is one of the subjects the student selected
    "level_fit":           0.12,  # how well the course level suits the preferred level
    "duration_fit":        0.05,  # course length vs. the time the student has (only if requested)
    "popularity":          0.10,  # log-scaled subscriber count (kept small on purpose)
    "known_skill_penalty": 0.10,  # subtracted when a course only repeats skills already known
}
WEIGHT_PURPOSE = {
    "interest_match":      "Rewards courses whose text overlaps with the student's interests.",
    "desired_skill_match": "Rewards courses about the skills the student wants to learn.",
    "career_match":        "Rewards courses matching keywords linked to the career goal (hand-written map).",
    "subject_match":       "Small bonus when the dataset's subject equals a subject the student picked.",
    "level_fit":           "1.0 exact level, 0.7 'All Levels', 0.3 adjacent level, 0 otherwise.",
    "duration_fit":        "1.0 if the course fits the time limit; decays if the limit is relaxed.",
    "popularity":          "Tie-breaker: log(subscribers) scaled 0-1. Small weight + relevance gate so it cannot dominate.",
    "known_skill_penalty": "Subtracted when the title matches current skills but NOT interests/desired skills.",
}

# Noise floor: a course must reach at least this text similarity (for at least one of
# interests / desired skills / career goal) to be considered relevant at all. This stops
# popularity or level from promoting courses that are unrelated to the student's topics.
MIN_RELEVANCE = 0.10

LEVEL_ORDER = ["Beginner Level", "Intermediate Level", "Expert Level"]   # + "All Levels"
LEVEL_RANK = {"Beginner Level": 0, "All Levels": 1, "Intermediate Level": 2, "Expert Level": 3}

# The 'subject' column is only trustworthy for Udemy rows (see the dataset audit in the notebook).
TRUSTED_METADATA_PLATFORMS = ("Udemy",)

# Optional constraints are relaxed in this order when too few courses survive.
# The budget is NEVER relaxed automatically.
RELAX_ORDER = ["platform", "duration", "subject", "level"]

# Short forms students commonly type. Each expansion keeps the original token as well.
SKILL_ALIASES = {
    "ml": "machine learning ml", "ai": "artificial intelligence ai", "dl": "deep learning",
    "nlp": "natural language processing nlp", "js": "javascript", "py": "python",
    "cpp": "c++", "powerbi": "power bi", "ds": "data science", "ux": "user experience ux",
    "ui": "user interface ui", "dsa": "data structures algorithms",
}

# Hand-written career -> keyword map. It is an ASSUMPTION made by the developer (not learned
# from data). Extend it freely. Unknown goals fall back to the words of the goal itself.
CAREER_KEYWORDS = {
    "data scientist": ["data science", "machine learning", "python", "statistics", "data analysis", "sql"],
    "data analyst": ["data analysis", "sql", "excel", "statistics", "tableau", "python"],
    "machine learning engineer": ["machine learning", "deep learning", "python", "neural networks", "tensorflow"],
    "ai engineer": ["artificial intelligence", "machine learning", "deep learning", "python", "neural networks"],
    "web developer": ["web development", "javascript", "html", "css", "react", "node"],
    "frontend developer": ["javascript", "html", "css", "react", "angular"],
    "full stack developer": ["web development", "javascript", "node", "react", "database", "html"],
    "software engineer": ["programming", "python", "java", "algorithms", "software"],
    "graphic designer": ["graphic design", "photoshop", "illustrator", "logo", "typography"],
    "financial analyst": ["finance", "financial modeling", "excel", "accounting", "valuation", "investing"],
    "business analyst": ["business analysis", "excel", "sql", "data analysis", "requirements"],
    "trader": ["trading", "stock", "options", "forex", "technical analysis"],
    "musician": ["guitar", "piano", "music theory", "singing", "songwriting"],
}


# %% text utilities
# Words that were fused together in the damaged edX titles (e.g. "withPython", "LinearModels").
# Splitting is applied ONLY to non-Udemy rows, and never to the well-known CamelCase brands below.
_KEEP_CAMEL = {
    "javascript", "typescript", "powerpoint", "wordpress", "youtube", "linkedin", "github", "tensorflow",
    "pytorch", "mysql", "postgresql", "mongodb", "iphone", "ipad", "ios", "macos", "ebay", "devops",
    "openai", "photoshop", "indesign", "autocad", "solidworks", "quickbooks", "sharepoint", "facebook",
    "nodejs", "jquery", "vmware", "matlab", "hadoop", "numpy", "scipy", "pandas",
}
_STOP = set(ENGLISH_STOP_WORDS)
TOKEN_PATTERN = r"(?u)\b\w[\w+#]*"          # keeps c++, c#, and one-letter names such as "r"


def _split_camel(text: str) -> str:
    """'Machine Learning withPython' -> 'Machine Learning with Python' (known brands untouched)."""
    def fix(match):
        word = match.group(0)
        if word.lower() in _KEEP_CAMEL:
            return word
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", word)
    return re.sub(r"[A-Za-z]+", fix, text)


def normalize_text(text, split_camel: bool = False) -> str:
    """Lower-case, drop punctuation (keeps + and #), collapse spaces. Safe for NaN."""
    if not isinstance(text, str):
        return ""
    if split_camel:
        text = _split_camel(text)
    text = text.lower().replace("\u2026", " ")
    text = re.sub(r"[^\w\s+#]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_phrases(raw) -> List[str]:
    """Accept 'Python, Data Science and AI' or ['Python', 'AI'] and return clean phrases."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        parts = [str(p) for p in raw]
    else:
        parts = re.split(r"[,;\n/|]+|\s+and\s+|\s+&\s+", str(raw), flags=re.IGNORECASE)
    return [p.strip() for p in parts if p and p.strip()]


def normalize_level(value) -> Optional[str]:
    """Map free text such as 'beginner' / 'advanced' to the dataset's level labels."""
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip().lower()
    if v.startswith("all") or "any" in v:
        return "All Levels"
    if any(k in v for k in ("begin", "basic", "novice", "fundamental")):
        return "Beginner Level"
    if "inter" in v:
        return "Intermediate Level"
    if any(k in v for k in ("adv", "expert")):
        return "Expert Level"
    return None


def parse_duration_hours(value) -> float:
    """'1.5 hours' -> 1.5, '34 mins' -> 0.567, anything else (e.g. '4 questions') -> NaN."""
    if not isinstance(value, str):
        return np.nan
    m = re.match(r"^\s*([\d.]+)\s*(hours?|mins?|minutes?)\s*$", value.strip().lower())
    if not m:
        return np.nan
    try:
        number = float(m.group(1))
    except ValueError:
        return np.nan
    return number if m.group(2).startswith("hour") else number / 60.0


# %% cleaning
def load_and_clean(path: str):
    """
    Read the course CSV and return (courses, cleaning_log).

    * The ORIGINAL file is never modified; cleaning happens on an in-memory copy.
    * Original columns are preserved (values are only type-converted); derived columns are added:
      price_num, is_paid, duration_hours, subject_for_matching, match_title, popularity_raw.
    * Every removal is recorded in cleaning_log so the process is auditable.
    """
    raw = pd.read_csv(path, dtype=str)
    log = [("rows read from file", len(raw))]
    df = raw.copy()

    # 1) columns that carry (almost) no data
    junk_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=junk_cols)
    log.append((f"empty/placeholder columns dropped: {junk_cols}", 0))

    # 2) rows that are not real courses (missing title, broken CSV quoting, header text such as 'hour')
    has_url = df["url"].fillna("").str.startswith(("http://", "https://"))
    has_title = df["course_title"].notna() & (df["course_title"].str.strip() != "")
    not_header_text = df["course_title"].fillna("").str.lower() != "hour"
    valid = has_url & has_title & not_header_text
    log.append(("malformed rows removed (no title / no valid URL / broken quoting)", int((~valid).sum())))
    df = df[valid].copy()

    # 3) numeric conversion (non-numeric text becomes NaN, never a guessed number)
    price_text = df["price"].astype(str).str.strip().str.lower()
    df["price_num"] = pd.to_numeric(df["price"], errors="coerce")
    df.loc[price_text.eq("free"), "price_num"] = 0.0
    for col in ["num_subscribers", "num_reviews", "num_lectures", "Rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["duration_hours"] = df["content_duration"].map(parse_duration_hours)

    # 4) categorical clean-up
    df["level"] = df["level"].where(df["level"].isin(LEVEL_ORDER + ["All Levels"]))
    df["subject"] = df["subject"].astype("object").map(lambda s: s.strip() if isinstance(s, str) else s)
    df["Platform"] = df["Platform"].fillna("Unknown")

    # 5) derived columns that can be justified from existing data
    df["is_paid"] = np.where(df["price_num"].isna(), pd.NA, df["price_num"] > 0)
    trusted = df["Platform"].isin(TRUSTED_METADATA_PLATFORMS)
    df["metadata_verified"] = trusted
    df["subject_for_matching"] = df["subject"].where(trusted)

    # 6) duplicates: same URL (or same platform+title) = same course; keep the row with most reviews
    before = len(df)
    df = df.sort_values("num_reviews", ascending=False, na_position="last", kind="mergesort")
    df = df.drop_duplicates(subset="url", keep="first")
    after_url = len(df)
    df = df.drop_duplicates(subset=["Platform", "course_title"], keep="first").sort_index()
    log.append(("duplicate courses removed (same URL)", before - after_url))
    log.append(("duplicate courses removed (same platform + title)", after_url - len(df)))

    # 7) text used for matching (original title stays untouched in 'course_title')
    df["match_title"] = [normalize_text(t, split_camel=(p not in TRUSTED_METADATA_PLATFORMS))
                         for t, p in zip(df["course_title"], df["Platform"])]
    df["popularity_raw"] = np.log1p(df["num_subscribers"].fillna(0))
    df = df.reset_index(drop=True)
    log.append(("courses available to the recommender", len(df)))
    return df, pd.DataFrame(log, columns=["step", "rows"])


# %% index
class CourseIndex:
    """
    Holds the cleaned courses and a sparse TF-IDF matrix (courses x terms).
    Similarity is computed on demand for a query, so no n x n course-by-course matrix is stored
    (that matrix would need roughly 1.6 GB for 14,000 courses).
    """

    def __init__(self, courses: pd.DataFrame, ngram_range=(1, 2)):
        self.courses = courses.reset_index(drop=True)
        # Text = cleaned title + subject (subject only where trustworthy).
        # Level is handled by its own score, so it is NOT put into the text (avoids double counting).
        text = self.courses["match_title"] + " " + self.courses["subject_for_matching"].map(normalize_text)
        self.vectorizer = TfidfVectorizer(
            token_pattern=TOKEN_PATTERN, stop_words=list(_STOP), ngram_range=ngram_range,
            sublinear_tf=True, dtype=np.float32,
        )
        self.matrix = self.vectorizer.fit_transform(text)          # sparse, L2-normalised rows
        terms = self.vectorizer.get_feature_names_out()
        self.unigrams = sorted(t for t in terms if " " not in t)
        self.unigram_set = set(self.unigrams)
        pop = self.courses["popularity_raw"].to_numpy(dtype=float)
        span = pop.max() - pop.min()
        self.popularity = (pop - pop.min()) / span if span > 0 else np.zeros_like(pop)
        self.titles_tokens = [set(t.split()) for t in self.courses["match_title"]]
        self.doc_tokens = [set(t.split()) for t in text]          # title + subject words (used for coverage checks)

    def similarity(self, phrases: List[str]) -> np.ndarray:
        """Cosine similarity between the phrases (joined) and every course -> array of length n."""
        if not phrases:
            return np.zeros(len(self.courses), dtype=np.float32)
        query = self.vectorizer.transform([" ".join(phrases)])
        return cosine_similarity(query, self.matrix).ravel()

    @property
    def subjects(self) -> List[str]:
        return sorted(self.courses["subject_for_matching"].dropna().unique())

    @property
    def platforms(self) -> List[str]:
        return sorted(self.courses["Platform"].dropna().unique())


def prepare_phrases(raw, index: CourseIndex):
    """
    Turn free text into clean search phrases.
    Returns (phrases, notes). Handles aliases ('ml'), small typos ('pyhton'), stop-words and NaN.
    """
    notes, phrases = [], []
    for item in split_phrases(raw):
        tokens_out = []
        for tok in normalize_text(item).split():
            if tok in SKILL_ALIASES:
                tokens_out.extend(SKILL_ALIASES[tok].split())
            elif tok in _STOP or tok in index.unigram_set or len(tok) < 4:
                tokens_out.append(tok)
            else:
                close = difflib.get_close_matches(tok, index.unigrams, n=1, cutoff=0.8)
                if close:
                    notes.append(f"'{tok}' was read as '{close[0]}'")
                    tokens_out.append(close[0])
                else:
                    tokens_out.append(tok)
        phrase = " ".join(tokens_out).strip()
        if phrase and any(t not in _STOP for t in phrase.split()):
            phrases.append(phrase)
    return phrases, notes


def phrase_coverage(index: CourseIndex, label: str, phrases: List[str]):
    """
    Tell the student how well each phrase is covered by the dataset.
    Returns (messages, n_phrases_with_full_match, n_phrases_checked).
    """
    msgs, full, checked = [], 0, 0
    for p in phrases:
        words = [t for t in p.split() if t not in _STOP]
        if not words:
            continue
        checked += 1
        unseen = [t for t in words if t not in index.unigram_set]
        n_full = sum(all(t in doc for t in words) for doc in index.doc_tokens)
        if unseen:
            msgs.append(f"No course contains the word(s) {', '.join(repr(t) for t in unseen)} (from {label} '{p}').")
        elif n_full == 0:
            msgs.append(f"No course contains all the words of {label} '{p}'; matches below use only some of its words.")
        elif n_full < 5:
            msgs.append(f"Only {n_full} course(s) contain all the words of {label} '{p}'.")
        if n_full > 0 and not unseen:
            full += 1
    return msgs, full, checked


def career_phrases(goal: str):
    """Return (keywords, found_in_map). Unknown goals fall back to the goal's own words."""
    key = normalize_text(goal)
    if not key:
        return [], False
    if key in CAREER_KEYWORDS:
        return list(CAREER_KEYWORDS[key]), True
    for known in CAREER_KEYWORDS:                       # 'aspiring data scientist' contains 'data scientist'
        if known in key:
            return list(CAREER_KEYWORDS[known]), True
    close = difflib.get_close_matches(key, list(CAREER_KEYWORDS), n=1, cutoff=0.75)
    if close:
        return list(CAREER_KEYWORDS[close[0]]), True
    return [key], False


# %% profile and result containers
@dataclass
class StudentProfile:
    """What the student tells the system. Lists or comma-separated strings are both accepted."""
    interests: Union[str, List[str]] = ""
    current_skills: Union[str, List[str]] = ""
    desired_skills: Union[str, List[str]] = ""
    level: Optional[str] = None                 # 'Beginner' | 'Intermediate' | 'Advanced' | None
    career_goal: str = ""
    max_budget: Optional[float] = None          # in the dataset's price units; None = no limit
    max_duration_hours: Optional[float] = None  # None = no limit
    subjects: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    strict_subject: bool = False                # True = ONLY courses of the chosen subjects
    name: str = "unnamed profile"


@dataclass
class RecommendationResult:
    table: pd.DataFrame
    messages: List[str]
    funnel: pd.DataFrame
    relaxed: List[str]
    breakdown: pd.DataFrame
    eligible_mask: np.ndarray
    profile: StudentProfile
    query_info: Dict


# %% scoring helpers
def level_fit_scores(course_levels: pd.Series, wanted: Optional[str]) -> np.ndarray:
    """1.0 exact | 0.7 'All Levels' | 0.3 adjacent | 0.0 otherwise or unknown."""
    out = np.zeros(len(course_levels))
    if wanted is None or wanted == "All Levels":
        return out
    w = LEVEL_ORDER.index(wanted)
    for i, lv in enumerate(course_levels):
        if lv == wanted:
            out[i] = 1.0
        elif lv == "All Levels":
            out[i] = 0.7
        elif lv in LEVEL_ORDER and abs(LEVEL_ORDER.index(lv) - w) == 1:
            out[i] = 0.3
    return out


def duration_fit_scores(hours: pd.Series, limit: Optional[float]) -> np.ndarray:
    """1.0 if within the limit, linearly decaying to 0 at twice the limit; unknown duration -> 0."""
    if not limit:
        return np.zeros(len(hours))
    h = hours.to_numpy(dtype=float)
    fit = np.where(h <= limit, 1.0, np.clip(1 - (h - limit) / limit, 0, 1))
    return np.nan_to_num(fit, nan=0.0)


def _build_masks(index: CourseIndex, profile: StudentProfile, wanted_level: Optional[str]) -> Dict[str, np.ndarray]:
    """One boolean mask per ACTIVE eligibility filter (True = course passes)."""
    c = index.courses
    masks: Dict[str, np.ndarray] = {}
    if profile.max_budget is not None:
        masks["budget"] = (c["price_num"] <= profile.max_budget).fillna(False).to_numpy()   # unknown price fails
    if wanted_level and wanted_level != "All Levels":
        masks["level"] = c["level"].isin([wanted_level, "All Levels"]).to_numpy()
    if profile.max_duration_hours:
        masks["duration"] = (c["duration_hours"] <= profile.max_duration_hours).fillna(False).to_numpy()
    if profile.subjects and profile.strict_subject:
        masks["subject"] = c["subject_for_matching"].isin(profile.subjects).to_numpy()
    if profile.platforms:
        masks["platform"] = c["Platform"].isin(profile.platforms).to_numpy()
    return masks


FILTER_TEXT = {
    "budget": "price within your maximum budget",
    "level": "level matches your preference (or 'All Levels')",
    "duration": "duration within your time limit",
    "subject": "subject is one of your chosen subjects",
    "platform": "platform is one of your chosen platforms",
}


def _funnel(relevant: np.ndarray, masks: Dict[str, np.ndarray], relaxed: List[str]) -> pd.DataFrame:
    rows = [("Courses relevant to your topics", int(relevant.sum()), int(relevant.sum()))]
    running = relevant.copy()
    for name, m in masks.items():
        alone = int((relevant & m).sum())
        if name not in relaxed:
            running &= m
        rows.append((f"+ filter: {FILTER_TEXT[name]}" + (" (relaxed)" if name in relaxed else ""),
                     int(running.sum()), alone))
    return pd.DataFrame(rows, columns=["step", "courses_left", "left_if_this_filter_alone"])


# %% recommend
def recommend(index: CourseIndex, profile: StudentProfile, top_k: int = 10,
              weights: Optional[Dict[str, float]] = None, relax: Optional[List[str]] = None,
              auto_relax: bool = True, min_relevance: float = MIN_RELEVANCE,
              allow_popular_fallback: bool = False) -> RecommendationResult:
    """
    Personalized recommendation:
      1) turn the student's text into search phrases (aliases, typo repair)
      2) TF-IDF cosine similarity to interests / desired skills / career keywords / current skills
      3) relevance gate -> eligibility filters (budget is mandatory) -> weighted score -> top_k
      4) explanations are generated from the conditions that were actually satisfied
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    c = index.courses
    n = len(c)
    messages: List[str] = []
    relaxed: List[str] = list(relax or [])

    # ---- 1) understand the input -------------------------------------------------------
    int_ph, n1 = prepare_phrases(profile.interests, index)
    des_ph, n2 = prepare_phrases(profile.desired_skills, index)
    cur_ph, n3 = prepare_phrases(profile.current_skills, index)
    career_kw, career_known = career_phrases(profile.career_goal)
    car_ph, n4 = prepare_phrases(career_kw, index)
    messages += [f"Spelling/alias note: {x}." for x in n1 + n2 + n3 + n4]
    if profile.career_goal and not career_known:
        messages.append(f"Career goal '{profile.career_goal}' has no predefined keyword profile; "
                        "its own words were used for matching.")
    wanted_level = normalize_level(profile.level)
    if profile.level and wanted_level is None:
        messages.append(f"Level '{profile.level}' was not recognised (use Beginner, Intermediate or Advanced); level ignored.")

    # how well does the dataset cover each topic? (clear messages instead of silently misleading results)
    cov1, f1, c1 = phrase_coverage(index, "interest", int_ph)
    cov2, f2, c2 = phrase_coverage(index, "desired skill", des_ph)
    messages += cov1 + cov2
    if c1 + c2 > 0 and f1 + f2 == 0:
        messages.append("None of your topics match any course completely, so any results below are only loosely related.")

    # ---- 2) similarity components ------------------------------------------------------
    comp = {}
    if int_ph:
        comp["interest_match"] = index.similarity(int_ph)
    if des_ph:
        comp["desired_skill_match"] = index.similarity(des_ph)
    if car_ph:
        comp["career_match"] = index.similarity(car_ph)
    sim_current = index.similarity(cur_ph) if cur_ph else None

    empty_table = pd.DataFrame()
    if not comp:
        messages.append("No interests, desired skills or career goal were given, so the recommendations "
                        "cannot be personalised.")
        if allow_popular_fallback:
            masks = _build_masks(index, profile, wanted_level)
            ok = np.ones(n, dtype=bool)
            for m in masks.values():
                ok &= m
            order = np.argsort(-index.popularity * ok)[:top_k]
            order = [i for i in order if ok[i]]
            table = _format_table(index, order, np.zeros(n), ["Most popular course among those that pass your filters "
                                                              "(NOT personalised)."] * len(order))
            messages.append("Showing the most popular courses that pass your filters instead (not personalised).")
            return RecommendationResult(table, messages, pd.DataFrame(), relaxed, pd.DataFrame(), ok, profile, {})
        return RecommendationResult(empty_table, messages, pd.DataFrame(), relaxed, pd.DataFrame(),
                                    np.zeros(n, dtype=bool), profile, {})

    text_max = np.max(np.vstack(list(comp.values())), axis=0)
    relevant = text_max >= min_relevance

    # ---- 3) eligibility filters (budget mandatory; the rest optional and relaxable) ----------
    masks = _build_masks(index, profile, wanted_level)

    def eligible_now():
        ok = relevant.copy()
        for name, m in masks.items():
            if name not in relaxed:
                ok &= m
        return ok

    ok = eligible_now()
    if ok.sum() < top_k and auto_relax:
        for name in RELAX_ORDER:
            if ok.sum() >= top_k:
                break
            if name in masks and name not in relaxed:
                relaxed.append(name)
                ok = eligible_now()
        if relaxed:
            messages.append("Too few courses matched every filter, so these optional filters were relaxed: "
                            + ", ".join(relaxed) + ". The budget filter was kept.")
    funnel = _funnel(relevant, masks, relaxed)
    if ok.sum() == 0:
        limiting = funnel.iloc[1:].sort_values("left_if_this_filter_alone")
        if relevant.sum() == 0:
            messages.append("No course is similar enough to your topics. Try broader or different keywords.")
        elif len(limiting):
            row = limiting.iloc[0]
            messages.append("No course satisfies all constraints. The most limiting filter is "
                            f"'{row['step'].replace('+ filter: ', '')}' (only {row['left_if_this_filter_alone']} "
                            "relevant courses pass it alone). Consider raising the budget or relaxing optional filters.")
        messages = [m for m in messages if not m.startswith("None of your topics")]
        return RecommendationResult(empty_table, messages, funnel, relaxed, pd.DataFrame(), ok, profile, {})
    if 0 < ok.sum() < top_k:
        messages.append(f"Only {int(ok.sum())} courses satisfy your constraints (you asked for {top_k}).")

    # ---- 4) score ------------------------------------------------------------------
    comp["subject_match"] = (c["subject_for_matching"].isin(profile.subjects).to_numpy().astype(float)
                             if profile.subjects else None)
    comp["level_fit"] = level_fit_scores(c["level"], wanted_level) if wanted_level and wanted_level != "All Levels" else None
    comp["duration_fit"] = duration_fit_scores(c["duration_hours"], profile.max_duration_hours) if profile.max_duration_hours else None
    comp["popularity"] = index.popularity
    active = {k: v for k, v in comp.items() if v is not None}
    total_w = sum(w[k] for k in active)
    contrib = {k: w[k] * v / total_w for k, v in active.items()}
    score = np.sum(list(contrib.values()), axis=0)
    if sim_current is not None:
        topical = np.max(np.vstack([comp[k] for k in ("interest_match", "desired_skill_match", "career_match") if k in comp]), axis=0)
        known_only = np.clip(sim_current - topical, 0, None)
        contrib["known_skill_penalty"] = -w["known_skill_penalty"] * known_only
        score = score + contrib["known_skill_penalty"]
    score = np.clip(score, 0, 1)

    # ---- 5) rank + explain -------------------------------------------------------------
    cand = np.where(ok)[0]
    order = cand[np.lexsort((c["course_title"].to_numpy()[cand], -index.popularity[cand], -score[cand]))][:top_k]
    reasons = [_explain(index, i, profile, wanted_level, relaxed, int_ph, des_ph, car_ph, cur_ph,
                        comp, score, career_known) for i in order]
    table = _format_table(index, order, score, reasons)
    breakdown = pd.DataFrame({k: [v[i] for i in order] for k, v in contrib.items()},
                             index=table["course_title"].str.slice(0, 45).values)
    info = {"interest_phrases": int_ph, "desired_phrases": des_ph, "current_phrases": cur_ph,
            "career_phrases": car_ph, "weights_used": {k: round(w[k] / total_w, 3) for k in active},
            "level": wanted_level}
    return RecommendationResult(table, messages, funnel, relaxed, breakdown, ok, profile, info)


def _format_table(index, order, score, reasons) -> pd.DataFrame:
    c = index.courses
    rows = []
    for rank, (i, why) in enumerate(zip(order, reasons), start=1):
        r = c.iloc[i]
        paid = r["is_paid"]
        rows.append({
            "rank": rank,
            "course_title": r["course_title"],
            "platform": r["Platform"],
            "subject": r["subject"] if (isinstance(r["subject"], str) and r["metadata_verified"]) else "Not available",
            "level": r["level"] if isinstance(r["level"], str) else "Not available",
            "price": r["price_num"],
            "pricing": "Unknown" if pd.isna(paid) else ("Paid" if paid else "Free"),
            "num_subscribers": r["num_subscribers"],
            "num_reviews": r["num_reviews"],
            "content_duration": r["content_duration"] if isinstance(r["content_duration"], str) else "Not available",
            "url": r["url"],
            "relevance_score": round(float(score[i]), 3),
            "why": " ".join(why) if isinstance(why, list) else why,
        })
    return pd.DataFrame(rows)


def _explain(index, i, profile, wanted_level, relaxed, int_ph, des_ph, car_ph, cur_ph,
             comp, score, career_known) -> List[str]:
    """Reasons come only from conditions that were actually checked for this course."""
    c = index.courses
    r = c.iloc[i]
    toks = index.titles_tokens[i]
    why = []

    padded = f" {r['match_title']} "

    def found(phrases):
        """Whole phrases found in the title (e.g. 'machine learning'); otherwise single matching words."""
        hits, covered = [], set()
        for p in phrases:
            if f" {p} " in padded:
                hits.append(p)
                covered.update(p.split())
        loose = {t for p in phrases for t in p.split() if t not in _STOP} & toks - covered
        return hits + sorted(loose)

    t = found(des_ph)
    if t:
        why.append(f"Title contains your desired-skill terms: {', '.join(t)}.")
    t = found(int_ph)
    if t:
        why.append(f"Title matches your interests: {', '.join(t)}.")
    t = found(car_ph)
    if t and profile.career_goal:
        src = "keywords linked to" if career_known else "the words of"
        why.append(f"Title matches {src} your career goal '{profile.career_goal}': {', '.join(t)}.")
    if profile.subjects and r["subject_for_matching"] in profile.subjects:
        why.append(f"Subject '{r['subject_for_matching']}' is one you selected.")
    if wanted_level and wanted_level != "All Levels":
        if r["level"] == wanted_level:
            why.append(f"Level '{r['level']}' matches your preferred level.")
        elif r["level"] == "All Levels":
            why.append("Listed for 'All Levels', so it suits your level.")
        elif isinstance(r["level"], str):
            why.append(f"Level is '{r['level']}', which differs from your preferred level (filter relaxed).")
    if profile.max_budget is not None and not pd.isna(r["price_num"]):
        why.append("Free." if r["price_num"] == 0 else f"Price {r['price_num']:g} is within your budget of {profile.max_budget:g}.")
    if profile.max_duration_hours and not pd.isna(r["duration_hours"]):
        within = r["duration_hours"] <= profile.max_duration_hours
        why.append(f"Duration {r['content_duration']} {'fits' if within else 'exceeds (filter relaxed)'} your {profile.max_duration_hours:g} h limit.")
    if cur_ph:
        cur_terms = {t for p in cur_ph for t in p.split() if t not in _STOP}
        overlap = sorted(cur_terms & toks)
        topical = {t for p in int_ph + des_ph + car_ph for t in p.split() if t not in _STOP} & toks
        if overlap and not topical:
            why.append(f"Note: it mostly repeats skills you already have ({', '.join(overlap)}); score reduced.")
    if not pd.isna(r["num_subscribers"]) and r["num_subscribers"] > 0:
        why.append(f"Popularity signal (minor factor): {int(r['num_subscribers']):,} subscribers.")
    if not r["metadata_verified"]:
        why.append(f"Caution: {r['Platform']} rows have unverified price/level/duration/subject metadata in this dataset.")
    return why


# %% learning path
PATH_WEIGHTS = {"skill_similarity": 1.0, "level_fit": 0.15, "popularity": 0.05}


def learning_path(index: CourseIndex, result: RecommendationResult, max_steps: int = 5,
                  min_relevance: float = MIN_RELEVANCE) -> pd.DataFrame:
    """
    Suggested learning sequence. One course per skill the student wants to learn (or per interest if
    no desired skills were given), chosen only from courses that passed the same eligibility filters.
    Steps are ordered by the course's level label (Beginner -> All Levels -> Intermediate -> Expert),
    then by the order in which the student listed the skills.
    THIS IS A SUGGESTION: prerequisites and exact skill coverage are NOT verified by the dataset.
    """
    info = result.query_info
    if not info:
        return pd.DataFrame()
    skills = info["desired_phrases"] or info["interest_phrases"]
    wanted_level = info["level"]
    c = index.courses
    lvl = level_fit_scores(c["level"], wanted_level)
    used, steps = set(), []
    for pos, skill in enumerate(skills[:max_steps]):
        sim = index.similarity([skill])
        s = (PATH_WEIGHTS["skill_similarity"] * sim + PATH_WEIGHTS["level_fit"] * lvl
             + PATH_WEIGHTS["popularity"] * index.popularity)
        s = np.where(result.eligible_mask & (sim >= min_relevance), s, -1)
        for u in used:
            s[u] = -1
        best = int(np.argmax(s))
        if s[best] < 0:
            steps.append({"skill": skill, "order": pos, "level_rank": 9, "course_title": "(no suitable course found)"})
            continue
        used.add(best)
        r = c.iloc[best]
        steps.append({"skill": skill, "order": pos, "level_rank": LEVEL_RANK.get(r["level"], 1.5),
                      "course_title": r["course_title"], "level": r["level"] if isinstance(r["level"], str) else "Not available",
                      "platform": r["Platform"], "price": r["price_num"], "content_duration": r["content_duration"],
                      "url": r["url"]})
    path = pd.DataFrame(steps).sort_values(["level_rank", "order"]).reset_index(drop=True)
    path.insert(0, "step", range(1, len(path) + 1))
    return path.drop(columns=["order", "level_rank"])


# %% evaluation metrics
def precision_at_k(recommended: List, relevant: set, k: int) -> float:
    """Share of the top-k recommendations that are relevant. Needs real relevance labels."""
    top = list(recommended)[:k]
    return sum(1 for x in top if x in relevant) / k if k else 0.0


def recall_at_k(recommended: List, relevant: set, k: int) -> float:
    """Share of all relevant courses that appear in the top-k. Needs real relevance labels."""
    if not relevant:
        return 0.0
    return sum(1 for x in list(recommended)[:k] if x in relevant) / len(relevant)
