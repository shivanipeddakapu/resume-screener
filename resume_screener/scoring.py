"""Screening: the LLM extracts facts once per resume, then every score is computed in Python.

`screen()` is the slow part (one LLM call per resume). `compute_fit()` is pure and instant, so the
UI can change weights and the shortlist threshold without calling the model again.
"""

import logging
import math
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from langchain_core.language_models import BaseChatModel

from .config import PROFILE_CHAR_BUDGET
from .ingest import Resume
from .prompts import facts_prompt, requirements_prompt
from .retrieval import ResumeIndex, cosine
from .schemas import CandidateFacts, JobRequirements

log = logging.getLogger(__name__)

COMPONENTS = ["Must-have", "Experience", "Relevance", "Education", "Nice-to-have"]
DEGREE_RANK = {"none": 0, "diploma": 1, "bachelor": 2, "master": 3, "doctorate": 4}
# Fields treated as equivalent for "Computer Science or equivalent"-style requirements
COMPUTING_WORDS = {"computer", "computing", "software", "information", "it", "data", "computational"}

# Whole-resume cosine similarity with nomic-embed-text (search_query / search_document prefixes)
RELEVANCE_FLOOR, RELEVANCE_CEIL = 0.55, 0.80


# ── text matching helpers ─────────────────────────────────────────────────────
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def mentions(text: str, term: str) -> bool:
    """Whole-word, case-insensitive match that copes with C++, C#, .NET and keeps Java out of JavaScript."""
    term = _norm(term)
    return bool(term) and re.search(rf"(?<![\w+#]){re.escape(term)}(?![\w+#])", _norm(text)) is not None


def skill_alternatives(skill: str) -> list[str]:
    """'cloud (AWS)' -> ['cloud (AWS)', 'cloud', 'AWS']; 'CI/CD' stays whole; 'Spark or Flink' -> both."""
    parts = [p.strip() for p in re.split(r"[()]|\bor\b|,", skill) if p.strip()]
    return list(dict.fromkeys([skill.strip(), *parts]))


def mentions_skill(text: str, skill: str) -> bool:
    return any(mentions(text, alt) for alt in skill_alternatives(skill))


def same_skill(a: str, b: str) -> bool:
    for x in skill_alternatives(a):
        for y in skill_alternatives(b):
            x_, y_ = _norm(x), _norm(y)
            if x_ and y_ and (x_ == y_ or mentions(x_, y_) or mentions(y_, x_)):
                return True
    return False


DEGREE_PATTERNS = [  # checked from highest to lowest; short forms like B.E. / M.S. need their dots
    ("doctorate", r"\b(ph\.?\s?d|doctorate)\b"),
    ("master", r"\b(m\.?\s?tech|m\.?\s?sc|m\.\s?s\b|mba|m\.\s?a\b|m\.?\s?com|mca)\b|(?<!scrum )\bmaster'?s?\b(?!\s*data)"),
    ("bachelor", r"\b(b\.?\s?tech|b\.\s?e\b|b\.?\s?sc|b\.?\s?com|b\.\s?a\b|bca)\b|\bbachelor'?s?\b"),
    ("diploma", r"\bdiploma\b"),
]


def degree_from_text(text: str) -> str:
    """Highest degree level written in the resume (backs up the LLM, which often misses abbreviations)."""
    lowered = text.lower()
    return next((level for level, pattern in DEGREE_PATTERNS if re.search(pattern, lowered)), "none")


def evidence_supported(evidence: str, resume_text: str, min_overlap: float = 0.6) -> bool:
    """True when most words of the LLM's quoted evidence really occur in the resume."""
    words = set(re.findall(r"[a-z0-9+#]+", evidence.lower()))
    if not words:
        return False
    resume_words = set(re.findall(r"[a-z0-9+#]+", resume_text.lower()))
    return len(words & resume_words) / len(words) >= min_overlap


def verified_skills(wanted: list[str], facts: CandidateFacts | None, resume_text: str) -> dict[str, str]:
    """Map each wanted skill the resume demonstrates to its evidence.

    A skill counts if it is written in the resume, or if the LLM claimed it with a quote that
    actually appears in the resume (this keeps "EC2 -> AWS" style inferences but drops hallucinations).
    """
    found: dict[str, str] = {}
    claims = facts.skills if facts else []
    for skill in wanted:
        quote = snippet(resume_text, skill)
        claim = next((c for c in claims if same_skill(c.skill, skill)), None)
        if quote:
            found[skill] = quote
        elif claim and evidence_supported(claim.evidence, resume_text):
            found[skill] = f"{claim.evidence.strip()} (inferred)"
    return found


def snippet(text: str, skill: str, before: int = 45, after: int = 70) -> str | None:
    """The words around the first mention of a skill in the resume, or None if it is not mentioned."""
    flat = re.sub(r"\s+", " ", text)
    for alt in skill_alternatives(skill):
        term = _norm(alt)
        if not term:
            continue
        m = re.search(rf"(?<![\w+#]){re.escape(term)}(?![\w+#])", flat.lower())
        if m:
            start, end = max(0, m.start() - before), min(len(flat), m.end() + after)
            start = flat.rfind(" ", 0, start) + 1 if start else 0
            end = flat.find(" ", end) if flat.find(" ", end) != -1 else len(flat)
            return ("…" if start else "") + flat[start:end].strip() + ("…" if end < len(flat) else "")
    return None


# ── component scores (0-100) ──────────────────────────────────────────────────
def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def coverage(found: dict[str, str], wanted: list[str]) -> float:
    """Share of `wanted` that was found; NaN when the job lists nothing (the component is then skipped)."""
    return 100.0 * sum(s in found for s in wanted) / len(wanted) if wanted else math.nan


def experience_score(years: float, min_years: float) -> float:
    """20 points for any profile, the rest scaled by how much of the required experience is there."""
    target = max(min_years, 1.0)
    return clamp(20 + 80 * min(max(years, 0.0) / target, 1.0))


def _computing(field: str) -> bool:
    return bool(set(re.findall(r"[a-z]+", field.lower())) & COMPUTING_WORDS)


def field_matches(candidate_field: str, wanted_fields: list[str]) -> bool:
    if not wanted_fields:
        return True
    cand_words = set(re.findall(r"[a-z]+", candidate_field.lower()))
    return any(
        (cand_words & set(re.findall(r"[a-z]+", f.lower())) - {"of", "and", "in", "or"})
        or (_computing(f) and _computing(candidate_field))
        for f in wanted_fields
    )


def education_score(degree: str, field: str, req: JobRequirements) -> float:
    have, need = DEGREE_RANK.get(degree, 0), DEGREE_RANK.get(req.degree, 0)
    if need == 0:
        level = 100.0 if have >= 1 else 70.0
    else:
        level = 100.0 if have >= need else 60.0 if have == need - 1 else 20.0
    return level if field_matches(field, req.degree_fields) else level * 0.75


def relevance_score(similarity: float) -> float:
    return clamp((similarity - RELEVANCE_FLOOR) / (RELEVANCE_CEIL - RELEVANCE_FLOOR) * 100)


# ── LLM steps ─────────────────────────────────────────────────────────────────
def looks_like_skill(item: str) -> bool:
    """Reject the fragments small models sometimes emit: echoed instructions, punctuation, sentences."""
    return bool(re.search(r"[A-Za-z]", item)) and ":" not in item and len(item.split()) <= 4


def clean_skills(items: list[str], limit: int) -> list[str]:
    out: list[str] = []
    for item in (i.strip(" .;-\"'") for i in items):
        if looks_like_skill(item) and _norm(item) not in {_norm(o) for o in out}:
            out.append(item)
    return out[:limit]


def analyse_job(llm: BaseChatModel, job_description: str) -> JobRequirements:
    req = llm.with_structured_output(JobRequirements, method="json_schema").invoke(requirements_prompt(job_description))
    req.must_have = clean_skills(req.must_have, 10)
    req.nice_to_have = [s for s in clean_skills(req.nice_to_have, 8) if not any(same_skill(s, m) for m in req.must_have)]
    req.min_years = max(0.0, float(req.min_years or 0))
    # the model often misses the degree line; read it from the posting as a fallback
    if req.degree == "none":
        req.degree = degree_from_text(job_description)
    if not req.degree_fields:
        req.degree_fields = degree_fields_from_text(job_description)
    return req


def degree_fields_from_text(text: str) -> list[str]:
    """'Bachelor's degree in Computer Science or equivalent' -> ['Computer Science']."""
    found = re.findall(r"degree in ([A-Za-z][A-Za-z &]{1,40}?)(?=\s+or\b|\s+and\b|[,.;)\n]|$)", text, flags=re.IGNORECASE)
    return list(dict.fromkeys(f.strip() for f in found))


def resume_view(index: ResumeIndex, resume: Resume, jd_vec: list[float]) -> str:
    """Whole resume if it is short; otherwise its chunks most similar to the job description."""
    if len(resume.text) <= PROFILE_CHAR_BUDGET:
        return resume.text
    chunks = [c for c in index.chunks if c.resume_id == resume.id]
    vecs = index.embedder.embed_documents([c.text for c in chunks])
    ranked = sorted(zip(chunks, vecs, strict=True), key=lambda cv: -cosine(jd_vec, cv[1]))
    picked, size = [], 0
    for chunk, _ in ranked:
        if size + len(chunk.text) > PROFILE_CHAR_BUDGET:
            break
        picked.append(chunk.text)
        size += len(chunk.text)
    return "\n…\n".join(picked)


def extract_facts(structured_llm, req: JobRequirements, text: str, retries: int = 2) -> CandidateFacts | None:
    for attempt in range(1, retries + 1):
        try:
            return structured_llm.invoke(facts_prompt(req, text))
        except Exception as e:  # malformed structured output or a dropped connection
            log.warning("Fact extraction failed (attempt %d/%d): %s", attempt, retries, e)
            time.sleep(attempt)
    return None


def candidate_row(resume: Resume, facts: CandidateFacts | None, req: JobRequirements, similarity: float) -> dict:
    found = verified_skills(req.must_have + req.nice_to_have, facts, resume.text)
    llm_degree, field = (facts.degree, facts.degree_field) if facts else ("none", "")
    degree = max(llm_degree, degree_from_text(resume.text), key=lambda d: DEGREE_RANK.get(d, 0))
    years = max(0.0, float(facts.years_experience)) if facts else 0.0
    relevant = min(years, max(0.0, float(facts.relevant_years))) if facts else 0.0
    return {
        "Candidate": (facts.name.strip() if facts else "") or Path(resume.name).stem.replace("_", " "),
        "Resume File": resume.name,
        "Years": years,
        "Relevant years": relevant,
        "Degree": f"{degree.title()}{' · ' + field if field else ''}" if degree != "none" else "—",
        "Matched": [s for s in req.must_have if s in found],
        "Missing": [s for s in req.must_have if s not in found],
        "Bonus": [s for s in req.nice_to_have if s in found],
        "Evidence": {s: found[s] for s in found},
        "Must-have": coverage(found, req.must_have),
        "Nice-to-have": coverage(found, req.nice_to_have),
        "Experience": experience_score(relevant, req.min_years) if facts else 0.0,
        "Education": education_score(degree, field, req),
        "Relevance": relevance_score(similarity),
        "Headline": facts.headline.strip() if facts else "Could not be profiled by the LLM; scored from text only.",
        "Strengths": [x.strip() for x in facts.strengths if x.strip()][:3] if facts else [],
        "Gaps": [x.strip() for x in facts.gaps if x.strip()][:3] if facts else [],
        "Profiled": facts is not None,
    }


def screen(
    index: ResumeIndex,
    llm: BaseChatModel,
    job_description: str,
    workers: int,
    progress: Callable[[int, int], None] = lambda done, total: None,
) -> tuple[pd.DataFrame, JobRequirements]:
    """Profile every resume against the job. Returns one row of component scores per candidate."""
    req = analyse_job(llm, job_description)
    jd_vec = index.embed_query(job_description)
    similarity = index.relevance(jd_vec)
    views = [resume_view(index, r, jd_vec) for r in index.resumes]

    structured = llm.with_structured_output(CandidateFacts, method="json_schema")
    facts: dict[int, CandidateFacts | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_facts, structured, req, view): i for i, view in enumerate(views)}
        for done, future in enumerate(as_completed(futures), start=1):
            facts[futures[future]] = future.result()
            progress(done, len(views))

    rows = [candidate_row(r, facts.get(i), req, similarity[r.id]) for i, r in enumerate(index.resumes)]
    return pd.DataFrame(rows), req


# ── fit score ─────────────────────────────────────────────────────────────────
def active_components(candidates: pd.DataFrame) -> list[str]:
    """Components that have a value for at least one candidate (e.g. no nice-to-haves -> skipped)."""
    return [c for c in COMPONENTS if c in candidates and candidates[c].notna().any()]


def weight_shares(weights: dict[str, float], active: list[str] = COMPONENTS) -> dict[str, float]:
    raw = {c: max(0.0, float(weights.get(c, 0))) if c in active else 0.0 for c in COMPONENTS}
    total = sum(raw.values())
    return {c: (v / total if total else (1 / len(active) if c in active else 0.0)) for c, v in raw.items()}


def compute_fit(candidates: pd.DataFrame, weights: dict[str, float], threshold: float) -> pd.DataFrame:
    """Weighted fit score, rank and shortlist flag. Pure and fast; never calls the LLM."""
    df = candidates.copy()
    for c in COMPONENTS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)  # None -> NaN, keeps the column numeric
    shares = weight_shares(weights, active_components(df))
    for c in COMPONENTS:
        df[f"pts:{c}"] = df[c].fillna(0.0) * shares[c]
    df["Fit"] = sum(df[f"pts:{c}"] for c in COMPONENTS).round(1)
    df = df.sort_values(["Fit", "Relevance"], ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    df["Shortlisted"] = df["Fit"] >= threshold
    return df
