"""LLM features on top of screening results: grounded Q&A and the hiring brief."""

import re
from collections import Counter

import pandas as pd
from langchain_core.language_models import BaseChatModel

from .prompts import answer_prompt, brief_prompt
from .retrieval import ResumeIndex, format_context
from .schemas import JobRequirements


def answer(question: str, index: ResumeIndex, llm: BaseChatModel) -> tuple[str, list[str]]:
    """Hybrid-retrieve resume chunks for the question and answer from them only."""
    chunks = index.search(question)
    reply = llm.invoke(answer_prompt(question, format_context(chunks))).content
    return tidy_bullets(reply), sorted({c.resume_name for c in chunks})


def tidy_bullets(text: str) -> str:
    """Turn '•'-style bullets (which Markdown ignores) into real list items on their own lines."""
    return re.sub(r"\s*[•●▪]\s*", "\n- ", text).strip()


def results_table(ranked: pd.DataFrame, top: int = 8) -> str:
    """Compact text table of the ranking that the brief prompt can read."""
    gaps = Counter(skill for missing in ranked["Missing"] for skill in missing)
    lines = [
        f"Candidates: {len(ranked)}; shortlisted: {int(ranked['Shortlisted'].sum())}",
        "Most common missing must-have: "
        + (f"{gaps.most_common(1)[0][0]} (missing for {gaps.most_common(1)[0][1]} of {len(ranked)})" if gaps else "none"),
        "",
        "rank | candidate | fit | years | matched must-haves | missing must-haves | headline",
    ]
    for _, r in ranked.head(top).iterrows():
        lines.append(
            f"{r['Rank']} | {r['Candidate']} | {r['Fit']:.0f} | {r['Years']:g} | "
            f"{', '.join(r['Matched']) or '-'} | {', '.join(r['Missing']) or '-'} | {r['Headline']}"
        )
    return "\n".join(lines)


def write_brief(llm: BaseChatModel, req: JobRequirements, ranked: pd.DataFrame, threshold: float) -> str:
    return llm.invoke(brief_prompt(req, results_table(ranked), threshold)).content
