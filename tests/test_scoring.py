import math

import pandas as pd
import pytest

from resume_screener.ingest import Resume
from resume_screener.schemas import CandidateFacts, JobRequirements, SkillEvidence
from resume_screener.scoring import (
    COMPONENTS,
    candidate_row,
    clean_skills,
    compute_fit,
    coverage,
    degree_from_text,
    education_score,
    evidence_supported,
    experience_score,
    mentions,
    relevance_score,
    same_skill,
    skill_alternatives,
    verified_skills,
    weight_shares,
)

RESUME_TEXT = "Built order services in Core Java and Kafka. Deployed on AWS EC2. Some JavaScript. C++ for latency."
REQ = JobRequirements(
    title="SDE", must_have=["Java", "Python", "C++"], nice_to_have=["Kafka"], min_years=2,
    degree="bachelor", degree_fields=["Computer Science"],
)  # fmt: skip


@pytest.mark.parametrize(
    ("text", "term", "expected"),
    [
        ("Core Java developer", "Java", True),
        ("JavaScript only", "Java", False),
        ("C++ and Rust", "C++", True),
        ("C++ and Rust", "C", False),
        ("C# / .NET", "C#", True),
        ("Google Ads", "Go", False),
        ("anything", "", False),
    ],
)
def test_mentions(text, term, expected):
    assert mentions(text, term) is expected


def test_same_skill():
    assert same_skill("AWS", "aws (ec2)")
    assert not same_skill("Java", "JavaScript")


def test_evidence_must_really_appear_in_resume():
    assert evidence_supported("Deployed on AWS EC2", RESUME_TEXT)
    assert not evidence_supported("Led a team of 40 Python engineers at Google", RESUME_TEXT)
    assert not evidence_supported("", RESUME_TEXT)


def test_verified_skills_drops_hallucinations_and_keeps_text_mentions():
    facts = CandidateFacts(
        name="A", headline="h", years_experience=3, relevant_years=3, degree="bachelor", degree_field="CS",
        skills=[
            SkillEvidence(skill="Python", evidence="Wrote Python ML pipelines at Meta"),  # not in resume
            SkillEvidence(skill="Kafka", evidence="order services in Core Java and Kafka"),
        ],
    )  # fmt: skip
    found = verified_skills(["Java", "Python", "C++", "Kafka"], facts, RESUME_TEXT)
    assert set(found) == {"Java", "C++", "Kafka"}
    assert "Core Java" in found["Java"]  # evidence is the real sentence from the resume


def test_inferred_skill_needs_evidence_that_exists():
    def facts_with(evidence):
        return CandidateFacts(
            name="A", headline="h", years_experience=1, relevant_years=1, degree="bachelor", degree_field="CS",
            skills=[SkillEvidence(skill="Cloud", evidence=evidence)],
        )  # fmt: skip

    assert verified_skills(["Cloud"], facts_with("Deployed on AWS EC2"), RESUME_TEXT) == {
        "Cloud": "Deployed on AWS EC2 (inferred)"
    }
    assert verified_skills(["Cloud"], facts_with("Ran GCP clusters"), RESUME_TEXT) == {}


def test_component_scores():
    assert coverage({"a": ""}, ["a", "b"]) == 50
    assert math.isnan(coverage({}, []))  # nothing asked for -> component skipped
    assert experience_score(0, 2) == 20
    assert experience_score(1, 2) == 60
    assert experience_score(5, 2) == 100
    assert experience_score(3, 0) == 100
    assert relevance_score(0.1) == 0
    assert relevance_score(0.9) == 100


@pytest.mark.parametrize(
    ("degree", "field", "expected"),
    [
        ("bachelor", "Computer Science", 100),
        ("master", "Information Technology", 100),  # computing fields count as equivalent
        ("bachelor", "Commerce", 75),
        ("diploma", "Computer Science", 60),
        ("none", "", 15),
    ],
)
def test_education_score(degree, field, expected):
    assert education_score(degree, field, REQ) == expected


def test_candidate_row_without_llm_facts_falls_back_to_text():
    row = candidate_row(Resume(id="r0", name="Jane_Doe.pdf", text=RESUME_TEXT), None, REQ, similarity=0.675)
    assert row["Candidate"] == "Jane Doe"
    assert row["Matched"] == ["Java", "C++"]
    assert row["Missing"] == ["Python"]
    assert row["Bonus"] == ["Kafka"]
    assert row["Experience"] == 0 and row["Profiled"] is False
    assert row["Relevance"] == pytest.approx(50)


def test_weight_shares():
    shares = weight_shares({"Must-have": 50, "Experience": 50})
    assert shares["Must-have"] == 0.5 and shares["Relevance"] == 0
    assert weight_shares({}) == {c: 1 / len(COMPONENTS) for c in COMPONENTS}


def frame():
    base = dict.fromkeys(COMPONENTS, 0.0)
    return pd.DataFrame(
        [
            {"Candidate": "Low", **base, "Must-have": 20.0},
            {"Candidate": "High", **dict.fromkeys(COMPONENTS, 90.0)},
            {"Candidate": "Mid", **dict.fromkeys(COMPONENTS, 60.0)},
        ]
    )


def test_compute_fit_ranks_and_shortlists_without_mutating_input():
    candidates = frame()
    df = compute_fit(candidates, dict.fromkeys(COMPONENTS, 1), threshold=60)
    assert list(df["Candidate"]) == ["High", "Mid", "Low"]
    assert list(df["Rank"]) == [1, 2, 3]
    assert list(df["Fit"]) == [90, 60, 4]
    assert list(df["Shortlisted"]) == [True, True, False]
    assert df.loc[0, [f"pts:{c}" for c in COMPONENTS]].sum() == pytest.approx(90)
    assert "Fit" not in candidates.columns


def test_compute_fit_follows_weights():
    df = compute_fit(frame(), {"Must-have": 1}, threshold=50)
    assert df.set_index("Candidate").loc["Low", "Fit"] == 20


def test_skill_alternatives_and_compound_requirements():
    assert skill_alternatives("cloud (AWS)") == ["cloud (AWS)", "cloud", "AWS"]
    assert skill_alternatives("Spark or Flink") == ["Spark or Flink", "Spark", "Flink"]
    assert skill_alternatives("CI/CD") == ["CI/CD"]
    assert same_skill("AWS", "cloud (AWS)")
    assert "cloud (AWS)" in verified_skills(["cloud (AWS)"], None, RESUME_TEXT)


def test_clean_skills_drops_echoed_instructions_and_duplicates():
    raw = ["concrete skills: Java", ":[", "Python", "python", "a whole sentence about many things", "C++"]
    assert clean_skills(raw, 10) == ["Python", "C++"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("B.Com with Business Analytics", "bachelor"),
        ("B.E. in Computer Science", "bachelor"),
        ("MBA in Marketing", "master"),
        ("M.S. in Data Science", "master"),
        ("PhD in Physics", "doctorate"),
        ("Diploma in IT", "diploma"),
        ("I will be a great fit", "none"),  # "be" is not B.E.
        ("p99 latency under 10 ms", "none"),  # "ms" is not M.S.
        ("Certified Scrum Master", "none"),
        ("Led master data management", "none"),
    ],
)
def test_degree_from_text(text, expected):
    assert degree_from_text(text) == expected


def test_experience_uses_relevant_years_and_degree_falls_back_to_text():
    facts = CandidateFacts(
        name="Mia", years_experience=6, relevant_years=0, skills=[], degree="none", degree_field="Marketing",
        headline="Marketer",
    )  # fmt: skip
    row = candidate_row(Resume("r1", "mia.pdf", "MBA in Marketing. SEO lead for 6 years."), facts, REQ, 0.6)
    assert row["Years"] == 6 and row["Relevant years"] == 0
    assert row["Experience"] == 20  # an unrelated career earns only the base score
    assert row["Degree"].startswith("Master")  # MBA found in the text even though the LLM said "none"


def test_components_without_data_are_skipped():
    candidates = frame()
    candidates["Nice-to-have"] = None  # what pandas builds when every candidate has no value
    df = compute_fit(candidates, dict.fromkeys(COMPONENTS, 1), threshold=60)
    assert df["Fit"].dtype == float and df["Nice-to-have"].dtype == float  # regression: object dtype broke charts
    assert df.set_index("Candidate").loc["High", "Fit"] == 90  # not diluted by the empty component
    assert weight_shares(dict.fromkeys(COMPONENTS, 1), ["Must-have", "Experience"])["Must-have"] == 0.5


def test_degree_fields_from_job_text():
    from resume_screener.scoring import degree_fields_from_text

    assert degree_fields_from_text("Bachelor's degree in Computer Science or equivalent") == ["Computer Science"]
    assert degree_fields_from_text("No degree needed") == []
