"""Every chart must build from realistic results, including a job with no nice-to-have skills."""

import math

import pandas as pd
import pytest

from resume_screener import charts
from resume_screener.schemas import JobRequirements
from resume_screener.scoring import COMPONENTS, compute_fit, weight_shares

REQ = JobRequirements(title="SDE", must_have=["Java", "Python"], nice_to_have=[], min_years=1)


@pytest.fixture
def ranked():
    rows = [
        {"Candidate": "Ann", "Years": 3.0, "Matched": ["Java"], "Missing": ["Python"], "Bonus": [],
         "Must-have": 50.0, "Experience": 100.0, "Relevance": 80.0, "Education": 100.0, "Nice-to-have": math.nan},
        {"Candidate": "Bob", "Years": 0.0, "Matched": [], "Missing": ["Java", "Python"], "Bonus": [],
         "Must-have": 0.0, "Experience": 20.0, "Relevance": 10.0, "Education": 75.0, "Nice-to-have": math.nan},
    ]  # fmt: skip
    return compute_fit(pd.DataFrame(rows), dict.fromkeys(COMPONENTS, 1), threshold=60)


def test_all_charts_build(ranked):
    shares = weight_shares(dict.fromkeys(COMPONENTS, 1), ["Must-have", "Experience", "Relevance", "Education"])
    figures = [
        charts.fit_breakdown(ranked, 60),
        charts.skills_matrix(ranked, REQ),
        charts.experience_vs_relevance(ranked),
        charts.component_bars(ranked.iloc[0], shares),
        charts.compare_bars([ranked.iloc[0], ranked.iloc[1]]),
    ]
    for fig in figures:
        fig.to_dict()  # forces plotly to validate every property
    bars = figures[3].data[0]
    assert "n/a" in bars.text  # skipped component is labelled, not drawn as NaN
