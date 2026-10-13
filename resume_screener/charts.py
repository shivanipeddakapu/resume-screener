"""Plotly figures sharing one visual style (colours mirror the CSS tokens in ui.py)."""

import pandas as pd
import plotly.graph_objects as go

from .schemas import JobRequirements
from .scoring import COMPONENTS

INK, MUTED, GRID = "#0f172a", "#64748b", "#e2e8f0"
DANGER = "#b91c1c"
COMPONENT_COLORS = dict(zip(COMPONENTS, ["#2f5bd3", "#0d9488", "#7c3aed", "#c2410c", "#64748b"], strict=True))
SERIES = ["#2f5bd3", "#0d9488", "#c2410c"]
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def _style(fig: go.Figure, height: int, title: str | None = None, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=48 if title else 12, b=8),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=MUTED, size=12),
        hoverlabel=dict(font=dict(family=FONT), bgcolor="white", bordercolor=GRID),
        showlegend=legend,
        legend=dict(orientation="h", y=-0.12, x=0, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def _wrap(label: str, width: int = 12) -> str:
    """Break long axis labels onto several lines so neighbouring columns don't collide."""
    lines, line = [], ""
    for word in label.split():
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return "<br>".join([*lines, line])


def fit_breakdown(df: pd.DataFrame, threshold: float, n: int = 12) -> go.Figure:
    """Ranking as stacked bars: how many fit points each component contributes."""
    d = df.head(n).iloc[::-1]
    labels = [f"{'★ ' if s else ''}{c}" for c, s in zip(d["Candidate"], d["Shortlisted"], strict=True)]
    fig = go.Figure()
    for comp in [c for c in COMPONENTS if d[c].notna().any()]:  # skipped components get no bar or legend
        fig.add_bar(
            y=labels,
            x=d[f"pts:{comp}"],
            name=comp,
            orientation="h",
            marker_color=COMPONENT_COLORS[comp],
            customdata=d[comp],
            hovertemplate=f"{comp}: %{{customdata:.0f}}/100 → %{{x:.1f}} pts<extra></extra>",
        )
    fig.add_scatter(
        y=labels,
        x=d["Fit"],
        mode="text",
        text=d["Fit"].map("{:.0f}".format),
        textposition="middle right",
        textfont=dict(color=INK, size=12),
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_vline(x=threshold, line_dash="dot", line_color=DANGER, line_width=1.5)
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_xaxes(range=[0, 108], title=None)
    fig.update_yaxes(title=None, gridcolor="rgba(0,0,0,0)")
    return _style(fig, max(300, 36 * len(d) + 110), "Fit score breakdown  ·  ★ shortlisted", legend=True)


def skills_matrix(df: pd.DataFrame, req: JobRequirements, n: int = 12) -> go.Figure:
    """Candidates × required skills: which must-haves (and nice-to-haves) each resume demonstrates."""
    d = df.head(n).iloc[::-1]
    skills = req.must_have + req.nice_to_have
    z = [[1 if s in (row["Matched"] + row["Bonus"]) else 0 for s in skills] for _, row in d.iterrows()]
    text = [["✓" if v else "·" for v in r] for r in z]
    cols = [_wrap(s if s in req.must_have else f"{s} (+)") for s in skills]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=cols,
            y=list(d["Candidate"]),
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=14),
            colorscale=[[0, "#f1f5f9"], [1, "#2f5bd3"]],
            zmin=0,
            zmax=1,
            showscale=False,
            xgap=3,
            ygap=3,
            hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
        )
    )
    fig.update_xaxes(side="bottom", tickangle=0, gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    title = "Skills matrix" + ("  ·  (+) nice-to-have" if req.nice_to_have else "")
    return _style(fig, max(280, 34 * len(d) + 120), title)


def experience_vs_relevance(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for shortlisted, color, name in ((True, "#2f5bd3", "Shortlisted"), (False, "#94a3b8", "Not shortlisted")):
        d = df[df["Shortlisted"] == shortlisted]
        fig.add_scatter(
            x=d["Years"],
            y=d["Relevance"],
            mode="markers+text",
            name=name,
            text=d["Candidate"],
            textposition="top center",
            textfont=dict(size=11, color=INK),
            marker=dict(size=10 + d["Fit"] / 5, color=color, line=dict(color="white", width=1.5)),
            customdata=d["Fit"],
            hovertemplate="%{text}<br>%{x:g} yrs · relevance %{y:.0f} · fit %{customdata:.0f}<extra></extra>",
        )
    top_years = float(df["Years"].max()) if len(df) else 1.0
    fig.update_xaxes(title="Years of experience", range=[-0.6, max(top_years, 1.0) * 1.15 + 0.6])
    fig.update_yaxes(title="Relevance to the job", range=[-8, 122])  # headroom for the name labels
    fig = _style(fig, 360, "Experience vs. relevance", legend=True)
    fig.update_layout(legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig


def component_bars(row: pd.Series, shares: dict[str, float]) -> go.Figure:
    comps = COMPONENTS[::-1]
    fig = go.Figure(
        go.Bar(
            x=[0 if pd.isna(row[c]) else row[c] for c in comps],
            y=[f"{c} · {shares[c]:.0%}" for c in comps],
            orientation="h",
            marker_color=[COMPONENT_COLORS[c] for c in comps],
            text=["n/a" if pd.isna(row[c]) else f"{row[c]:.0f}" for c in comps],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.0f}/100<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 112], title=None)
    fig.update_yaxes(title=None, gridcolor="rgba(0,0,0,0)")
    return _style(fig, 290, "Score by component  ·  weight")


def compare_bars(rows: list[pd.Series]) -> go.Figure:
    comps = [c for c in COMPONENTS if any(pd.notna(r[c]) for r in rows)]
    fig = go.Figure()
    for i, r in enumerate(rows):
        fig.add_bar(
            x=comps + ["Fit"],
            y=[0 if pd.isna(r[c]) else r[c] for c in comps] + [r["Fit"]],
            name=f"#{r['Rank']} {r['Candidate']}",
            marker_color=SERIES[i % len(SERIES)],
            hovertemplate="%{x}: %{y:.0f}<extra>%{fullData.name}</extra>",
        )
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(range=[0, 105], title=None)
    return _style(fig, 340, legend=True)
