"""Design system: CSS tokens and small HTML components for the Streamlit app.

Cards are Streamlit containers created with a key starting with "card" (Streamlit adds the
class `st-key-<key>`), so they can hold real widgets and still share one style.
"""

import html

import streamlit as st

CSS = """
<style>
:root {
  --bg: #f5f7fb; --surface: #ffffff; --ink: #0f172a; --muted: #5b6577; --border: #e2e8f0;
  --primary: #2f5bd3; --primary-strong: #1e40af; --primary-soft: #eef2ff;
  --success: #15803d; --success-soft: #dcfce7; --danger: #b91c1c; --danger-soft: #fee2e2;
  --warning: #b45309; --warning-soft: #fef3c7;
  --radius: 14px;
  --shadow: 0 1px 2px rgba(15,23,42,.05), 0 6px 20px rgba(15,23,42,.06);
  --shadow-hover: 0 2px 4px rgba(15,23,42,.06), 0 12px 28px rgba(15,23,42,.10);
}

.stApp { background: var(--bg); }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1320px; }
h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.01em; }

/* ── hero ─────────────────────────────────────────── */
.hero { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px;
  padding: 28px 32px; border-radius: 20px; color: #fff; margin-bottom: 20px;
  background: radial-gradient(1200px 300px at 0% 0%, #3b6cf0 0%, transparent 60%),
              linear-gradient(120deg, #1e3a8a 0%, #2f5bd3 100%);
  box-shadow: var(--shadow); }
.hero h1 { color: #fff; font-size: clamp(1.5rem, 2.6vw, 2.1rem); font-weight: 700; margin: 0 0 6px; padding: 0; }
.hero p { margin: 0; opacity: .9; font-size: clamp(.9rem, 1.2vw, 1rem); max-width: 640px; }
.hero .tags { display: flex; flex-wrap: wrap; gap: 8px; }
.hero .tag { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); border-radius: 999px;
  padding: 5px 12px; font-size: .8rem; font-weight: 500; white-space: nowrap; }

/* ── cards (keyed Streamlit containers) ───────────── */
div[class*="st-key-card"] { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 20px 22px; box-shadow: var(--shadow); }
.section-title { display: flex; align-items: baseline; gap: 10px; margin: 0 0 12px; }
.section-title h3 { font-size: 1.05rem; font-weight: 600; margin: 0; padding: 0; }
.section-title span { color: var(--muted); font-size: .85rem; }

/* ── KPI grid ─────────────────────────────────────── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin: 4px 0 20px; }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 18px;
  box-shadow: var(--shadow); position: relative; overflow: hidden; transition: transform .18s ease, box-shadow .18s ease;
  animation: rise .35s ease both; }
.kpi:hover { transform: translateY(-2px); box-shadow: var(--shadow-hover); }
.kpi::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--accent, var(--primary)); }
.kpi .label { color: var(--muted); font-size: .75rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
.kpi .value { color: var(--ink); font-size: 1.75rem; font-weight: 700; line-height: 1.25; margin-top: 4px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi .sub { color: var(--muted); font-size: .8rem; margin-top: 2px; }
.kpi-grid .kpi:nth-child(2) { animation-delay: .05s; } .kpi-grid .kpi:nth-child(3) { animation-delay: .1s; }
.kpi-grid .kpi:nth-child(4) { animation-delay: .15s; }
@keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

/* ── chips & badges ───────────────────────────────── */
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 10px; }
.chip { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 999px; font-size: .78rem;
  font-weight: 600; border: 1px solid transparent; }
.chip.ok { background: var(--success-soft); color: var(--success); }
.chip.miss { background: var(--danger-soft); color: var(--danger); }
.chip.neutral { background: var(--primary-soft); color: var(--primary-strong); }
.chip-empty { color: var(--muted); font-size: .85rem; font-style: italic; }
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px; font-size: .8rem; font-weight: 600; }
.badge.ok { background: var(--success-soft); color: var(--success); }
.badge.miss { background: var(--danger-soft); color: var(--danger); }
.badge.warn { background: var(--warning-soft); color: var(--warning); }

/* ── candidate header ─────────────────────────────── */
.cand-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
.cand-head h2 { font-size: 1.4rem; margin: 0; padding: 0; }
.cand-head .file { color: var(--muted); font-size: .85rem; }
.score-ring { display: flex; align-items: baseline; gap: 6px; }
.score-ring b { font-size: 2.2rem; color: var(--primary-strong); line-height: 1; }
.score-ring span { color: var(--muted); font-size: .85rem; }
.summary { background: var(--primary-soft); border-left: 3px solid var(--primary); padding: 10px 14px; border-radius: 8px;
  color: var(--ink); font-size: .92rem; margin-top: 8px; }

/* ── sidebar ──────────────────────────────────────── */
section[data-testid="stSidebar"] { background: #0f1b3d; }
section[data-testid="stSidebar"] * { color: #dbe4ff; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #fff; }
section[data-testid="stSidebar"] .stButton button { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.2); }
section[data-testid="stSidebar"] .stButton button:hover { background: rgba(255,255,255,.16); border-color: rgba(255,255,255,.35); }
section[data-testid="stSidebar"] [data-testid="stExpander"] details { background: rgba(255,255,255,.06);
  border-color: rgba(255,255,255,.15); }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background: rgba(255,255,255,.08); }
.engine { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12); border-radius: 12px; padding: 12px 14px;
  font-size: .85rem; line-height: 1.7; }
.engine code { background: rgba(255,255,255,.1); color: #fff; padding: 1px 6px; border-radius: 6px; font-size: .8rem; white-space: nowrap; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.dot.on { background: #22c55e; box-shadow: 0 0 0 3px rgba(34,197,94,.25); }
.dot.off { background: #ef4444; box-shadow: 0 0 0 3px rgba(239,68,68,.25); }

/* ── native widget polish ─────────────────────────── */
.stButton button, .stDownloadButton button, .stFormSubmitButton button { border-radius: 10px; font-weight: 600;
  transition: transform .12s ease, box-shadow .12s ease; }
.stButton button:hover, .stDownloadButton button:hover { transform: translateY(-1px); }
.stButton button:focus-visible, .stDownloadButton button:focus-visible { outline: 3px solid #93b4ff; outline-offset: 2px; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { padding: 10px 16px; border-radius: 10px 10px 0 0; font-weight: 600; }
.stTabs [data-baseweb="tab"]:hover { background: var(--primary-soft); }
[data-testid="stChatMessage"] { border-radius: 12px; }
[data-testid="stExpander"] details { border-radius: var(--radius); border-color: var(--border); background: var(--surface); }

/* ── responsive ───────────────────────────────────── */
@media (max-width: 1000px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 3.5rem; }
  .hero { padding: 20px; border-radius: 16px; }
  .kpi-grid { grid-template-columns: 1fr; gap: 10px; }
  div[class*="st-key-card"] { padding: 14px; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def escape(x) -> str:
    return html.escape(str(x))


def hero(title: str, subtitle: str, tags: list[str]) -> None:
    tag_html = "".join(f'<span class="tag">{escape(t)}</span>' for t in tags)
    st.markdown(
        f'<header class="hero"><div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>'
        f'<div class="tags" aria-label="Highlights">{tag_html}</div></header>',
        unsafe_allow_html=True,
    )


def section_title(title: str, hint: str = "") -> None:
    hint_html = f"<span>{escape(hint)}</span>" if hint else ""
    st.markdown(f'<div class="section-title"><h3>{escape(title)}</h3>{hint_html}</div>', unsafe_allow_html=True)


def kpi_grid(items: list[tuple[str, str, str, str]]) -> None:
    """items: (label, value, sub-text, accent colour)."""
    cards = "".join(
        f'<div class="kpi" style="--accent:{accent}" title="{escape(value)}"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div><div class="sub">{escape(sub)}</div></div>'
        for label, value, sub, accent in items
    )
    st.markdown(f'<section class="kpi-grid" aria-label="Key metrics">{cards}</section>', unsafe_allow_html=True)


def chips(items: list[str], tone: str = "neutral", empty: str = "none") -> str:
    if not items:
        return f'<span class="chip-empty">{escape(empty)}</span>'
    icon = {"ok": "✓ ", "miss": "✕ "}.get(tone, "")
    return '<div class="chips">' + "".join(f'<span class="chip {tone}">{icon}{escape(i)}</span>' for i in items) + "</div>"


def badge(text: str, tone: str) -> str:
    return f'<span class="badge {tone}" role="status">{escape(text)}</span>'


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)
