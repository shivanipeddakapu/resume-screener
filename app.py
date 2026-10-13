"""
Resume Screener — Streamlit app
-------------------------------
Upload a ZIP of resume PDFs and a job description. A local LLM (Ollama) extracts the job's
requirements and each candidate's facts; scores are computed deterministically from those facts
plus embedding similarity, and a hybrid (vector + BM25) index powers grounded Q&A.

Run:  streamlit run app.py
"""

import hashlib
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from resume_screener import charts, ui
from resume_screener.assistant import answer, write_brief
from resume_screener.config import DEFAULT_SHORTLIST, DEFAULT_WEIGHTS, EMBEDDING_MODEL, LLM_MODEL, MAX_WORKERS
from resume_screener.ingest import read_resumes, unpack_pdfs
from resume_screener.llm import make_embeddings, make_llm, ollama_status
from resume_screener.retrieval import ResumeIndex
from resume_screener.scoring import COMPONENTS, active_components, compute_fit, screen, weight_shares

SAMPLES = Path(__file__).parent / "samples"
SAMPLE_ZIP = SAMPLES / "sample_resumes.zip"
SAMPLE_JD = SAMPLES / "job_description.txt"

SUGGESTED_QUESTIONS = [
    "Who has built distributed or large-scale systems?",
    "Which candidates have worked with AWS?",
    "What degrees do the candidates hold?",
]

st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="wide", initial_sidebar_state="auto")
ui.inject_css()

# ── session state ─────────────────────────────────────────────────────────────
RESULT_KEYS = ("index", "index_sig", "candidates", "req", "brief", "skipped", "elapsed", "chat", "use_sample")
STATE_DEFAULTS = {
    "index": None, "index_sig": None, "candidates": None, "req": None, "brief": "", "skipped": [],
    "elapsed": 0.0, "chat": [], "use_sample": False, "jd_input": "", "threshold": DEFAULT_SHORTLIST,
}  # fmt: skip
for key, default in STATE_DEFAULTS.items():
    st.session_state.setdefault(key, default)
for comp in COMPONENTS:
    st.session_state.setdefault(f"w_{comp}", DEFAULT_WEIGHTS[comp])


@st.cache_data(ttl=15, show_spinner=False)
def cached_ollama_status():
    return ollama_status()


def load_sample() -> None:
    st.session_state.use_sample = True
    st.session_state.jd_input = SAMPLE_JD.read_text(encoding="utf-8")


def reset_scoring() -> None:
    st.session_state.threshold = DEFAULT_SHORTLIST
    for c in COMPONENTS:
        st.session_state[f"w_{c}"] = DEFAULT_WEIGHTS[c]


def clear_session() -> None:
    if st.session_state.index is not None:
        st.session_state.index.release()
    for key in RESULT_KEYS:
        st.session_state[key] = STATE_DEFAULTS[key]


def ask(question: str) -> None:
    st.session_state.pending_question = question


def current_weights() -> dict[str, float]:
    return {c: st.session_state[f"w_{c}"] for c in COMPONENTS}


# ── sidebar ───────────────────────────────────────────────────────────────────
status = cached_ollama_status()
with st.sidebar:
    st.markdown("## 📄 Resume Screener")
    st.caption("Explainable screening on local models")

    dot, state_text = ("on", f"Ollama {status.version} connected") if status.reachable else ("off", "Ollama offline")
    ui.html_block(
        f'<div class="engine" role="status"><span class="dot {dot}"></span><b>{state_text}</b><br>'
        f"LLM <code>{LLM_MODEL}</code><br>Embeddings <code>{EMBEDDING_MODEL}</code></div>"
    )
    if status.problem:
        st.warning(status.problem)
        if st.button("Re-check Ollama", width="stretch"):
            cached_ollama_status.clear()
            st.rerun()

    st.markdown("### Scoring")
    st.slider("Shortlist threshold", 0, 100, key="threshold", help="Candidates whose fit score reaches this are shortlisted.")
    with st.expander("Component weights", expanded=False):
        for comp in COMPONENTS:
            st.slider(comp, 0, 100, key=f"w_{comp}")
        st.caption("Share of the fit score: " + " · ".join(f"{c} {v:.0%}" for c, v in weight_shares(current_weights()).items()))
    st.button("Reset scoring", on_click=reset_scoring, width="stretch")
    st.caption("Changes re-rank instantly — the LLM is not called again.")

    st.markdown("### Session")
    st.button("Clear results", on_click=clear_session, width="stretch", disabled=st.session_state.candidates is None)

# ── header + inputs ───────────────────────────────────────────────────────────
ui.hero(
    "Resume Screener",
    "Rank a stack of resumes against a job description with explainable scores — on your own machine.",
    ["🔒 Runs locally", "🧾 Evidence-backed skills", "⚖️ Tunable fit score"],
)

with st.container(key="card_input"):
    col_zip, col_jd = st.columns([1, 1.6], gap="large")
    with col_zip:
        ui.section_title("1 · Resumes", "ZIP of PDF files")
        upload = st.file_uploader("Upload a ZIP of resumes", type=["zip"], label_visibility="collapsed")
        if upload:
            st.caption(f"📦 {upload.name} · {upload.size / 1024:.0f} KB")
        elif st.session_state.use_sample:
            st.caption("🧪 Using the 6 bundled sample resumes (fictional people)")
        st.button("🧪 Try with sample data", on_click=load_sample, width="stretch")
    with col_jd:
        ui.section_title("2 · Job description")
        st.text_area(
            "Job description", key="jd_input", height=200, label_visibility="collapsed",
            placeholder="Paste the job description — responsibilities, required skills, experience…",
        )  # fmt: skip
    run_clicked = st.button(
        "🚀 Screen candidates", type="primary", width="stretch", disabled=not status.ready,
        help=None if status.ready else status.problem,
    )  # fmt: skip


def run_pipeline(zip_bytes: bytes, jd: str) -> None:
    sig = hashlib.sha256(zip_bytes).hexdigest()
    llm = make_llm()
    started = time.perf_counter()
    with st.status("Screening candidates…", expanded=True) as box:
        try:
            if st.session_state.index_sig != sig:  # rebuild only when the ZIP changed
                box.write("📦 Reading resumes…")
                resumes, skipped = read_resumes(unpack_pdfs(zip_bytes))
                if not resumes:
                    raise ValueError("None of the PDFs contain readable text (scanned images need OCR first).")
                if st.session_state.index is not None:
                    st.session_state.index.release()
                st.session_state.index = ResumeIndex(resumes, make_embeddings(), log=box.write)
                st.session_state.update(index_sig=sig, skipped=skipped, chat=[])
            else:
                box.write("♻️ Reusing the index for this ZIP.")
            index = st.session_state.index

            box.write(f"🧠 Profiling {len(index.resumes)} resumes with `{LLM_MODEL}`…")
            bar = st.progress(0.0, text="Reading the job requirements…")
            candidates, req = screen(
                index, llm, jd, MAX_WORKERS,
                lambda done, total: bar.progress(done / total, text=f"Profiled {done}/{total} resumes"),
            )  # fmt: skip
            box.write("📝 Writing the hiring brief…")
            ranked = compute_fit(candidates, current_weights(), st.session_state.threshold)
            brief = write_brief(llm, req, ranked, st.session_state.threshold)
        except Exception as e:
            box.update(label="Screening failed", state="error")
            st.error(f"{type(e).__name__}: {e}")
            return
        elapsed = time.perf_counter() - started
        st.session_state.update(candidates=candidates, req=req, brief=brief, elapsed=elapsed)
        box.update(label=f"Done ✅ — {len(candidates)} resumes screened in {elapsed:.0f}s", state="complete", expanded=False)


# Always-present slot for run progress / errors, so the elements below keep the same position
# on every rerun (otherwise Streamlit briefly shows stale copies of the tabs after a run).
run_slot = st.container()
if run_clicked:
    jd_text = st.session_state.jd_input.strip()
    zip_data = upload.getvalue() if upload else SAMPLE_ZIP.read_bytes() if st.session_state.use_sample else None
    with run_slot:
        if zip_data is None:
            st.error("Upload a ZIP of resume PDFs, or click **Try with sample data**.")
        elif not jd_text:
            st.error("Paste a job description first.")
        else:
            run_pipeline(zip_data, jd_text)

# ── empty state ───────────────────────────────────────────────────────────────
if st.session_state.candidates is None:
    st.write("")
    steps = [
        ("📦", "Upload", "Drop in a ZIP of PDF resumes and paste the job description."),
        ("🧠", "Extract", "A local LLM pulls out each candidate's skills — with quotes — plus experience and education."),
        ("📊", "Decide", "Scores are computed transparently; tune the weights, compare candidates and ask questions."),
    ]
    for col, (icon, title, text) in zip(st.columns(3, gap="medium"), steps, strict=True):
        with col, st.container(key=f"card_step_{title.lower()}"):
            ui.html_block(f"<h4 style='margin:0 0 6px'>{icon} {title}</h4><p style='margin:0;color:var(--muted)'>{text}</p>")
    st.stop()

# ── results ───────────────────────────────────────────────────────────────────
index: ResumeIndex = st.session_state.index
req = st.session_state.req
threshold = st.session_state.threshold
shares = weight_shares(current_weights(), active_components(st.session_state.candidates))
df = compute_fit(st.session_state.candidates, current_weights(), threshold)
n_total, n_short = len(df), int(df["Shortlisted"].sum())
best = df.iloc[0]

if st.session_state.skipped:
    st.warning(
        f"No readable text in {len(st.session_state.skipped)} PDF(s), likely scanned images: {', '.join(st.session_state.skipped)}"
    )
if not df["Profiled"].all():
    st.warning(f"{int((~df['Profiled']).sum())} resume(s) could not be profiled by the LLM and were scored from their text only.")


def candidate_label(i: int) -> str:
    r = df.iloc[i]
    return f"#{r['Rank']} · {r['Candidate']} — fit {r['Fit']:.0f}"


tab_overview, tab_candidates, tab_chat, tab_report, tab_compare = st.tabs(
    ["📊 Overview", "🧑‍💼 Candidates", "💬 Ask AI", "🔍 Candidate report", "⚖️ Compare"]
)

with tab_overview:
    ui.kpi_grid([
        ("Candidates", str(n_total), f"screened in {st.session_state.elapsed:.0f}s", "#2f5bd3"),
        ("Shortlisted", str(n_short), f"{n_short / n_total:.0%} · fit ≥ {threshold}", "#15803d"),
        ("Median fit", f"{df['Fit'].median():.0f}", f"range {df['Fit'].min():.0f}–{df['Fit'].max():.0f}", "#b45309"),
        ("Best match", best["Candidate"], f"fit {best['Fit']:.0f}", "#7c3aed"),
    ])  # fmt: skip
    left, right = st.columns([1.2, 1], gap="medium")
    with left, st.container(key="card_ranking"):
        st.plotly_chart(charts.fit_breakdown(df, threshold), width="stretch", config=charts.PLOTLY_CONFIG)
    with right, st.container(key="card_brief"):
        ui.section_title("Hiring brief", req.title)
        st.markdown("**Must-have**")
        ui.html_block(ui.chips(req.must_have, "neutral", "none detected"))
        if req.nice_to_have:
            st.markdown("**Nice-to-have**")
            ui.html_block(ui.chips(req.nice_to_have, "neutral"))
        with st.container(height=300, border=False):
            st.markdown(st.session_state.brief)
        st.caption("Written by the LLM from the scores at screening time.")

    c1, c2 = st.columns([1.2, 1], gap="medium")
    with c1, st.container(key="card_matrix"):
        st.plotly_chart(charts.skills_matrix(df, req), width="stretch", config=charts.PLOTLY_CONFIG)
    with c2, st.container(key="card_scatter"):
        st.plotly_chart(charts.experience_vs_relevance(df), width="stretch", config=charts.PLOTLY_CONFIG)

    with st.expander("How is the fit score calculated?"):
        st.markdown(
            "**Fit = " + " + ".join(f"{shares[c]:.0%} {c}" for c in COMPONENTS if shares[c]) + "** (each component is 0–100)\n\n"
            "* **Must-have / Nice-to-have** – share of the job's skills the resume demonstrates. A skill counts if it is "
            "written in the resume, or if the LLM quotes evidence that really appears in it.\n"
            f"* **Experience** – years in roles similar to this one against the job's minimum ({req.min_years:g} yrs).\n"
            "* **Education** – degree level against the requirement, reduced if the field doesn't match.\n"
            "* **Relevance** – embedding similarity between the job description and the whole resume.\n\n"
            f"Candidates with fit ≥ {threshold} are shortlisted. Adjust weights and threshold in the sidebar."
        )

with tab_candidates, st.container(key="card_table"):
    f1, f2 = st.columns([2, 1.2], gap="medium")
    search = f1.text_input(
        "Search", placeholder="Search names, skills or any keyword in the resumes…", label_visibility="collapsed"
    )
    view = f2.segmented_control("Status", ["All", "Shortlisted", "Not shortlisted"], default="All", label_visibility="collapsed")

    shown = df
    if view and view != "All":
        shown = shown[shown["Shortlisted"] == (view == "Shortlisted")]
    if search.strip():
        term = search.strip().lower()
        text_by_file = {r.name: r.text.lower() for r in index.resumes}
        shown = shown[
            shown["Candidate"].str.lower().str.contains(term, regex=False)
            | shown["Resume File"].map(lambda f: term in text_by_file.get(f, ""))
        ]

    table = shown.assign(Status=shown["Shortlisted"].map({True: "★ Shortlisted", False: "—"}))[
        [
            "Rank",
            "Candidate",
            "Fit",
            "Status",
            "Matched",
            "Missing",
            "Bonus",
            "Years",
            "Relevant years",
            "Degree",
            *COMPONENTS,
            "Resume File",
        ]
    ]
    st.dataframe(
        table, hide_index=True, width="stretch", height=min(560, 38 * len(table) + 40),
        column_config={
            "Rank": st.column_config.NumberColumn("#", width="small"),
            "Fit": st.column_config.ProgressColumn("Fit score", min_value=0, max_value=100, format="%.0f"),
            "Matched": st.column_config.ListColumn("Must-haves shown"),
            "Missing": st.column_config.ListColumn("Must-haves missing"),
            "Bonus": st.column_config.ListColumn("Nice-to-haves"),
            "Years": st.column_config.NumberColumn("Years", format="%.1f"),
            "Relevant years": st.column_config.NumberColumn("Relevant yrs", format="%.1f"),
            **{c: st.column_config.NumberColumn(c, format="%.0f") for c in COMPONENTS},
        },
    )  # fmt: skip
    st.caption(f"Showing {len(shown)} of {n_total} candidates")
    export = df.drop(columns=[c for c in df.columns if c.startswith("pts:")] + ["Evidence"]).assign(
        **{col: df[col].apply(", ".join) for col in ("Matched", "Missing", "Bonus", "Strengths", "Gaps")}
    )
    st.download_button(
        "⬇️ Download results (CSV)", export.to_csv(index=False).encode("utf-8"), "screening_results.csv", "text/csv"
    )

with tab_chat, st.container(key="card_chat"):
    ui.section_title("Ask about the candidate pool", "answers cite the resumes they come from")
    for col, suggestion in zip(st.columns(len(SUGGESTED_QUESTIONS)), SUGGESTED_QUESTIONS, strict=True):
        col.button(suggestion, on_click=ask, args=(suggestion,), width="stretch")

    history = st.container(height=420)
    with history:
        if not st.session_state.chat:
            st.caption("No questions yet — try a suggestion above or type below.")
        for role, message, sources in st.session_state.chat:
            with st.chat_message(role):
                st.markdown(message)
                if sources:
                    st.caption("Searched: " + ", ".join(sources))

    typed = st.chat_input("Ask about skills, experience, education…")
    question = typed or st.session_state.pop("pending_question", None)
    if question:
        with history:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"), st.spinner("Searching the resumes…"):
                try:
                    reply, sources = answer(question, index, make_llm())
                except Exception as e:
                    reply, sources = f"⚠️ {type(e).__name__}: {e}", []
                st.markdown(reply)
                if sources:
                    st.caption("Searched: " + ", ".join(sources))
        st.session_state.chat += [("user", question, []), ("assistant", reply, sources)]

with tab_report:
    pick = st.selectbox("Candidate", range(n_total), format_func=candidate_label, label_visibility="collapsed")
    row = df.iloc[pick]
    left, right = st.columns([1.25, 1], gap="medium")
    with left, st.container(key="card_report"):
        badge = ui.badge("★ Shortlisted", "ok") if row["Shortlisted"] else ui.badge("Below threshold", "miss")
        ui.html_block(
            f'<div class="cand-head"><div><h2>{ui.escape(row["Candidate"])}</h2>'
            f'<div class="file">📄 {ui.escape(row["Resume File"])} · {row["Years"]:g} yrs ({row["Relevant years"]:g} relevant) · {ui.escape(row["Degree"])}</div></div>'
            f'<div><div class="score-ring"><b>{row["Fit"]:.0f}</b><span>fit</span></div>{badge}</div></div>'
            + (f'<div class="summary">{ui.escape(row["Headline"])}</div>' if row["Headline"] else "")
        )
        st.markdown("**Must-haves shown**")
        ui.html_block(ui.chips(row["Matched"], "ok"))
        st.markdown("**Must-haves missing**")
        ui.html_block(ui.chips(row["Missing"], "miss"))
        if row["Bonus"]:
            st.markdown("**Nice-to-haves**")
            ui.html_block(ui.chips(row["Bonus"], "neutral"))
        s_col, g_col = st.columns(2)
        with s_col:
            st.markdown("**Strengths**")
            st.markdown("\n".join(f"- {s}" for s in row["Strengths"]) or "_none listed_")
        with g_col:
            st.markdown("**Gaps**")
            st.markdown("\n".join(f"- {g}" for g in row["Gaps"]) or "_none listed_")
    with right:
        with st.container(key="card_components"):
            st.plotly_chart(charts.component_bars(row, shares), width="stretch", config=charts.PLOTLY_CONFIG)
        with st.container(key="card_evidence"):
            ui.section_title("Skill evidence", "quoted from the resume")
            if row["Evidence"]:
                st.dataframe(
                    pd.DataFrame(list(row["Evidence"].items()), columns=["Skill", "Evidence"]),
                    hide_index=True, width="stretch",
                )  # fmt: skip
            else:
                st.caption("No required skills found.")

with tab_compare, st.container(key="card_compare"):
    ui.section_title("Compare candidates", "pick up to 3")
    chosen = st.multiselect(
        "Candidates to compare", range(n_total), default=list(range(min(2, n_total))), format_func=candidate_label,
        max_selections=3, label_visibility="collapsed",
    )  # fmt: skip
    if chosen:
        rows = [df.iloc[i] for i in chosen]
        left, right = st.columns([1.2, 1], gap="medium")
        with left:
            st.plotly_chart(charts.compare_bars(rows), width="stretch", config=charts.PLOTLY_CONFIG)
        with right:
            facts = ["Fit", *COMPONENTS, "Years"]
            st.dataframe(
                pd.DataFrame({f"#{r['Rank']} {r['Candidate']}": [round(r[f], 1) for f in facts] for r in rows}, index=facts),
                width="stretch",
            )
            for r in rows:
                st.markdown(f"**{r['Candidate']}** · missing must-haves")
                ui.html_block(ui.chips(r["Missing"], "miss"))
    else:
        st.caption("Select at least one candidate.")
