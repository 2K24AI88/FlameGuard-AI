"""FlameGuard AI: NASA Space Apps 2026, Flame in Freefall.

Run: streamlit run app.py
Pages: Overview, Research Explorer, Evidence Lab (tests, limits, scenario check), Telemetry Lab, Model Lab, Methods & Limits.
"""
from __future__ import annotations

import io
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import context as cx
import evidence as ev
import gaps as gp
import flame_core as fc
import telemetry as tl

ROOT = Path(__file__).parent
DATA = ROOT / "data"
LOCAL_FILE = Path.home() / "Downloads" / "Saffire_Data_Table2017.161.00.42.03-EU.csv"
DOI = "https://doi.org/10.60555/0t15-1z43"

st.set_page_config(page_title="FlameGuard AI", page_icon="🔥", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: #4a5565 !important; opacity: 1 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ helpers
def style(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        colorway=["#3b6ea5", "#c0794a", "#5b9279", "#8a6fa8", "#b5a042", "#6b7b8c"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(color="#1f2933"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=height,
    )
    return fig


@st.cache_data
def get_catalog() -> pd.DataFrame:
    return fc.load_catalog()


@st.cache_data
def get_insights() -> pd.DataFrame:
    return fc.load_insights()


@st.cache_data
def get_facts() -> pd.DataFrame:
    return fc.load_facts()


@st.cache_data
def get_measurements() -> pd.DataFrame:
    return fc.load_measurements()


@st.cache_data
def get_limits() -> pd.DataFrame:
    return fc.load_limits()


@st.cache_resource
def get_fact_search(_facts: pd.DataFrame) -> fc.FactSearch:
    return fc.FactSearch(_facts)


@st.cache_data
def get_measured() -> pd.DataFrame:
    return ev.load_measured()


@st.cache_data
def get_bass2() -> pd.DataFrame:
    return gp.load_bass2()


@st.cache_resource
def get_retriever(_df: pd.DataFrame) -> fc.Retriever:
    return fc.Retriever(_df)


@st.cache_data(show_spinner="Cleaning telemetry...")
def load_clean_bytes(raw: bytes):
    return tl.clean_telemetry(pd.read_csv(io.BytesIO(raw), low_memory=False))


@st.cache_data(show_spinner="Cleaning telemetry...")
def load_clean_path(path: str, mtime: float):
    return tl.clean_telemetry(pd.read_csv(path, low_memory=False))


def limit_note(text: str) -> None:
    st.caption("⚠️ " + text)


@contextmanager
def guarded(label: str):
    """Show a clear error for a section that fails, so a problem is visible instead of leaving a silent gap."""
    try:
        yield
    except Exception as e:  # noqa: BLE001 - deliberately broad: report, never hide
        st.error(f"{label} could not be shown: {type(e).__name__}: {e}")


def show_chart(build, height=None, fallback=None) -> None:
    """Draw a Plotly chart. If building it fails, say so and show the underlying table instead."""
    try:
        st.plotly_chart(style(build(), height), width="stretch")
    except Exception as e:  # noqa: BLE001
        st.warning(f"This chart could not be drawn ({type(e).__name__}: {e}). The table shows the same data.")
        if fallback is not None:
            st.dataframe(fallback, width="stretch", hide_index=True)


EXAMPLE_QUESTIONS = [
    "Are standard ground flammability tests enough for spacecraft materials?",
    "Can smoke detectors catch the small particles smoke makes in microgravity?",
    "Does lower pressure slow flame spread?",
    "What is missing for Moon and Mars gravity?",
]


def render_answer(card: dict) -> None:
    """Show an evidence-grounded answer: findings with citations, strength, caveats, gaps. Nothing is generated."""
    if card["hits"].empty:
        st.warning("Nothing in the sourced findings matches this question. That is a gap in this prototype, not an answer.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Evidence strength", card["strength"])
    c2.metric("Distinct sources", card["n_sources"])
    c3.metric("Measured rows supporting this answer", card["n_measured"])
    st.caption(
        "'Measured rows supporting this answer' counts only rows matched to this question whose status is not 'Unconfirmed'. "
        "It is separate from the total of measured rows shown on the Overview."
    )
    st.caption(
        "Strength rule: 1 distinct source = Limited, 2 = Moderate, 3 or more = Strong. Strong also needs at least one "
        "supporting measured row (one that is not 'Unconfirmed') and no caveats on the matched findings. It counts what is stored here and is not "
        "a measure of scientific certainty."
    )
    st.markdown("**Sourced findings**")
    cite_links(card["hits"])
    if card["caveats"]:
        st.markdown("**Caveats to carry along**")
        for c in dict.fromkeys(card["caveats"]):
            st.markdown(f"- {c}")
    st.markdown("**What is not covered**")
    for g in card["gaps"]:
        st.warning(g)
    if len(card["experiments"]):
        st.markdown("**Experiments behind these findings**")
        st.dataframe(card["experiments"][["name", "platform", "topic", "source_read"]], width="stretch", hide_index=True)


# ------------------------------------------------------------------ sidebar
PAGES = [
    "Overview",
    "Research Explorer",
    "Evidence Lab",
    "Safety Context",
    "Telemetry Lab",
    "Model Lab",
    "Methods & Limits",
]
with st.sidebar:
    st.markdown("## 🛰️ FlameGuard AI")
    st.caption("NASA Space Apps 2026 · Flame in Freefall")
    page = st.radio("Workspace", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("Research prototype. Not for flight, certification or safety decisions.")


def cite_links(facts_df: pd.DataFrame) -> None:
    for _, f in facts_df.iterrows():
        cav = f" *Caveat: {f['caveat']}*" if isinstance(f["caveat"], str) and f["caveat"] else ""
        st.markdown(f"- {f['statement']} [{f['source_title']} · {f['source_location']}]({f['source_url']}){cav}")


# ------------------------------------------------------------------ pages
def page_overview() -> None:
    cat, facts, meas = get_catalog(), get_facts(), get_measurements()
    vs = fc.verification_summary(cat, meas, get_limits())
    st.title("🔥 FlameGuard AI")
    st.subheader("Evidence-first fire-safety insights from NASA microgravity combustion research")
    st.write(
        "NASA has run decades of fire experiments in microgravity, but the results sit in scattered papers, "
        "pages and data tables. FlameGuard AI pulls them together, keeps every claim tied to its source, "
        "and turns it into something a safety engineer can search, compare and brief from."
    )
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("### 1. Summarize")
        st.write("A catalog of NASA combustion experiments, sourced findings, and one-click evidence briefs with citations.")
        st.caption("Research Explorer: Catalog, Insights, Brief")
    with c2.container(border=True):
        st.markdown("### 2. Rank")
        st.write("Rank experiments by criteria you weight yourself, and compare real test results side by side.")
        st.caption("Research Explorer: Rank · Evidence Lab")
    with c3.container(border=True):
        st.markdown("### 3. Interpret")
        st.write(
            "Ask a fire-safety question and get cited findings, an evidence-strength rating, caveats and gaps. "
            "Plus an exploratory look at real telemetry."
        )
        st.caption("Research Explorer: Ask & Evidence · Compare · Telemetry Lab")
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Experiments catalogued", len(cat))
    m2.metric("Sourced findings", len(facts))
    m3.metric("Measured test rows", len(meas))
    m4.metric("Sources read in detail", f"{vs['catalog_read']} of {vs['catalog_total']}")
    st.markdown("#### How much of this is verified?")
    v1, v2 = st.columns(2)
    with v1.container(border=True):
        st.markdown("**Catalog sources**")
        st.write(f"{vs['catalog_read']} read in detail · {vs['catalog_overview']} overview only · {vs['catalog_name_only']} name only")
    with v2.container(border=True):
        st.markdown("**Measured test rows**")
        st.write(
            f"{vs['rows_confirmed']} confirmed by author · {vs['rows_two_readings']} two automated readings agree · "
            f"{vs['rows_read_once']} read once and match NASA's table · {vs['rows_unconfirmed']} unconfirmed"
        )
    st.caption("Statuses come from the `verification` column in data/. They change as rows are checked against the PDFs. Details: Evidence Lab → Verification.")
    st.markdown("#### What makes this different")
    st.markdown(
        "- **Every claim has a citation**, and entries not read in detail say so.\n"
        "- **Blank means not stated**: missing values are never filled in.\n"
        "- **Evidence gaps are shown, not hidden**: Research gaps counts recorded tests by gravity, oxygen and pressure. Zero Moon or Mars gravity tests are recorded, and NASA says PSI has none yet.\n"
        "- **Exploratory telemetry analysis** of real data. The time unit is checked against the file's own clock, but sensor "
        "calibration is not documented, so lags are labeled exploratory."
    )
    with st.expander("Worked example: are standard ground flammability tests enough?"):
        render_answer(fc.answer_card(EXAMPLE_QUESTIONS[0], get_fact_search(facts), cat, meas, get_limits()))


def page_research() -> None:
    st.header("🔬 Research Explorer")
    cat, facts = get_catalog(), get_facts()
    t_cat, t_rank, t_ask, t_cmp, t_ins, t_brief = st.tabs(["Catalog", "Rank", "Ask & Evidence", "Compare", "Insights", "Brief"])

    with t_cat:
        fuel = st.multiselect("Fuel state", sorted(cat["fuel_state"].unique()), default=sorted(cat["fuel_state"].unique()))
        view = cat[cat["fuel_state"].isin(fuel)]
        st.dataframe(
            view[["name", "program", "platform", "fuel_state", "topic", "description", "source_read", "psi_id", "psi_id_status", "source_url"]],
            width="stretch", hide_index=True,
            column_config={"source_url": st.column_config.LinkColumn("source")},
        )
        st.subheader("What does the catalog cover?")
        cov = (cat[fc.SCORE_COLS].sum()).rename(index=fc.SCORE_LABELS).reset_index()
        cov.columns = ["criterion", "experiments (weighted)"]
        cov = cov.sort_values("experiments (weighted)")
        show_chart(
            lambda: px.bar(cov, x="experiments (weighted)", y="criterion", orientation="h",
                           title="How many catalogued experiments meet each criterion"),
            360, cov,
        )
        st.caption("Moon/Mars-gravity tests is zero because none of the sources read describes one. That is the gap.")
        st.dataframe(fc.evidence_gap_table(cat), width="stretch")
        limit_note("Criteria are checks against the sources read. The basis for each entry is in data/experiments.csv.")

    with t_rank:
        st.write("Set what matters to you. The score is the weighted share of criteria an experiment meets, so you can see why it ranks where it does.")
        cols = st.columns(4)
        weights = {c: cols[i % 4].slider(fc.SCORE_LABELS[c], 0, 10, 5, key=f"w_{c}") for i, c in enumerate(fc.SCORE_COLS)}
        if sum(weights.values()) == 0:
            st.warning("Set at least one weight above zero.")
        else:
            ranked = fc.rank_experiments(cat, weights)
            top10 = ranked.head(10).iloc[::-1]

            def build_rank():
                fig = px.bar(top10, x="score", y="name", orientation="h", title="Top 10 for your weights (0-100)")
                fig.update_xaxes(range=[0, 100])
                return fig

            show_chart(build_rank, 400, top10[["name", "score"]])
            st.dataframe(ranked[["name", "score"] + fc.SCORE_COLS + ["source_read"]], width="stretch", hide_index=True)
            top = ranked.iloc[0]
            st.info(f"**Top for your weights: {top['name']}** ({top['score']}/100). {top['description']}")
            st.caption(f"Basis: {top['criteria_basis']}")
        limit_note("This ranks how well an experiment fits your criteria. It does not measure scientific importance or fire hazard.")

    with t_ask:
        st.write(
            "Ask a fire-safety question. The app retrieves the closest sourced findings (TF-IDF text search), then applies "
            "transparent rules to report evidence strength, caveats and gaps. It never writes new claims."
        )
        pick = st.selectbox("Example questions", ["(type your own below)"] + EXAMPLE_QUESTIONS)
        typed = st.text_input("Your question", placeholder="e.g. do smoke detectors work for small particles in space?")
        q = typed.strip() or ("" if pick.startswith("(") else pick)
        if q:
            st.markdown(f"**Question:** {q}")
            with guarded("The answer"):
                render_answer(fc.answer_card(q, get_fact_search(facts), cat, get_measurements(), get_limits()))

    with t_cmp:
        st.write("Compare two catalogued experiments side by side, with warnings about what the comparison cannot support.")
        names = dict(zip(cat["name"], cat["id"]))
        keys = list(names)
        a = st.selectbox("First experiment", keys, index=keys.index("Saffire-I to III") if "Saffire-I to III" in keys else 0, key="cmp_x")
        b = st.selectbox("Second experiment", [k for k in keys if k != a], index=0, key="cmp_y")
        with guarded("The comparison"):
            res = fc.compare_experiments(names[a], names[b], cat, facts, get_measurements())
            n1, n2 = res["names"]
            c1, c2 = st.columns(2)
            c1.metric(f"{n1}: findings / measured rows", f"{res['facts'][0]} / {res['measured'][0]}")
            c2.metric(f"{n2}: findings / measured rows", f"{res['facts'][1]} / {res['measured'][1]}")
            st.dataframe(res["criteria"], width="stretch", hide_index=True)
            st.caption(f"Criteria met by both: {res['shared_criteria']}. 1 = yes, 0.5 = partly, 0 = no, judged against the sources read.")
            for w in res["warnings"]:
                st.warning(w)
            if not res["warnings"]:
                st.success("No comparison warnings were triggered.")

    with t_ins:
        ins = get_insights()
        q = st.selectbox("Choose a fire-safety question", ins["question"].tolist())
        row = ins[ins["question"] == q].iloc[0]
        st.markdown("#### What the sources say")
        st.write(row["answer"])
        st.markdown("#### Sources")
        cite_links(facts[facts["id"].isin(row["fact_ids"].split(";"))])
        st.warning(f"Limitation: {row['caution']}")
        limit_note("Curated from the cited sources. Not validated operational safety guidance.")

    with t_brief:
        st.write("Generate a cited evidence brief for the experiments you choose, as Markdown you can download.")
        names = dict(zip(cat["name"], cat["id"]))
        picks = st.multiselect("Experiments", list(names), default=["Saffire-I to III", "SAME"])
        if picks:
            brief = fc.make_brief([names[p] for p in picks], cat, facts, get_measurements(), get_limits())
            st.download_button("Download brief (.md)", brief, file_name="flameguard_brief.md", mime="text/markdown")
            with st.expander("Preview", expanded=True):
                st.markdown(brief)


def page_evidence() -> None:
    st.header("📋 Evidence Lab")
    st.info(
        "Values are transcribed from NASA papers and reports. A blank means the source does not state it. "
        "Every row carries a verification status; most still need a person to check them against the PDF."
    )
    meas, lim = get_measurements(), get_limits()
    if meas.empty:
        st.error("No measured test rows loaded (data/measurements.csv is missing or empty).")
        return
    vs = fc.verification_summary(get_catalog(), meas, lim)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Measured test rows", len(meas))
    k2.metric("Two automated readings agree", vs["rows_two_readings"])
    k3.metric("Read once, matched to table", vs["rows_read_once"])
    k4.metric("Confirmed by author", vs["rows_confirmed"])
    k5.metric("Unconfirmed limit rows", vs["rows_unconfirmed"])
    st.caption(
        "Automated agreement is not independent human verification. Only 'Confirmed by author' means a person "
        "checked the row against the source."
    )
    t_tab, t_burn, t_heat, t_lim, t_b2, t_gap, t_scn, t_ver = st.tabs(
        ["Test table", "Burn time", "Heat & gases", "Flammability limits", "BASS-II matrix", "Research gaps",
         "Scenario check", "Verification"]
    )

    with t_tab, guarded("The test table"):
        exps = st.multiselect("Experiment", sorted(meas["experiment"].unique()), default=sorted(meas["experiment"].unique()))
        v = meas[meas["experiment"].isin(exps)]
        st.dataframe(
            v[["test_id", "experiment", "material", "flow_cm_s", "direction", "o2_pct", "pressure_kpa", "burn_s",
               "spread_mm_s", "peak_heat_release_w", "total_heat_kj", "status", "uncertain_fields", "notes", "source_location", "source_url"]],
            width="stretch", hide_index=True,
            column_config={"source_url": st.column_config.LinkColumn("source")},
        )
        limit_note("Saffire-I and II rows come from the table in NTRS 20170008805; Saffire IV-VI rows from the 2024 ICES paper. See 'uncertain_fields' for values with a known question.")

    with t_burn, guarded("The burn-time chart"):
        b = meas.dropna(subset=["burn_s"]).copy()
        b["label"] = b["experiment"] + " " + b["test_id"].astype(str)
        show_chart(
            lambda: px.bar(b, x="label", y="burn_s", color="direction", hover_data=["material", "flow_cm_s", "o2_pct"],
                           title="Reported burn duration by test", labels={"burn_s": "Burn time (s)", "label": ""}),
            420, b[["label", "material", "direction", "burn_s"]],
        )
        st.markdown(
            "- **Saffire-I:** the same fabric at the same 20 cm/s flow burned for 420 s with concurrent flow and 70 s with opposed flow.\n"
            "- **Saffire-II:** silicone did not sustain spread in microgravity (no burn time recorded); thick PMMA slabs burned for 600 to 900 s with only about 10 cm of damage."
        )
        st.warning("Tests differ in sample size, material and how much burned, so burn times alone do not show why they differ. Do not read this as proof of cause.")

    with t_heat, guarded("The heat and gas view"):
        h = meas.dropna(subset=["peak_heat_release_w", "pressure_kpa"]).copy()
        show_chart(
            lambda: px.scatter(h, x="pressure_kpa", y="peak_heat_release_w", color="material", size="o2_num",
                               hover_data=["experiment", "test_id", "o2_pct", "burn_s"],
                               title="Peak heat release vs pressure (marker size = oxygen %)",
                               labels={"pressure_kpa": "Pressure (kPa)", "peak_heat_release_w": "Peak heat release (W)"}),
            420, h[["experiment", "test_id", "material", "pressure_kpa", "peak_heat_release_w"]],
        )
        st.dataframe(h[["experiment", "test_id", "material", "pressure_kpa", "o2_pct", "peak_heat_release_w", "total_heat_kj", "co2_rise_ppm", "co_rise_ppm", "status", "uncertain_fields"]],
                     width="stretch", hide_index=True)
        st.warning(
            "Pressure, oxygen and material all change between these tests, so this chart cannot isolate the effect of "
            "pressure. NASA's own summary says reduced pressure slows flame spread in Saffire. "
            "Total heat values carry a known mapping question (see uncertain_fields)."
        )

    with t_lim, guarded("The flammability limits"):
        if lim.empty:
            st.info("No flammability-limit table found (data/measured_results.csv).")
        else:
            st.write("Flow and oxygen limits where flames quench or blow off, mostly from BASS-II. Values are as listed in the source table.")
            st.warning("Every row here is currently unconfirmed: the BASS-II report could not be re-read to cross-check it.")
            st.dataframe(lim[["experiment", "material", "o2_pct", "flow_cm_s", "flow_direction", "gravity", "metric", "value", "unit", "status", "note", "source_locator", "source_url"]],
                         width="stretch", hide_index=True,
                         column_config={"source_url": st.column_config.LinkColumn("source")})
            ext = lim[lim["metric"].str.contains("radiative-extinction", na=False)].copy()
            if not ext.empty:
                ext["label"] = ext["material"] + " @ " + ext["o2_pct"].astype(str) + "% O2"
                show_chart(
                    lambda: px.bar(ext, x="label", y="value", title="Listed radiative-extinction flow velocity (cm/s), unconfirmed", labels={"value": "cm/s", "label": ""}),
                    380, ext[["label", "value"]],
                )
                st.caption("For the 100 micrometre sheets, the listed value is 1.0 cm/s at about 21% oxygen and 2.6 cm/s at about 16%. Sample widths differ between those rows.")
        limit_note("These rows were read by automated extraction and need confirmation against the reports.")

    with t_b2, guarded("The BASS-II test matrix"):
        b2 = get_bass2()
        if b2.empty:
            st.info("No BASS-II matrix found (data/bass2_matrix.csv).")
        else:
            st.write(
                "Test conditions and gas readings for BASS-II burns, listed by investigator. This is the author-supplied file "
                "`BASS-II_Test_Matrix_by_PI_Final.xlsx`, converted to a table (entries reading 'none' are left blank). The author states it comes "
                "from PSI (PSI-25); that origin is not independently confirmed here."
            )
            b2 = b2.assign(
                o2_drop=(b2["o2_initial_pct"] - b2["o2_final_pct"]).round(2),
                co_change_ppm=(b2["co_final_ppm"] - b2["co_initial_ppm"]).round(1),
            )
            b2["check"] = np.where(b2["o2_drop"] > 5, "final oxygen far below initial: check source", "")
            b2["missing_fields"] = b2[["o2_final_pct", "co2_final_pct", "co_initial_ppm", "co_final_ppm"]].isna().sum(axis=1)
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Tests listed", len(b2))
            q2.metric("Initial oxygen range", f"{b2['o2_initial_pct'].min():g} to {b2['o2_initial_pct'].max():g}%")
            q3.metric("Tests with both CO readings", int(b2[["co_initial_ppm", "co_final_ppm"]].notna().all(axis=1).sum()))
            q4.metric("Rows flagged to check", int((b2["check"] != "").sum()))
            pis = st.multiselect("Investigator sheet", sorted(b2["investigator_sheet"].unique()),
                                 default=sorted(b2["investigator_sheet"].unique()), key="b2_pis")
            word = st.text_input("Material contains", "", key="b2_word")
            v = b2[b2["investigator_sheet"].isin(pis)]
            if word.strip():
                v = v[v["material"].str.contains(word.strip(), case=False, na=False)]
            st.dataframe(v, width="stretch", hide_index=True)
            vo = v.dropna(subset=["o2_initial_pct", "o2_final_pct"])
            show_chart(
                lambda: px.scatter(vo, x="o2_initial_pct", y="o2_final_pct", color="investigator_sheet", hover_data=["test_no", "material"],
                                   title="Oxygen before and after each test (% by volume)",
                                   labels={"o2_initial_pct": "Initial oxygen (%)", "o2_final_pct": "Final oxygen (%)"}),
                420, vo[["test_no", "material", "o2_initial_pct", "o2_final_pct"]],
            )
            vc = v.dropna(subset=["co_change_ppm"])
            show_chart(
                lambda: px.scatter(vc, x="o2_initial_pct", y="co_change_ppm", color="investigator_sheet", hover_data=["test_no", "material"],
                                   title="Change in CO during each test (final minus initial, ppm)",
                                   labels={"o2_initial_pct": "Initial oxygen (%)", "co_change_ppm": "CO change (ppm)"}),
                420, vc[["test_no", "material", "o2_initial_pct", "co_change_ppm"]],
            )
            st.warning(
                "These are readings, not a model. Materials, sample sizes, burn durations and flow settings differ between tests, so "
                "neither chart shows what causes a change. 'Fan display' and 'air display' are instrument readings, not cm/s, so they "
                "cannot be compared with flow speeds in other tabs."
            )
            with st.expander("Worked example: why final CO alone can mislead"):
                ex = b2[b2["test_no"].isin(["F1", "F2", "F3"])][
                    ["test_no", "material", "fan_display", "air_display", "o2_initial_pct", "co_initial_ppm", "co_final_ppm", "co_change_ppm"]]
                st.dataframe(ex, width="stretch", hide_index=True)
                st.markdown(
                    "In these three Nomex tests, test F2 had the higher air-display reading and a final CO of 82 ppm, but its CO started at "
                    "105 ppm, so CO **fell** by 23 ppm. Test F1 started at 0 ppm and ended at 2 ppm. Three tests with different starting "
                    "gas cannot support a rule such as 'more airflow means more CO'."
                )

    with t_gap, guarded("Research gap intelligence"):
        tests = gp.condition_tests(meas, get_bass2())
        st.write(
            "Where is evidence recorded, thin or missing? Every count below is computed from the stored test rows "
            f"({len(tests)} tests with a known oxygen level). A count means tests recorded here, not verified or good quality."
        )
        mh = gp.moon_habitat_summary(tests)
        g1, g2, g3, g4 = st.columns(4)
        g1.metric(f"Tests at {gp.MOON_O2_MIN:g}%+ oxygen and {gp.MOON_P_MAX_KPA:g} kPa or less", len(mh["both"]))
        g2.metric(f"Tests at {gp.MOON_O2_MIN:g}%+ oxygen, any pressure", len(mh["o2_only"]))
        g3.metric(f"Tests at {gp.MOON_P_MAX_KPA:g} kPa or less, any oxygen", len(mh["p_only"]))
        g4.metric("Partial-gravity tests", mh["partial_gravity_tests"])
        st.caption(
            "The lunar-habitat range is from NASA's Space Apps 2026 challenge text: likely 30% oxygen or more at 10.2 psi "
            f"({gp.MOON_P_MAX_KPA:g} kPa) or less."
        )
        if len(mh["both"]):
            st.markdown("**Recorded tests inside that range:** " + ", ".join(sorted(mh["both"]["experiment"].unique())))
            st.dataframe(mh["both"][["experiment", "test_id", "o2", "pressure_kpa", "status"]], width="stretch", hide_index=True)
            st.caption("Different material, flow and sample size from a Moon base. They show the atmosphere has been tested, not that it is understood.")
        gm = gp.gravity_matrix(tests)
        show_chart(
            lambda: px.imshow(gm, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                              labels=dict(x="Oxygen band", y="Gravity", color="Tests"),
                              title="Recorded tests by gravity and oxygen"),
            330, gm.reset_index(),
        )
        st.caption("Ground (1 g) tests are outside this collection, so there is no row for them. Zeros in the Moon and Mars rows are counted, not assumed.")
        pm = gp.pressure_matrix(tests)
        show_chart(
            lambda: px.imshow(pm, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                              labels=dict(x="Pressure band", y="Oxygen band", color="Tests"),
                              title="Microgravity tests by oxygen and pressure"),
            400, pm.reset_index(),
        )
        st.caption("Tests with no recorded pressure fall under 'pressure not recorded': every BASS-II matrix row (the file has no pressure column) and the Saffire-I and II rows (their sources state none).")
        st.download_button(
            "Download the recorded tests as a table (.csv)", tests.to_csv(index=False),
            file_name="flameguard_recorded_tests.csv", mime="text/csv",
            help="One row per recorded test with oxygen, pressure, gravity and verification status, for your own modeling.",
        )
        st.caption(
            "For modeling: conditions per recorded test. It mixes experiments with different materials, flows and measurements, and some "
            "rows are unconfirmed. It is a starting table, not a validated dataset."
        )
        st.subheader("Explore a gap")
        e1, e2, e3 = st.columns(3)
        gsel = e1.selectbox("Gravity", gp.GRAVITY_ROWS, key="gap_g")
        osel = e2.selectbox("Oxygen band", [b[0] for b in gp.O2_BANDS], index=4, key="gap_o")
        psel = e3.selectbox("Pressure band", [b[0] for b in gp.P_BANDS] + [gp.NO_P], index=0, key="gap_p") if gsel == "Microgravity" else None
        d = gp.describe_gap(tests, gsel, osel, psel)
        st.markdown(f"**{d['cell']}: {d['status']}**")
        st.write(d["why"])
        if len(d["related"]):
            st.markdown("**Nearest recorded microgravity tests**")
            st.dataframe(d["related"][["experiment", "test_id", "o2", "pressure_kpa", "d_o2", "d_pressure_kpa", "within_tolerance", "status"]],
                         width="stretch", hide_index=True)
        for name, note in d["partial_gravity_notes"]:
            st.markdown(f"- **{name}:** {note}")
        st.info("Suggested research question (a template filled from the cell, not generated science): " + d["question"])

    with t_scn, guarded("The scenario check"):
        measured = get_measured()
        st.write(
            "Pick a scenario and see which measured conditions are closest, and what the collection does not cover. "
            "This lists evidence. It does not predict a fire or rate a design."
        )
        preset = st.selectbox("Preset", list(ev.PRESETS))
        st.caption("Moon and suit presets use the NASA Space Apps 2026 challenge text: a lunar habitat will likely run at 30% oxygen or more and 10.2 psi or less; spacesuits use 100% oxygen at 4.3 psi. The text gives no airflow, so 20 cm/s is only a starting value: change it.")
        pr = ev.PRESETS[preset]
        c1, c2, c3 = st.columns(3)
        o2 = c1.number_input("Oxygen (%)", min_value=5.0, max_value=100.0, value=float(pr["o2"]), step=0.5, key=f"scn_o2_{preset}")
        flow = c2.number_input("Airflow (cm/s)", min_value=0.0, max_value=200.0, value=float(pr["flow"]), step=0.5, key=f"scn_flow_{preset}")
        use_p = c3.checkbox("Specify pressure", value=pr["pressure"] is not None, key=f"scn_usep_{preset}")
        pressure = None
        if use_p:
            pressure = c3.number_input("Pressure (hPa)", min_value=100.0, max_value=1100.0, value=float(pr["pressure"] or 1013.0), step=10.0, key=f"scn_p_{preset}")
        near = ev.nearest_evidence(measured, o2, flow, k=3)
        gaps = ev.coverage_gaps(measured, o2, flow, pressure)
        margin = ev.oxygen_margin(measured)
        st.subheader("Nearest measured conditions")
        if near.empty:
            st.warning("No microgravity rows with both oxygen and flow are available.")
        else:
            st.dataframe(
                near[["experiment", "material", "o2_pct", "flow_cm_s", "flow_direction", "metric", "value", "unit",
                      "d_o2", "d_flow", "within_tolerance", "source_locator", "verification"]],
                width="stretch", hide_index=True,
            )
            st.caption("Closeness uses tolerances of 2 oxygen points and 5 cm/s. Those are choices, not physics.")
        st.subheader("Saffire tests with oxygen and pressure recorded")
        st.caption("The table above uses rows with oxygen and flow. These rows come from the Saffire test table, which records pressure.")
        sp = gp.condition_tests(meas)
        sp = sp[sp["pressure_kpa"].notna()]
        np_ = gp.nearest_tests(sp, o2, pressure / 10 if pressure is not None else None, k=3)
        if np_.empty:
            st.info("No Saffire test rows with oxygen and pressure are available.")
        else:
            st.dataframe(np_[["experiment", "test_id", "o2", "pressure_kpa", "d_o2", "d_pressure_kpa", "within_tolerance", "status"]],
                         width="stretch", hide_index=True)
            st.caption("Tolerances are 2 oxygen points and 10 kPa. They are choices, not physics.")
        st.subheader("Not covered")
        for g in gaps:
            st.warning(g)
        if margin:
            st.info(
                f"For one PMMA rod, steady spread occurred at {margin['micro_o2']:g}% oxygen in microgravity versus a "
                f"{margin['ground_o2']:g}% ground blow-off limit (difference {margin['margin']:g} points). One material only, unconfirmed."
            )
        brief = ev.build_brief(o2, flow, pressure, near, gaps, margin)
        st.download_button("Download scenario brief (.md)", brief, file_name="flameguard_scenario_brief.md", mime="text/markdown")
        with st.expander("Preview brief"):
            st.markdown(brief)
        limit_note("Rows come from a sourced table that was read by automated extraction. Confirm against the reports.")

    with t_ver, guarded("The verification view"):
        st.write("What has been checked, and exactly what still needs a person. Change a row's `verification` text to start with 'Confirmed by author' once you have checked it against the PDF.")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Confirmed by author", vs["rows_confirmed"])
        s2.metric("Two readings agree", vs["rows_two_readings"])
        s3.metric("Read once, matches table", vs["rows_read_once"])
        s4.metric("Unconfirmed", vs["rows_unconfirmed"])
        todo = fc.confirmation_checklist(meas, lim)
        st.markdown(f"**{len(todo)} rows still need a human check**")
        st.dataframe(todo, width="stretch", hide_index=True, column_config={"source_url": st.column_config.LinkColumn("source")})
        st.download_button("Download checklist (.csv)", todo.to_csv(index=False), file_name="flameguard_confirmation_checklist.csv", mime="text/csv")
        st.caption("'Two readings agree' means two automated reads of the PDF gave the same value. It is not a human check.")


# ---- telemetry lab -------------------------------------------------------
def resolve_telemetry():
    up = st.file_uploader("Upload the SAFFIRE telemetry CSV (optional)", type="csv")
    if up is not None:
        df, rep = load_clean_bytes(up.getvalue())
        return df, rep, f"uploaded file: {up.name}", None
    sample = DATA / "telemetry_sample.csv"
    if sample.exists():
        df, rep = load_clean_path(str(sample), sample.stat().st_mtime)
        meta_path = DATA / "telemetry_sample.meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        return df, rep, "bundled sample (data/telemetry_sample.csv)", meta
    if LOCAL_FILE.exists():
        df, rep = load_clean_path(str(LOCAL_FILE), LOCAL_FILE.stat().st_mtime)
        return df, rep, f"local file: {LOCAL_FILE.name}", None
    return None, None, None, None


def page_telemetry() -> None:
    st.header("🛰️ Telemetry Lab (exploratory)")
    st.caption("The project author labels this file SAFFIRE-III. The file name does not say which experiment it is, so confirm that on the PSI page.")
    st.warning("Historical research telemetry. Not a live connection, not a validated fire detector.")
    df, rep, source, meta = resolve_telemetry()
    if df is None:
        st.error("No telemetry loaded.")
        st.markdown(
            "Upload the CSV above, or build the bundled sample once:\n\n"
            "`python make_sample.py \"C:/Users/you/Downloads/Saffire_Data_Table2017.161.00.42.03-EU.csv\"`"
        )
        return
    if df.empty:
        st.error("No rows survived timeline cleaning. Check that this is the SAFFIRE telemetry table.")
        return

    st.caption(f"Source: {source}")
    with st.expander("What NASA's sources say about this experiment"):
        cite_links(get_facts()[get_facts()["id"].isin(["F02", "F10", "F39"])])
        st.caption("No Saffire-III results were found in the sources read, so nothing below is compared to a published result.")
    if meta and meta.get("downsample_step", 1) > 1:
        st.info(f"This sample keeps every {meta['downsample_step']}th row, so response delays have coarser resolution.")

    inf = tl.infer_time_unit(df)
    unit = st.radio(
        "Time unit of the RelativeSeconds column",
        ["milliseconds", "seconds"],
        index=1 if inf["unit"] == "seconds" else 0,
        horizontal=True,
        help="Checked against the CygnusTimestamp column in this file (see the note below). The documentation was not available to confirm it.",
    )
    st.caption("Unit check: " + inf["reason"])
    if inf["start_utc"]:
        st.caption(
            f"The Cygnus clock places this recording from {inf['start_utc']} to {inf['end_utc']} UTC. NASA's Saffire fact sheet says "
            "Saffire-III ignition was planned for early June 2017, which fits, but the file does not name its experiment, so that is "
            "not confirmed. The '2017.161' in the file name is day 161 (10 June), later than these timestamps, so it is probably when the table was exported."
        )
    df = df.assign(t_s=tl.elapsed_seconds(df, unit))
    samp = tl.sampling_summary(df["t_s"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Usable rows", f"{len(df):,}")
    c2.metric("Rows dropped (bad timeline)", f"{rep['rows_dropped']:,}")
    c3.metric("Duration", f"{samp['duration_s'] / 60:.1f} min")
    c4.metric("Median step", f"{samp['median_dt_s']:.3g} s" if samp.get("median_dt_s") else "n/a")
    st.caption(f"Duration and step assume the unit is {unit}. If that is wrong, every time below is wrong by the same factor.")

    sensors = tl.sensor_columns(df)
    t_q, t_sig, t_ign, t_rep, t_cmp = st.tabs(
        ["Data quality", "Signals", "Ignition → response", "Replay", "Compare"]
    )

    with t_q:
        q = tl.quality_report(df, sensors)
        k1, k2, k3 = st.columns(3)
        k1.metric("Channels checked", len(q))
        k2.metric("Constant channels", int((q["status"] != "varies").sum()))
        k3.metric(f"Channels with values below {int(tl.SENTINEL_LIMIT)}", int((q[f"below_{int(tl.SENTINEL_LIMIT)}"] > 0).sum()))
        st.dataframe(q.sort_values("missing_pct", ascending=False), width="stretch", hide_index=True)
        limit_note("Flags are exploratory. Constant channels and extreme values are not automatically errors: check sensor definitions and calibration.")

    with t_sig:
        labels = {tl.channel_label(c): c for c in sensors}
        pick = st.selectbox("Channel", list(labels))
        col = labels[pick]
        mask = st.checkbox("Mask radiometer values below -1000 (flagged sentinels)", value=True)
        s = tl.numeric_series(df, col, mask_sentinel=mask)
        plot = tl.decimate(pd.DataFrame({"t_s": df["t_s"], "v": s}).dropna())
        if plot.empty:
            st.error("No usable readings for this channel.")
        else:
            show_chart(
                lambda: px.line(plot, x="t_s", y="v", title=f"{pick} over time", labels={"t_s": "Elapsed time (s)", "v": pick}),
                420, plot.head(200),
            )
            v = s.dropna()
            m1, m2, m3 = st.columns(3)
            m1.metric("Median", f"{v.median():.4g}")
            m2.metric("Minimum", f"{v.min():.4g}")
            m3.metric("Maximum", f"{v.max():.4g}")
            st.caption(f"Plot is decimated to at most 6,000 points for speed. Statistics use all {len(v):,} valid readings.")
        limit_note("Units and calibration of this channel are not verified. This describes the signal only.")

    with t_ign:
        st.write(
            "Find when an igniter turned on, then measure how long other channels take to depart from their "
            "pre-ignition baseline. This is an exploratory lag in recorded data under the time unit you chose, "
            "not a verified response time."
        )
        igniters = [c for c in sensors if "Igniter" in c]
        if not igniters:
            st.warning("No igniter channels in this file.")
        else:
            ig_labels = {tl.channel_label(c): c for c in igniters}
            default = list(ig_labels).index("Sample 1 igniter current") if "Sample 1 igniter current" in ig_labels else 0
            ig = ig_labels[st.selectbox("Igniter channel", list(ig_labels), index=default)]
            ig_series = tl.numeric_series(df, ig)
            auto = st.checkbox("Detect threshold automatically", value=True)
            thr = None
            if not auto:
                thr = st.number_input("Threshold (channel units)", value=float(ig_series.median()), format="%.4f")
            intervals = tl.detect_on_intervals(df["t_s"], ig_series, threshold=thr)
            if not intervals:
                st.warning("No 'on' interval found. Try a manual threshold, or a different igniter channel.")
            else:
                iv = pd.DataFrame(intervals)
                iv.index = [f"Event {i + 1}" for i in range(len(iv))]
                st.dataframe(iv.round(2), width="stretch")
                st.caption(
                    "Sanity check on the unit: Saffire tests ran for minutes (Saffire-I burned for 420 s and 70 s), and Saffire-III repeated "
                    "Saffire-I at a higher flow speed. The unit comes from the file's own clock (see the unit check above); if these durations "
                    "still look like fractions of a second or as many hours, tell me and switch the unit above."
                )
                as_ms = iv["duration_s"] if unit == "milliseconds" else iv["duration_s"] / 1000
                as_s = iv["duration_s"] * 1000 if unit == "milliseconds" else iv["duration_s"]
                st.dataframe(
                    pd.DataFrame({"If RelativeSeconds is milliseconds (s)": as_ms.round(3), "If RelativeSeconds is seconds (s)": as_s.round(3)}),
                    width="stretch",
                )
                st.caption(
                    "For scale: Saffire-II ignited most samples for 9.2 s at 80 W (sources read). Igniter durations for "
                    "Saffire-I and III were not found, so this is only a plausibility guide."
                )
                pick_ev = st.selectbox("Ignition event", list(iv.index))
                onset = float(iv.loc[pick_ev, "start_s"])
                default_ch = [c for c in sensors if any(k in c for k in ("Radiometer", "SMK_DET", "TEMP_SIG", "Anemometer"))]
                chosen = st.multiselect(
                    "Response channels",
                    [tl.channel_label(c) for c in sensors],
                    default=[tl.channel_label(c) for c in default_ch][:8],
                )
                with st.expander("Detection settings"):
                    pre = st.slider("Baseline window before ignition (s)", 5, 300, 30)
                    horizon = st.slider("Look-ahead after ignition (s)", 10, 3600, 300)
                    k = st.slider("Sensitivity (noise multiples)", 2.0, 10.0, 5.0)
                    mm = st.slider("Consecutive samples required", 1, 10, 3)
                    mineff = st.slider("Minimum change (fraction of channel range)", 0.0, 0.3, 0.05)
                if chosen:
                    res = tl.response_table(
                        df, onset, [tl.LABEL_TO_CHANNEL.get(x, x) for x in chosen],
                        pre=float(pre), horizon=float(horizon), k=float(k), m=int(mm), min_effect=float(mineff),
                    )
                    shown = res[["label", "status", "delay_s", "direction", "peak_change"]].rename(
                        columns={"label": "Channel", "status": "Result", "delay_s": "Delay after ignition (s)",
                                 "direction": "Direction", "peak_change": "Peak change"}
                    )
                    st.dataframe(shown, width="stretch", hide_index=True)
                    got = res[res["status"] == "response"].sort_values("delay_s")
                    if not got.empty:
                        show_chart(
                            lambda: px.bar(got, x="delay_s", y="label", orientation="h",
                                           title="Exploratory: time from ignition to first sustained response",
                                           labels={"delay_s": "Delay (s)", "label": ""}),
                            80 + 40 * len(got), got[["label", "delay_s"]],
                        )
                    st.info(
                        "A delay here is the first sustained departure from baseline, in this file's sampling. "
                        "Sensor filtering, placement and sampling all add to it, and the sensor calibration is undocumented. "
                        "It does not rate any detector."
                    )

    with t_rep:
        st.write("Scrub through the test and watch several channels at once.")
        labels = {tl.channel_label(c): c for c in sensors}
        defaults = [x for x in ("Sample 1 igniter current", "Radiometer 1", "Smoke detector F2 scatter") if x in labels]
        picks = st.multiselect("Channels (up to 4)", list(labels), default=defaults, max_selections=4)
        if picks:
            tmax = float(df["t_s"].max())
            now = st.slider("Time (s)", 0.0, tmax, 0.0, step=max(tmax / 500, 0.01))
            def build_replay():
                fig = make_subplots(rows=len(picks), cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=picks)
                for i, p in enumerate(picks, start=1):
                    d = tl.decimate(pd.DataFrame({"t_s": df["t_s"], "v": tl.numeric_series(df, labels[p])}).dropna())
                    fig.add_trace(go.Scatter(x=d["t_s"], y=d["v"], mode="lines", name=p, showlegend=False), row=i, col=1)
                    fig.add_vline(x=now, line_color="#c0794a", row=i, col=1)
                return fig

            show_chart(build_replay, 180 * len(picks) + 80)
            idx = int(np.searchsorted(df["t_s"].to_numpy(), now))
            idx = min(idx, len(df) - 1)
            cols = st.columns(len(picks))
            for c, p in zip(cols, picks):
                val = tl.numeric_series(df, labels[p]).iloc[idx]
                c.metric(p, "n/a" if pd.isna(val) else f"{val:.4g}")
        limit_note("Replay of recorded data. Nothing here is live.")

    with t_cmp:
        st.write("Compare two channels. Radiometer sentinels are masked first.")
        labels = {tl.channel_label(c): c for c in sensors}
        names = list(labels)
        a = st.selectbox("First channel", names, key="cmp_a")
        b = st.selectbox("Second channel", [n for n in names if n != a], key="cmp_b")
        res = tl.compare_channels(df, labels[a], labels[b])
        if res["status"] != "ok":
            st.warning(f"Cannot compare: {res['status']} (paired readings: {res['n']}).")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Spearman (rank)", f"{res['spearman']:.3f}")
            m2.metric("Pearson (linear)", f"{res['pearson']:.3f}")
            m3.metric("Paired readings", f"{res['n']:,}")
            d = tl.decimate(pd.DataFrame({a: tl.numeric_series(df, labels[a]), b: tl.numeric_series(df, labels[b])}).dropna(), 4000)
            show_chart(lambda: px.scatter(d, x=b, y=a, opacity=0.5, title=f"{a} vs {b}"), 420, d.head(200))
        st.warning(
            "Correlation does not prove cause or a fire. Time-series readings are autocorrelated, so two channels "
            "that both trend over time can correlate strongly without any link."
        )


def page_model() -> None:
    st.header("🧠 Model Lab")
    st.write(
        "A sandbox for testing whether a table of experiment or sensor data can support a model. It inspects every column first, "
        "compares the model with a simple baseline on held-out rows, and says plainly when the result is not usable."
    )
    st.info(
        "This is not a fire-risk predictor. It does not predict fire growth or spacecraft hazards, and nothing here has been validated "
        "against real fires. Use it to explore sensor behavior or to see how conditions relate to outcomes in a table you trust."
    )
    src = st.radio("Data source", ["Upload my CSV", "Synthetic demo (NOT NASA data)"], horizontal=True)
    df = None
    is_demo = False
    if src.startswith("Upload"):
        up = st.file_uploader("CSV file", type="csv", key="model_upload")
        if up is not None:
            df = pd.read_csv(up)
    else:
        df = fc.make_demo_data()
        is_demo = True
        st.warning("SYNTHETIC data generated by code to test the pipeline. It is not NASA data. Do not report its results as findings.")

    if df is not None:
        st.dataframe(df.head(10), width="stretch")
        with st.expander("Inspect every column before training", expanded=not is_demo):
            st.dataframe(fc.inspect_columns(df), width="stretch", hide_index=True)
            st.caption("Check what each column is. Clock columns, constant columns and motion sensors are poor choices for a fire model.")
        masked_counts = {}
        probe = fc.inspect_columns(df)
        if probe["note"].str.contains("no data' codes").any():
            if st.checkbox("Treat values below -1000 as missing (recommended: they are 'no data' codes)", value=True, key="mask_sent"):
                df, masked_counts = fc.mask_sentinels(df)
                st.caption("Masked: " + ", ".join(f"{k} ({v:,} rows)" for k, v in masked_counts.items()))
        num_cols = list(df.select_dtypes("number").columns)
        if len(num_cols) < 2:
            st.error("Need at least two numeric columns.")
        else:
            sug = fc.suggest_target(df)
            target = st.selectbox("What to predict (target)", num_cols, index=num_cols.index(sug) if sug in num_cols else len(num_cols) - 1,
                                  help="Pick a physical quantity such as temperature, smoke, oxygen or heat, not a clock or a motion sensor.")
            pool = [c for c in num_cols if c != target and not fc.is_clock_column(c) and not fc.is_motion_column(c)]
            feats = st.multiselect("Inputs (features)", [c for c in num_cols if c != target], default=pool[:4])
            guess = "classification" if df[target].nunique() <= 5 else "regression"
            task = st.radio("Task", ["classification", "regression"], index=0 if guess == "classification" else 1, horizontal=True)
            ordered = fc.looks_time_ordered(df)
            split_label = st.radio(
                "How to test it", ["Chronological: train on earlier rows, test on later rows", "Random folds (only for independent rows)"],
                index=0 if ordered else 1, horizontal=False,
                help="Sensor recordings need the chronological test. Random folds let neighbouring readings sit in both sets and inflate scores.",
            )
            split = "chronological" if split_label.startswith("Chronological") else "random"
            clocks = [c for c in df.columns if fc.is_clock_column(c) and pd.api.types.is_numeric_dtype(df[c])]
            time_col = None
            if split == "chronological" and clocks:
                pick = st.selectbox("Order rows by", ["(keep the row order)"] + clocks, index=1 if ordered else 0)
                time_col = None if pick.startswith("(") else pick
            for w in fc.leakage_warnings(feats, target) + fc.target_warnings(df, target, feats, split):
                st.warning(w)
            if st.button("Train and evaluate") and feats:
                try:
                    st.session_state["res"] = fc.train_eval(df, feats, target, task, split=split, time_col=time_col)
                    st.session_state["res_target"] = target
                    st.session_state["res_demo"] = is_demo
                except ValueError as e:
                    st.session_state.pop("res", None)
                    st.error(str(e))

    res = st.session_state.get("res")
    if res:
        st.subheader("Results")
        (st.success if res["beats_baseline"] else st.error)(res["verdict"])
        if st.session_state.get("res_demo"):
            st.caption("These results are from synthetic demo data and say nothing about NASA experiments.")
        st.caption(f"{res['note']}{res['n_rows']:,} rows used; {res['n_eval']:,} held-out predictions scored; {res['split_text']}. "
                   "Higher is better for accuracy and R2; lower is better for errors. A baseline is the simple guess a model must beat.")
        st.table(pd.DataFrame(res["metrics"].items(), columns=["metric", "value"]))
        imp = res["importances"].iloc[::-1]
        show_chart(
            lambda: px.bar(imp, x="importance", y="feature", orientation="h", title="Permutation importance"),
            80 + 50 * len(imp), imp,
        )
        st.caption("Permutation importance shows what the model relies on, not proof of cause.")
        st.markdown(
            f"**What this result means.** Target: `{st.session_state.get('res_target', 'target')}`. Inputs: {', '.join(f'`{c}`' for c in res['columns'])}. "
            "It says how well these inputs track the target on later or unseen rows of this one file. It does not generalize to other "
            "experiments, materials or gravity levels, and it is not a safety assessment."
        )
        st.subheader("Try a condition")
        vals = {}
        cols = st.columns(min(4, len(res["columns"])))
        for i, c in enumerate(res["columns"]):
            lo, hi = res["ranges"][c]
            vals[c] = cols[i % len(cols)].number_input(c, value=float((lo + hi) / 2), format="%.4f", key=f"in_{c}")
        outside = [c for c in res["columns"] if not res["ranges"][c][0] <= vals[c] <= res["ranges"][c][1]]
        if outside:
            st.error(f"Outside the data range for: {', '.join(outside)}. This prediction is extrapolation and unreliable.")
        if not res["beats_baseline"]:
            st.warning("This model did not beat the baseline, so its predictions are not reliable.")
        pred = res["model"].predict(pd.DataFrame([vals])[res["columns"]])[0]
        st.metric(f"Predicted {st.session_state.get('res_target', 'target')}", f"{pred:.4g}")
        st.caption("A prediction is not a safety guarantee.")


def page_context() -> None:
    st.header("🧭 Safety Context")
    st.write(
        "Why this research matters, which experiments inform each part of spacecraft fire safety, and where NASA fire technology has "
        "helped on Earth. Each item is a short paraphrase of a NASA page with its link."
    )
    ctx = cx.load_context()
    t_hist, t_pil, t_earth = st.tabs(["Lessons from history", "Prevention, detection, suppression", "Earth applications"])
    with t_hist, guarded("Lessons from history"):
        for _, r in ctx[ctx["kind"] == "history"].iterrows():
            st.markdown(f"#### {r['title']}")
            st.write(r["summary"])
            st.caption(f"Caveat: {r['caveat']}  ·  [{r['source_title']}]({r['source_url']})")
    with t_pil, guarded("Safety pillars"):
        st.write(
            "NASA's spacecraft fire safety work focuses on prevention, detection and suppression. The lists below place each catalogued "
            "investigation by one of its own criteria (solid materials, smoke detection, suppression). The criteria are this project's "
            "checks against the sources read, not NASA categories."
        )
        for pillar, d in cx.pillar_table(get_catalog()).items():
            st.markdown(f"#### {pillar}  ·  {len(d)} investigations")
            st.dataframe(d, width="stretch", hide_index=True)
        st.caption("Fire safety involving rockets and propellants is outside the research in the PSI database, per NASA's Space Apps 2026 challenge text.")
    with t_earth, guarded("Earth applications"):
        for _, r in ctx[ctx["kind"] == "earth"].iterrows():
            st.markdown(f"#### {r['title']}")
            st.write(r["summary"])
            st.caption(f"Caveat: {r['caveat']}  ·  [{r['source_title']}]({r['source_url']})")


def page_methods() -> None:
    st.header("📐 Methods & Limits")
    cat = get_catalog()
    vs = fc.verification_summary(cat, get_measurements(), get_limits())
    st.markdown(
        f"""
### What each analysis does
| Part | Method | Is it machine learning? |
|---|---|---|
| Ask & Evidence | TF-IDF search over sourced findings, then rules for evidence strength, caveats and gaps. Returns stored statements with citations only. | Classic text retrieval plus transparent rules, no generation |
| Compare | Side-by-side criteria, counts of findings and measured rows, and warnings about what cannot be compared | No, transparent rules |
| Rank | Weighted share of criteria met; weights set by you | No, transparent arithmetic |
| Brief | Assembles stored findings, tests and caveats into Markdown | No |
| Evidence Lab | Plots of transcribed NASA test values | No |
| Data quality | Missing values, constant channels, sentinel counts | No |
| Ignition → response (exploratory) | Robust baseline (median and MAD) before ignition, then first sustained departure | No, signal analysis |
| BASS-II matrix | Table and charts of author-supplied test conditions and gas readings; unusual rows flagged | No |
| Safety Context | Short cited paraphrases of NASA history and spin-off pages; investigations listed under prevention, detection or suppression by one catalog criterion each | No, stored text and counting |
| Research gaps | Counts of recorded tests by gravity, oxygen and pressure band, computed from stored rows; a templated research question per cell | No, counting |
| Model Lab | Inspects columns, masks 'no data' codes, then tests a random forest and baselines on held-out rows (chronological option for time series); permutation importance | Yes |

### How sources were checked
- **{vs['catalog_read']} of {vs['catalog_total']} catalog entries** have a source page or paper read in detail, {vs['catalog_overview']} have an overview only (ACME experiments NASA's booklet does not describe separately), and {vs['catalog_name_only']} are known by name only.
- **Measured rows:** {vs['rows_confirmed']} confirmed by the author, {vs['rows_two_readings']} where two automated readings of the PDF agree (Saffire IV-VI), {vs['rows_read_once']} read once and matching NASA's table (Saffire-I and II), and {vs['rows_unconfirmed']} unconfirmed (flammability limits). No row has yet been checked by a person unless it says so.
- Known open questions are listed in each row's `uncertain_fields` (for example which method a printed heat total belongs to).
- PSI record numbers: PSI-98 and PSI-102 pages exist but their contents could not be read here; PSI-25, PSI-39, PSI-69 are unverified.

### Limits
- The catalog criteria are checks against the sources read, not NASA measurements, and the sources are not exhaustive.
- Saffire-III has no published results in the sources read. The telemetry analysis is exploratory.
- The telemetry time unit is inferred from the file itself: RelativeSeconds advances in step with the Cygnus clock in milliseconds. Sensor calibration is not documented in the sources read. The file is labeled SAFFIRE-III by the author; its name does not say so, and its timestamps (5 June 2017) fit NASA's schedule for Saffire-III without confirming it.
- Research-gap counts are tests recorded in this collection. They are not verified, not weighted by quality, and not a complete census of NASA data. The BASS-II matrix file was supplied by the author and its PSI origin is not independently confirmed.
- The Model Lab is a sandbox. It is not a validated fire-risk or fire-growth predictor, and a model that does not beat its baseline is reported as such.
- The "AI" in Ask & Evidence is text retrieval plus fixed rules. The only learned model is the optional Model Lab.
- SAFFIRE-I (burn times) and SAFFIRE-III (telemetry) are different experiments.
- The synthetic demo in the Model Lab is not NASA data.
- Not validated for flight, certification or safety decisions.

### Data
NASA Technical Reports Server (ntrs.nasa.gov), NASA science pages, and Physical Sciences Informatics (psi.nasa.gov). This project is not affiliated with or endorsed by NASA.
"""
    )


{
    "Overview": page_overview,
    "Research Explorer": page_research,
    "Evidence Lab": page_evidence,
    "Safety Context": page_context,
    "Telemetry Lab": page_telemetry,
    "Model Lab": page_model,
    "Methods & Limits": page_methods,
}[page]()
