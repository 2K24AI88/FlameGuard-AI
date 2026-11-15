"""Core logic for Flame in Freefall: catalog, ranking, retrieval, ML.

No UI code here, so everything can be tested without Streamlit.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data"
# Criteria are checks against the sources read (0 = no, 0.5 = partly, 1 = yes). Basis text is in the catalog.
SCORE_COLS = [
    "spacecraft_scale",
    "solid_materials",
    "smoke_detection",
    "suppression_extinction",
    "atmosphere_varied",
    "partial_gravity",
    "in_psi",
]
SCORE_LABELS = {
    "spacecraft_scale": "Spacecraft-scale fires",
    "solid_materials": "Solid spacecraft materials",
    "smoke_detection": "Smoke / detection",
    "suppression_extinction": "Suppression / extinction",
    "atmosphere_varied": "Oxygen, pressure or flow varied",
    "partial_gravity": "Moon/Mars-gravity tests",
    "in_psi": "Data listed in NASA PSI",
}
MEASUREMENT_TO_CATALOG = {
    "Saffire-I": "saffire", "Saffire-II": "saffire",
    "Saffire-IV": "saffire-iv-vi", "Saffire-V": "saffire-iv-vi", "Saffire VI": "saffire-iv-vi",
}


# ---------------------------------------------------------------- catalog
def load_catalog(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "experiments.csv", dtype={"psi_id": "string"})


def load_facts(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "facts.csv")


def load_insights(path: Path | None = None) -> pd.DataFrame:
    """Curated findings that cite fact ids (data/key_findings.csv). The older, simpler
    data/insights.csv is read by evidence.py and is left as it was."""
    return pd.read_csv(path or DATA_DIR / "key_findings.csv")


def _o2_number(value) -> float:
    """'21.7 to 21.5' -> 21.6 (midpoint); '~22.1' -> 22.1; blanks -> NaN."""
    nums = re.findall(r"\d+(?:\.\d+)?", str(value))
    return float(np.mean([float(n) for n in nums])) if nums else np.nan


def verification_status(text) -> str:
    """Plain-language status from a row's verification note. Anything not clearly cross-checked is 'Unconfirmed'."""
    t = str(text).lower()
    if t.startswith("confirmed by author"):
        return "Confirmed by author"
    if t.startswith("two automated readings agree"):
        return "Two readings agree"
    if "matches nasa table" in t or t.startswith("read from table text"):
        return "Read once, matches table"
    return "Unconfirmed"


def load_measurements(path: Path | None = None) -> pd.DataFrame:
    """Real test rows transcribed from NASA papers. Blank = not stated in the source."""
    m = pd.read_csv(path or DATA_DIR / "measurements.csv")
    m["o2_num"] = m["o2_pct"].map(_o2_number)
    m["catalog_id"] = m["experiment"].map(MEASUREMENT_TO_CATALOG)
    m["status"] = m["verification"].map(verification_status)
    m["uncertain_fields"] = m["uncertain_fields"].fillna("") if "uncertain_fields" in m.columns else ""
    return m


def load_limits(path: Path | None = None) -> pd.DataFrame:
    """Flammability-limit values (extinction / blow-off) from a second sourced table.

    The two Saffire-I rows duplicate rows already in measurements.csv and are skipped.
    Saffire-V rows are mapped to the Saffire-IV-to-VI catalog entry."""
    path = path or DATA_DIR / "measured_results.csv"
    if not Path(path).exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    d = d[~d["id"].str.startswith("sf1-")].copy()
    d["catalog_id"] = d["catalog_id"].replace({"saffire": "saffire-iv-vi"})
    d["status"] = d["verification"].map(verification_status)
    return d.reset_index(drop=True)


def verification_summary(catalog: pd.DataFrame, meas: pd.DataFrame, lim: pd.DataFrame) -> dict:
    """Counts a reader can check: how many sources were read in detail, and how many numbers are cross-checked."""
    rows = pd.concat([meas[["status"]], lim[["status"]]] if len(lim) else [meas[["status"]]], ignore_index=True)
    return {
        "catalog_total": int(len(catalog)),
        "catalog_read": int((catalog["source_read"] == "read").sum()),
        "catalog_overview": int((catalog["source_read"] == "overview only").sum()),
        "catalog_name_only": int((catalog["source_read"] == "name only").sum()),
        "rows_total": int(len(rows)),
        "rows_confirmed": int((rows["status"] == "Confirmed by author").sum()),
        "rows_two_readings": int((rows["status"] == "Two readings agree").sum()),
        "rows_read_once": int((rows["status"] == "Read once, matches table").sum()),
        "rows_unconfirmed": int((rows["status"] == "Unconfirmed").sum()),
    }


def confirmation_checklist(meas: pd.DataFrame, lim: pd.DataFrame) -> pd.DataFrame:
    """Every measured row that still needs a person to check it, with where to look."""
    a = meas.assign(table="measurements.csv", what=meas["experiment"] + " " + meas["test_id"].astype(str),
                    where=meas["source_location"], check=meas["uncertain_fields"].replace("", "all values"))
    parts = [a[["table", "what", "status", "where", "check", "source_url"]]]
    if len(lim):
        b = lim.assign(table="measured_results.csv", what=lim["experiment"] + " " + lim["id"].astype(str),
                       where=lim["source_locator"], check=lim["metric"])
        parts.append(b[["table", "what", "status", "where", "check", "source_url"]])
    out = pd.concat(parts, ignore_index=True)
    return out[~out["status"].isin(["Read once, matches table", "Confirmed by author"])].reset_index(drop=True)



def rank_experiments(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Weighted share (0-100) of criteria met. Transparent: score = sum(weight * criterion) / sum(weights)."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one weight must be above zero.")
    out = df.copy()
    out["score"] = (sum(out[c] * w for c, w in weights.items()) / total * 100).round(1)
    return out.sort_values(["score", "name"], ascending=[False, True]).reset_index(drop=True)


def evidence_gap_table(df: pd.DataFrame) -> pd.DataFrame:
    """How many catalogued experiments per fuel state and platform type."""
    d = df.copy()
    d["platform_group"] = np.where(d["platform"].str.contains("ISS"), "ISS", "Spacecraft / other")
    return d.pivot_table(index="fuel_state", columns="platform_group", values="id", aggfunc="count", fill_value=0)


# --------------------------------------------------------------- retrieval
class Retriever:
    """TF-IDF search over experiment descriptions. Retrieval only, no text generation."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        text = (df["name"] + " " + df["program"] + " " + df["topic"] + " " + df["fuel_state"] + " " + df["description"])
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(text)

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> pd.DataFrame:
        q = self.vec.transform([query])
        sims = (self.matrix @ q.T).toarray().ravel()
        idx = np.argsort(-sims)[:k]
        res = self.df.iloc[idx].copy()
        res["match"] = sims[idx].round(3)
        return res[res["match"] >= min_score]


class FactSearch:
    """TF-IDF search over sourced facts. Returns the stored statement with its citation.
    It cannot write new claims: every result is a row someone read in a NASA source."""

    def __init__(self, facts: pd.DataFrame):
        self.df = facts.reset_index(drop=True)
        text = self.df["statement"] + " " + self.df["topic"] + " " + self.df["experiment_ids"].str.replace(";", " ").str.replace("-", " ")
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(text)

    def search(self, query: str, k: int = 4, min_score: float = 0.08) -> pd.DataFrame:
        q = self.vec.transform([query])
        sims = (self.matrix @ q.T).toarray().ravel()
        idx = np.argsort(-sims)[:k]
        res = self.df.iloc[idx].copy()
        res["match"] = sims[idx].round(3)
        return res[res["match"] >= min_score]


def evidence_strength(n_sources: int, n_cross_checked_rows: int, n_caveats: int = 0) -> str:
    """Transparent rule, not a statistic. Counts distinct sources among the best-matching findings:
    1 = Limited, 2 = Moderate, 3 or more = Strong. Strong also needs at least one cross-checked
    measured row for the experiments involved and no caveats on the matched findings; otherwise it is
    capped at Moderate. Word matching cannot judge topical relevance, so the rule leans cautious."""
    if n_sources >= 3 and n_cross_checked_rows >= 1 and n_caveats == 0:
        return "Strong"
    if n_sources >= 2:
        return "Moderate"
    return "Limited"


CROSS_CHECKED = ("Two readings agree", "Read once, matches table", "Confirmed by author")


def answer_card(query: str, fact_search: "FactSearch", catalog: pd.DataFrame,
                measurements: pd.DataFrame, limits: pd.DataFrame | None = None, k: int = 5) -> dict:
    """Evidence-grounded answer to a question: the best-matching sourced findings, the experiments behind
    them, cross-checked measured rows, caveats to carry along, and what is NOT covered.
    Nothing is generated: every field comes from stored, cited rows. Only findings scoring at least half
    as well as the best match are kept, so loosely related findings cannot inflate the result."""
    hits = fact_search.search(query, k=k)
    if hits.empty:
        return {"hits": hits, "experiments": catalog.iloc[0:0], "strength": None, "n_sources": 0,
                "n_measured": 0, "caveats": [], "gaps": ["Nothing in the sourced findings matches this question."]}
    hits = hits[hits["match"] >= 0.5 * hits["match"].max()]
    ids = sorted({e for cell in hits["experiment_ids"] for e in cell.split(";")} & set(catalog["id"]))
    exps = catalog[catalog["id"].isin(ids)]
    n_sources = int(hits["source_url"].nunique())
    rows = measurements[measurements["catalog_id"].isin(ids)]
    n_meas = int(rows["status"].isin(CROSS_CHECKED).sum())
    if limits is not None and len(limits):
        n_meas += int(limits[limits["catalog_id"].isin(ids)]["status"].isin(CROSS_CHECKED).sum())
    caveats = [c for c in hits["caveat"].dropna().tolist() if c]
    gaps = []
    if len(exps) and exps["partial_gravity"].max() == 0:
        gaps.append("None of the experiments behind these findings tested Moon or Mars gravity.")
    unread = exps.loc[exps["source_read"] != "read", "name"].tolist()
    if unread:
        gaps.append("Not read in detail (overview or name only): " + ", ".join(unread) + ".")
    if n_meas == 0:
        gaps.append("No cross-checked measured test rows back these findings; they rest on NASA's written summaries.")
    return {"hits": hits, "experiments": exps, "strength": evidence_strength(n_sources, n_meas, len(caveats)),
            "n_sources": n_sources, "n_measured": n_meas, "caveats": caveats, "gaps": gaps}


def compare_experiments(a_id: str, b_id: str, catalog: pd.DataFrame, facts: pd.DataFrame,
                        measurements: pd.DataFrame) -> dict:
    """Side-by-side comparison of two catalog entries, with warnings about what the comparison cannot support."""
    c = catalog.set_index("id")
    if a_id not in c.index or b_id not in c.index:
        raise ValueError("Unknown experiment id.")
    a, b = c.loc[a_id], c.loc[b_id]
    crit = pd.DataFrame({
        "criterion": [SCORE_LABELS[k] for k in SCORE_COLS],
        a["name"]: [a[k] for k in SCORE_COLS],
        b["name"]: [b[k] for k in SCORE_COLS],
    })
    crit["differs"] = crit[a["name"]] != crit[b["name"]]

    def n_facts(i):
        return int(facts["experiment_ids"].str.split(";").apply(lambda xs: i in xs).sum())

    n_a, n_b = int((measurements["catalog_id"] == a_id).sum()), int((measurements["catalog_id"] == b_id).sum())
    warn = []
    if a["fuel_state"] != b["fuel_state"]:
        warn.append(f"Different fuel states ({a['fuel_state']} vs {b['fuel_state']}): results do not transfer directly.")
    if a["platform"] != b["platform"]:
        warn.append(f"Different platforms ({a['platform']} vs {b['platform']}): scale and conditions differ.")
    if n_a == 0 or n_b == 0:
        warn.append("At least one of these has no measured test rows here, so only written findings can be compared.")
    if a["source_read"] != "read" or b["source_read"] != "read":
        warn.append("At least one entry was not read in detail (overview or name only).")
    if a["partial_gravity"] == 0 and b["partial_gravity"] == 0:
        warn.append("Neither tested Moon or Mars gravity.")
    return {"criteria": crit, "facts": (n_facts(a_id), n_facts(b_id)), "measured": (n_a, n_b),
            "names": (a["name"], b["name"]), "warnings": warn,
            "shared_criteria": int(((crit[a["name"]] >= 1) & (crit[b["name"]] >= 1)).sum())}


def make_brief(exp_ids: list[str], catalog: pd.DataFrame, facts: pd.DataFrame,
               measurements: pd.DataFrame, limits: pd.DataFrame | None = None) -> str:
    """Markdown brief with citations for the chosen experiments. Only stored, sourced content is used."""
    lines = ["# Fire-safety evidence brief", "",
             "Built from sourced entries only. Items marked 'confirm' were read by automated extraction and need a human check.", ""]
    for eid in exp_ids:
        row = catalog[catalog["id"] == eid]
        if row.empty:
            continue
        r = row.iloc[0]
        lines += [f"## {r['name']}", "", f"{r['description']}", "",
                  f"- Platform: {r['platform']}", f"- Topic: {r['topic']}",
                  f"- Source: {r['source_url']} ({'read' if r['source_read'] == 'read' else r['source_read'] + ', details not read'})", ""]
        fx = facts[facts["experiment_ids"].str.split(";").apply(lambda xs: eid in xs)]
        if len(fx):
            lines.append("**Findings and setup**")
            for _, f in fx.iterrows():
                cav = f" (Caveat: {f['caveat']})" if isinstance(f["caveat"], str) and f["caveat"] else ""
                lines.append(f"- {f['statement']} [{f['source_title']}, {f['source_location']}]({f['source_url']}){cav}")
            lines.append("")
        ms = measurements[measurements["catalog_id"] == eid]
        if len(ms):
            lines.append("**Measured tests (blank = not stated)**")
            lines.append("")
            lines.append("| Test | Material | Flow cm/s | Direction | O2 % | Pressure kPa | Burn s | Peak heat W | Status |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for _, m in ms.iterrows():
                g = lambda k: "" if pd.isna(m[k]) else m[k]
                lines.append(f"| {m['test_id']} | {m['material']} | {g('flow_cm_s')} | {g('direction')} | {g('o2_pct')} | {g('pressure_kpa')} | {g('burn_s')} | {g('peak_heat_release_w')} | {m['status']} |")
            lines.append("")
        if limits is not None and len(limits):
            ls = limits[limits["catalog_id"] == eid]
            if len(ls):
                lines.append("**Flammability limits (confirm against the report)**")
                for _, l in ls.iterrows():
                    val = "" if pd.isna(l["value"]) else f"{l['value']:g} {l['unit']}"
                    lines.append(f"- {l['experiment']}: {l['metric']} {val}. {l['note']} [{l['source_locator']}]({l['source_url']}) (status: {l['status']})")
                lines.append("")
        met = [fc_label for c, fc_label in SCORE_LABELS.items() if r[c] >= 1]
        lines += [f"**Criteria met:** {', '.join(met) if met else 'none of the listed criteria'}.", f"Basis: {r['criteria_basis']}", ""]
    lines += ["---", "Research prototype built for NASA Space Apps 2026. Not for flight or safety decisions."]
    return "\n".join(lines)


# ---------------------------------------------------------------------- ML
def make_demo_data(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC data to test the pipeline. NOT NASA data and not real physics results.
    Shape is only loosely inspired by the idea that more O2 helps a flame survive and more
    diluent (CO2) helps put it out."""
    rng = np.random.default_rng(seed)
    o2 = rng.uniform(0.12, 0.34, n)
    co2 = rng.uniform(0.0, 0.5, n)
    pressure = rng.uniform(50, 150, n)
    droplet_mm = rng.uniform(1.0, 3.0, n)
    logit = 18 * (0.21 - o2) + 9 * co2 - 0.01 * (pressure - 100) + rng.normal(0, 0.8, n)
    extinct = (logit > 0).astype(int)
    burn_rate = np.clip(0.9 * o2 - 0.35 * co2 + 0.001 * pressure + rng.normal(0, 0.03, n), 0.01, None)
    return pd.DataFrame(
        {"o2_fraction": o2, "co2_fraction": co2, "pressure_kpa": pressure, "droplet_mm": droplet_mm,
         "extinction": extinct, "burn_rate_mm2_s": burn_rate}
    )


def leakage_warnings(features: list[str], target: str) -> list[str]:
    """Flag column names that suggest post-test measurements are being used as inputs."""
    suspicious = ("extinction", "burn_time", "burn_rate", "final", "end", "result", "outcome", "post")
    warn = []
    for f in features:
        if any(s in f.lower() for s in suspicious):
            warn.append(f"'{f}' may be measured after the test finishes. If so, using it as an input leaks the answer.")
    return warn


# ------------------------------------------------------- column inspection
_CLOCK_WORDS = ("timestamp", "relativeseconds", "epoch", "datetime")
_MOTION_WORDS = ("imu", "omega", "attitude", "gnc", "gyro", "quat", "bdy", ".acc", "_acc")


def is_clock_column(name: str) -> bool:
    n = str(name).lower()
    return any(w in n for w in _CLOCK_WORDS) or n in ("time", "t", "t_s", "seconds")


def is_motion_column(name: str) -> bool:
    n = str(name).lower()
    return any(w in n for w in _MOTION_WORDS)


def column_note(name: str, series: pd.Series) -> str:
    notes = []
    if series.dropna().nunique() <= 1:
        notes.append("constant: carries no information")
    if is_clock_column(name):
        notes.append("clock or time column: a position in time, not a physical input")
    if is_motion_column(name):
        notes.append("spacecraft motion or orientation sensor: not a fire quantity")
    if pd.api.types.is_numeric_dtype(series) and series.notna().any() and float(series.min()) < SENTINEL_LIMIT and not is_clock_column(name):
        notes.append(f"has values below {SENTINEL_LIMIT:g}, which in SAFFIRE radiometer columns are 'no data' codes: mask them first")
    return "; ".join(notes)


SENTINEL_LIMIT = -1000.0


def mask_sentinels(df: pd.DataFrame, limit: float = SENTINEL_LIMIT) -> tuple[pd.DataFrame, dict]:
    """Turn very negative 'no data' codes into missing values (clock columns are left alone). Returns (copy, counts per column)."""
    out, counts = df.copy(), {}
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]) and not is_clock_column(c):
            bad = out[c] < limit
            if bad.any():
                counts[c] = int(bad.sum())
                out.loc[bad, c] = np.nan
    return out, counts


def inspect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """What every column holds, before anyone trains on it."""
    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s)
        rows.append({
            "column": c, "type": str(s.dtype), "missing": int(s.isna().sum()), "missing %": round(100 * float(s.isna().mean()), 1),
            "unique": int(s.nunique(dropna=True)),
            "min": float(s.min()) if num and s.notna().any() else np.nan, "max": float(s.max()) if num and s.notna().any() else np.nan,
            "mean": float(s.mean()) if num and s.notna().any() else np.nan, "std": float(s.std()) if num and s.notna().sum() > 1 else np.nan,
            "note": column_note(c, s),
        })
    return pd.DataFrame(rows)


def looks_time_ordered(df: pd.DataFrame) -> bool:
    """True if a clock-like column mostly increases down the rows, so neighbouring rows are close in time."""
    for c in df.columns:
        if is_clock_column(c) and pd.api.types.is_numeric_dtype(df[c]):
            d = df[c].dropna().diff().dropna()
            if len(d) > 10 and float((d > 0).mean()) > 0.9:
                return True
    return False


def suggest_target(df: pd.DataFrame) -> str | None:
    """First numeric column that is not a clock, a motion sensor or constant. None if there is no sensible one."""
    for c in df.select_dtypes("number").columns:
        if not is_clock_column(c) and not is_motion_column(c) and df[c].nunique() > 1:
            return c
    return None


def target_warnings(df: pd.DataFrame, target: str, features: list[str], split: str = "random") -> list[str]:
    """Plain-language reasons a training setup may be meaningless, shown before training."""
    w = []
    if is_clock_column(target):
        w.append(f"'{target}' is a clock column. Predicting time is not a fire-safety question.")
    if is_motion_column(target):
        w.append(f"'{target}' looks like a spacecraft motion or orientation sensor, not a fire quantity such as temperature, smoke, "
                 "oxygen or heat. A model of it would say nothing about fire behavior.")
    if df[target].dropna().nunique() <= 1:
        w.append(f"'{target}' is constant, so there is nothing to learn.")
    for f in features:
        if is_clock_column(f):
            w.append(f"'{f}' is a clock column used as an input. A model can use it as a place in the recording and look accurate without learning physics.")
    if split == "random" and looks_time_ordered(df):
        w.append("Rows are in time order. Random folds put neighbouring readings in both the training and test sets, which inflates scores. "
                 "Use the chronological split.")
    return w


# ------------------------------------------------------------ train and test
def _splits(task: str, X: pd.DataFrame, y: pd.Series, split: str, folds: int, seed: int):
    if split == "chronological":
        return list(TimeSeriesSplit(n_splits=folds).split(X))
    if task == "classification":
        return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y))
    return list(KFold(folds, shuffle=True, random_state=seed).split(X))


def train_eval(df: pd.DataFrame, features: list[str], target: str, task: str, seed: int = 0,
               split: str = "random", time_col: str | None = None, max_rows: int = 4000) -> dict:
    """Held-out evaluation against simple baselines, plus permutation importance.

    split='random' uses shuffled folds (fine for independent test rows, such as a table of experiments).
    split='chronological' trains on earlier rows and tests on later ones (needed for a time series).
    Baselines are what a model must beat: always the training mean (regression) or majority class (classification)."""
    if split not in ("random", "chronological"):
        raise ValueError("split must be 'random' or 'chronological'.")
    extra = [time_col] if time_col and time_col not in features + [target] else []
    d = df[features + [target] + extra].dropna()
    if len(d) < 20:
        raise ValueError(f"Only {len(d)} usable rows after dropping missing values. Need at least 20.")
    if split == "chronological" and time_col:
        d = d.sort_values(time_col, kind="stable")
    note = ""
    if len(d) > max_rows:
        step = int(np.ceil(len(d) / max_rows))
        d = d.iloc[::step]
        note = f"Used every {step}th row ({len(d):,} of the rows) to keep training fast. "
    X = d[features].select_dtypes(include=[np.number]).reset_index(drop=True)
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns selected.")
    y = d[target].reset_index(drop=True)
    folds = 5

    if task == "classification":
        if y.nunique() < 2:
            raise ValueError("Target has only one class.")
        if split == "random" and y.value_counts().min() < folds:
            folds = max(2, int(y.value_counts().min()))
        mk_base = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        mk_model = lambda: RandomForestClassifier(200, random_state=seed, min_samples_leaf=2, n_jobs=-1)
    else:
        mk_base = lambda: make_pipeline(StandardScaler(), Ridge())
        mk_model = lambda: RandomForestRegressor(200, random_state=seed, min_samples_leaf=2, n_jobs=-1)

    idx, p_base, p_model, p_naive = [], [], [], []
    for tr, te in _splits(task, X, y, split, folds, seed):
        ytr = y.iloc[tr]
        if task == "classification" and ytr.nunique() < 2:
            continue  # an early chronological fold may hold a single class: cannot fit
        b, m = mk_base().fit(X.iloc[tr], ytr), mk_model().fit(X.iloc[tr], ytr)
        idx.append(te)
        p_base.append(b.predict(X.iloc[te]))
        p_model.append(m.predict(X.iloc[te]))
        naive = ytr.mode().iloc[0] if task == "classification" else ytr.mean()
        p_naive.append(np.full(len(te), naive))
    if not idx:
        raise ValueError("No usable test folds. The target may have too few classes in the earlier rows.")
    idx = np.concatenate(idx)
    yt = y.iloc[idx]
    p_base, p_model, p_naive = (np.concatenate(a) for a in (p_base, p_model, p_naive))

    if task == "classification":
        acc = {k: round(accuracy_score(yt, v), 3) for k, v in
               {"majority-class accuracy": p_naive, "logistic regression accuracy": p_base, "random forest accuracy": p_model}.items()}
        metrics = dict(acc)
        if set(y.unique()) <= {0, 1}:
            metrics["false 'extinction' calls (RF)"] = int(((p_model == 1) & (yt == 0)).sum())
        beats = acc["random forest accuracy"] >= acc["majority-class accuracy"] + 0.03
    else:
        rmse = lambda v: float(np.sqrt(mean_squared_error(yt, v)))
        metrics = {
            "mean-baseline R2": round(r2_score(yt, p_naive), 3),
            "ridge R2": round(r2_score(yt, p_base), 3),
            "random forest R2": round(r2_score(yt, p_model), 3),
            "mean-baseline RMSE": round(rmse(p_naive), 4),
            "ridge RMSE": round(rmse(p_base), 4),
            "random forest RMSE": round(rmse(p_model), 4),
            "random forest MAE": round(mean_absolute_error(yt, p_model), 4),
        }
        beats = metrics["random forest R2"] > 0 and metrics["random forest RMSE"] <= 0.9 * metrics["mean-baseline RMSE"]

    if beats:
        verdict = ("On rows held out of training, the model beats the simple baseline. That shows the inputs and the target are related "
                   "in this file. It does not show cause, and it may not carry over to other experiments.")
    elif task == "regression" and metrics["random forest R2"] < 0:
        verdict = ("R2 below 0 means the model is worse than always guessing the average. It has not learned anything usable. "
                   "A small error (MAE) does not change that when the target barely varies.")
    else:
        verdict = "The model does not beat the simple baseline, so it has not learned anything usable from these inputs."

    fitted = mk_model().fit(X, y)
    n_imp = min(len(X), 1500)
    sel = np.linspace(0, len(X) - 1, n_imp).astype(int)
    imp = permutation_importance(fitted, X.iloc[sel], y.iloc[sel], n_repeats=5, random_state=seed)
    importances = (
        pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    how = ("trained on earlier rows and tested on later rows" if split == "chronological"
           else f"{folds}-fold shuffled cross-validation")
    return {
        "n_rows": int(len(d)),
        "n_eval": int(len(yt)),
        "folds": folds,
        "split": split,
        "split_text": how,
        "note": note,
        "beats_baseline": bool(beats),
        "verdict": verdict,
        "metrics": metrics,
        "importances": importances,
        "ranges": {c: (float(X[c].min()), float(X[c].max())) for c in X.columns},
        "model": fitted,
        "columns": list(X.columns),
    }
