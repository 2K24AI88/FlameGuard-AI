"""Research Gap Intelligence: where is evidence recorded, thin, or missing?

Everything here is COUNTED from the stored test rows (data/measurements.csv and data/bass2_matrix.csv).
Nothing is assumed. A count means "tests recorded in this collection", not "verified" or "good quality".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# Lunar habitat per the NASA Space Apps 2026 challenge text: 30% oxygen or more, 10.2 psi or less.
MOON_O2_MIN = 30.0
MOON_P_MAX_KPA = round(10.2 * 6.894757, 1)  # 70.3 kPa

O2_BANDS = [("below 19%", 0.0, 19.0), ("19 to 21.5%", 19.0, 21.5), ("21.5 to 25%", 21.5, 25.0),
            ("25 to 30%", 25.0, 30.0), ("30% or more", 30.0, 101.0)]
P_BANDS = [(f"{MOON_P_MAX_KPA:g} kPa or less (lunar-habitat range)", 0.0, MOON_P_MAX_KPA + 1e-9),
           (f"{MOON_P_MAX_KPA:g} to 90 kPa", MOON_P_MAX_KPA + 1e-9, 90.0),
           ("90 kPa or more", 90.0, 1e9)]
NO_P = "pressure not recorded"
GRAVITY_ROWS = ["Microgravity", "Mars gravity (0.38 g)", "Moon gravity (0.16 g)"]

# What NASA says about partial gravity. Source: NASA Space Apps 2026 challenge text; NASA Glenn FM2 page.
PARTIAL_GRAVITY_NOTES = [
    ("FM2", "Planned lunar-surface experiment; no flight schedule or results on NASA's page "
            "(https://www.nasa.gov/glenn/glenn-expertise-space-exploration/physical-sciences-program/combustion-science/flammability-of-materials-on-the-moon-fm2/)."),
    ("LUCI", "Named in NASA's challenge text as an expected source of future partial-gravity data."),
    ("ACME E-FIELD Flames", "Per NASA's challenge text, mimics partial gravity through the electric field's effect on flame ions. "
                            "It is a proxy, not a test at lunar or Martian gravity."),
]


def load_bass2(path: Path | None = None) -> pd.DataFrame:
    p = path or DATA_DIR / "bass2_matrix.csv"
    if not Path(p).exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _gravity_label(text) -> str:
    t = str(text).lower()
    if "micro" in t:
        return "Microgravity"
    if "moon" in t or "lunar" in t:
        return "Moon gravity (0.16 g)"
    if "mars" in t:
        return "Mars gravity (0.38 g)"
    return "Normal gravity (ground)"


def condition_tests(meas: pd.DataFrame, bass2: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per recorded test: experiment, id, oxygen (%), pressure (kPa or NaN), gravity, status."""
    parts = []
    if len(meas):
        parts.append(pd.DataFrame({
            "source": "measurements.csv", "experiment": meas["experiment"], "test_id": meas["test_id"].astype(str),
            "o2": pd.to_numeric(meas["o2_num"], errors="coerce"), "pressure_kpa": pd.to_numeric(meas["pressure_kpa"], errors="coerce"),
            "gravity": meas["gravity"].map(_gravity_label), "status": meas["status"],
        }))
    if bass2 is not None and len(bass2):
        parts.append(pd.DataFrame({
            "source": "bass2_matrix.csv", "experiment": "BASS-II", "test_id": bass2["test_no"].astype(str),
            "o2": pd.to_numeric(bass2["o2_initial_pct"], errors="coerce"), "pressure_kpa": np.nan,
            "gravity": "Microgravity", "status": "Read from supplied file",
        }))
    if not parts:
        return pd.DataFrame(columns=["source", "experiment", "test_id", "o2", "pressure_kpa", "gravity", "status"])
    t = pd.concat(parts, ignore_index=True)
    return t[t["o2"].notna()].reset_index(drop=True)


def o2_band(value: float) -> str | None:
    for name, lo, hi in O2_BANDS:
        if lo <= value < hi:
            return name
    return None


def p_band(value) -> str:
    if value is None or pd.isna(value):
        return NO_P
    for name, lo, hi in P_BANDS:
        if lo <= value < hi:
            return name
    return NO_P


def status_word(n: int) -> str:
    return "none recorded" if n == 0 else ("thin (1-2 tests)" if n < 3 else f"{n} tests")


def gravity_matrix(tests: pd.DataFrame) -> pd.DataFrame:
    """Rows: gravity levels. Columns: oxygen bands. Cells: number of recorded tests."""
    out = pd.DataFrame(0, index=GRAVITY_ROWS, columns=[b[0] for b in O2_BANDS])
    for _, r in tests.iterrows():
        b = o2_band(r["o2"])
        if r["gravity"] in out.index and b:
            out.loc[r["gravity"], b] += 1
    return out


def pressure_matrix(tests: pd.DataFrame) -> pd.DataFrame:
    """Microgravity tests only. Rows: oxygen bands. Columns: pressure bands plus 'pressure not recorded'."""
    cols = [b[0] for b in P_BANDS] + [NO_P]
    out = pd.DataFrame(0, index=[b[0] for b in O2_BANDS], columns=cols)
    for _, r in tests[tests["gravity"] == "Microgravity"].iterrows():
        b = o2_band(r["o2"])
        if b:
            out.loc[b, p_band(r["pressure_kpa"])] += 1
    return out


def moon_habitat_summary(tests: pd.DataFrame) -> dict:
    """How many recorded tests fall in or near the lunar-habitat atmosphere NASA describes."""
    micro = tests[tests["gravity"] == "Microgravity"]
    both = micro[(micro["o2"] >= MOON_O2_MIN) & (micro["pressure_kpa"] <= MOON_P_MAX_KPA)]
    o2_only = micro[micro["o2"] >= MOON_O2_MIN]
    p_only = micro[micro["pressure_kpa"] <= MOON_P_MAX_KPA]
    return {"both": both.reset_index(drop=True), "o2_only": o2_only.reset_index(drop=True), "p_only": p_only.reset_index(drop=True),
            "partial_gravity_tests": int((~tests["gravity"].isin(["Microgravity", "Normal gravity (ground)"])).sum())}


def nearest_tests(tests: pd.DataFrame, o2: float, pressure_kpa: float | None = None, k: int = 5,
                  o2_tol: float = 2.0, p_tol: float = 10.0) -> pd.DataFrame:
    """Closest recorded microgravity tests in oxygen (and pressure, where both are known)."""
    d = tests[tests["gravity"] == "Microgravity"].copy()
    d["d_o2"] = (d["o2"] - o2).round(2)
    if pressure_kpa is not None:
        d["d_pressure_kpa"] = (d["pressure_kpa"] - pressure_kpa).round(1)
        dist = np.sqrt((d["d_o2"] / o2_tol) ** 2 + (d["d_pressure_kpa"].fillna(1e3) / p_tol) ** 2)
        d["within_tolerance"] = (d["d_o2"].abs() <= o2_tol) & (d["d_pressure_kpa"].abs() <= p_tol)
    else:
        dist = d["d_o2"].abs() / o2_tol
        d["d_pressure_kpa"] = np.nan
        d["within_tolerance"] = d["d_o2"].abs() <= o2_tol
    d["_dist"] = dist
    return d.sort_values("_dist").head(k).drop(columns="_dist").reset_index(drop=True)


def describe_gap(tests: pd.DataFrame, gravity: str, band: str, pband: str | None = None) -> dict:
    """Explain one cell: what is missing, which recorded tests are nearest, why they cannot answer it,
    and a TEMPLATE research question (not generated science)."""
    lo, hi = next((a, b) for n, a, b in O2_BANDS if n == band)
    mid = 35.0 if hi > 100 else (lo + hi) / 2
    g = tests[tests["gravity"] == gravity]
    g = g[(g["o2"] >= lo) & (g["o2"] < hi)]
    if pband and pband != NO_P:
        plo, phi = next((a, b) for n, a, b in P_BANDS if n == pband)
        g = g[(g["pressure_kpa"] >= plo) & (g["pressure_kpa"] < phi)]
    n = len(g)
    p_mid = None
    if pband and pband != NO_P:
        plo, phi = next((a, b) for n_, a, b in P_BANDS if n_ == pband)
        # "or less" band: compare to its upper bound (NASA's 10.2 psi); other bands use their middle
        p_mid = MOON_P_MAX_KPA if plo == 0 else (plo + min(phi, 101.3)) / 2
    micro = tests[tests["gravity"] == "Microgravity"]
    related = nearest_tests(micro, mid, p_mid, k=3)
    cell = f"{gravity}, oxygen {band}" + (f", {pband}" if pband else "")
    if gravity != "Microgravity":
        why = ("No partial-gravity tests are recorded here, and NASA's challenge text says PSI has none yet. "
               "Microgravity and normal-gravity results can bound partial gravity, but they cannot replace a test at that gravity.")
    elif n == 0:
        why = "No recorded test sits in this cell. The nearest ones differ in oxygen or pressure, and flame behavior changes with both."
    elif n < 3:
        why = "One or two tests cannot show a trend, and they differ in material and sample size."
    else:
        why = "There are several tests, but they differ in material, flow and sample size, so the cell is covered only loosely."
    question = (f"How do flame spread and extinction of spacecraft solid materials change in {gravity.lower()} at {band} oxygen"
                + (f" and {pband}" if pband and pband != NO_P else "") + "?")
    return {"cell": cell, "n": n, "status": status_word(n), "related": related, "why": why,
            "question": question, "partial_gravity_notes": PARTIAL_GRAVITY_NOTES if gravity != "Microgravity" else []}
