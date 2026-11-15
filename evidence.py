"""Evidence layer: measured results, coverage map, nearest-evidence scenario check, briefing.

All numbers come from data/measured_results.csv, where each row carries its source and locator.
Nothing here predicts fire behaviour. It shows what has been measured and where nothing has.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

PRESETS = {
    "Normal air (about 21% O2)": {"o2": 21.0, "flow": 5.0, "pressure": None},
    "Saffire-V exploration-style atmosphere (26.9% O2, 761 hPa)": {"o2": 26.9, "flow": 20.0, "pressure": 761.0},
    "Reduced-oxygen case (about 16% O2)": {"o2": 16.0, "flow": 2.0, "pressure": None},
    "Lunar habitat or rover (30% O2, 10.2 psi = 703 hPa)": {"o2": 30.0, "flow": 20.0, "pressure": 703.0},
    "Spacesuit (100% O2, 4.3 psi = 296 hPa)": {"o2": 100.0, "flow": 20.0, "pressure": 296.0},
}


def load_measured(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "measured_results.csv")


def condition_rows(meas: pd.DataFrame) -> pd.DataFrame:
    """Rows that define a tested condition in both oxygen and flow."""
    d = meas.dropna(subset=["o2_pct", "flow_cm_s"])
    return d[d["gravity"].str.contains("micro")].reset_index(drop=True)


def nearest_evidence(meas: pd.DataFrame, o2: float, flow: float, k: int = 3,
                     o2_tol: float = 2.0, flow_tol: float = 5.0) -> pd.DataFrame:
    """Closest measured conditions to (o2, flow), with raw differences and a coverage flag.

    Ordering uses distance scaled by the tolerances. Tolerances are choices, not physics."""
    d = condition_rows(meas).copy()
    if d.empty:
        return d
    d["d_o2"] = d["o2_pct"] - o2
    d["d_flow"] = d["flow_cm_s"] - flow
    d["closeness"] = np.sqrt((d["d_o2"] / o2_tol) ** 2 + (d["d_flow"] / flow_tol) ** 2)
    d["within_tolerance"] = (d["d_o2"].abs() <= o2_tol) & (d["d_flow"].abs() <= flow_tol)
    return d.sort_values("closeness").head(k).reset_index(drop=True)


def coverage_gaps(meas: pd.DataFrame, o2: float, flow: float, pressure: float | None,
                  o2_tol: float = 2.0, flow_tol: float = 5.0) -> list[str]:
    """Plain-language list of what is NOT covered for this scenario."""
    near = nearest_evidence(meas, o2, flow, k=len(meas), o2_tol=o2_tol, flow_tol=flow_tol)
    gaps = []
    if near.empty or not near["within_tolerance"].any():
        gaps.append(
            f"No measured microgravity condition in this collection is within {o2_tol:g} points of oxygen and "
            f"{flow_tol:g} cm/s of flow of this scenario."
        )
    if pressure is not None:
        p = meas["pressure_hpa"].dropna()
        if p.empty or (p - pressure).abs().min() > 100:
            gaps.append("No measured row here is within 100 hPa of the scenario pressure.")
    elif meas["pressure_hpa"].notna().sum() < len(meas) / 2:
        gaps.append("Most rows do not state the test pressure, so pressure effects cannot be compared.")
    gaps.append("Moon (about 1/6 g) and Mars (about 3/8 g) gravity: none of these rows were measured in partial gravity.")
    return gaps


def oxygen_margin(meas: pd.DataFrame) -> dict | None:
    """Ground blow-off oxygen minus lowest microgravity oxygen with steady spread (same rod)."""
    ground = meas.loc[meas["id"] == "b2-p2", "value"]
    micro = meas.loc[meas["id"] == "b2-p1", "value"]
    if ground.empty or micro.empty:
        return None
    g, m = float(ground.iloc[0]), float(micro.iloc[0])
    return {"ground_o2": g, "micro_o2": m, "margin": round(g - m, 2)}


def add_measured_counts(cat: pd.DataFrame, meas: pd.DataFrame) -> pd.DataFrame:
    """Add measured_points: how many measured rows in this collection belong to each catalog entry."""
    counts = meas.groupby("catalog_id").size()
    out = cat.copy()
    out["measured_points"] = out["id"].map(counts).fillna(0).astype(int)
    return out


def build_corpus(cat: pd.DataFrame, ins: pd.DataFrame, meas: pd.DataFrame) -> pd.DataFrame:
    """One searchable table: catalog entries, insights and measured findings."""
    cols = ["name", "program", "topic", "fuel_state", "description", "platform", "source_url", "kind"]
    a = cat.assign(kind="catalog")[cols]
    b = pd.DataFrame(
        {
            "name": ins["question"],
            "program": ins["research"],
            "topic": "insight",
            "fuel_state": "",
            "description": ins["evidence"] + " " + ins["limitation"],
            "platform": "NASA report",
            "source_url": ins["source"],
            "kind": "insight",
        }
    )
    vals = meas["value"].map(lambda v: "" if pd.isna(v) else f"{v:g}")
    c = pd.DataFrame(
        {
            "name": meas["experiment"] + ": " + meas["metric"],
            "program": meas["experiment"],
            "topic": meas["metric"],
            "fuel_state": meas["material"],
            "description": (meas["metric"] + " " + vals + " " + meas["unit"].fillna("") + ". " + meas["note"]).str.strip(),
            "platform": "measured row",
            "source_url": meas["source_url"],
            "kind": "measured",
        }
    )
    return pd.concat([a, b, c], ignore_index=True)


def build_brief(o2: float, flow: float, pressure: float | None, near: pd.DataFrame, gaps: list[str],
                margin: dict | None) -> str:
    """Markdown briefing for a scenario. States evidence and gaps. Makes no safety call."""
    lines = [
        "# Fire-evidence brief",
        "",
        f"**Scenario:** oxygen {o2:g}%, flow {flow:g} cm/s"
        + (f", pressure {pressure:g} hPa" if pressure is not None else ", pressure not specified"),
        "",
        "## Nearest measured conditions",
    ]
    if near.empty:
        lines.append("- None in this collection.")
    for _, r in near.iterrows():
        val = "" if pd.isna(r["value"]) else f" ({r['metric']}: {r['value']:g} {r['unit']})"
        lines.append(
            f"- {r['experiment']}, {r['material']}: O2 {r['o2_pct']:g}%, flow {r['flow_cm_s']:g} cm/s, {r['flow_direction']}{val}. "
            f"Differences: {r['d_o2']:+.1f} O2 points, {r['d_flow']:+.1f} cm/s. Source: {r['source']} ({r['source_locator']})."
        )
    lines += ["", "## Not covered"] + [f"- {g}" for g in gaps]
    if margin:
        lines += [
            "",
            "## Why ground ratings may not be enough",
            f"- For one PMMA rod, steady spread occurred at {margin['micro_o2']:g}% O2 in microgravity versus a "
            f"{margin['ground_o2']:g}% ground blow-off limit (difference {margin['margin']:g} O2 points). One material only.",
        ]
    lines += [
        "",
        "## Caution",
        "This brief lists measured conditions near the scenario. It does not predict a fire or rate any design. "
        "Not for flight or safety decisions.",
    ]
    return "\n".join(lines)
