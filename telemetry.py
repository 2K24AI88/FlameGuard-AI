"""Telemetry engine for SAFFIRE spacecraft-fire data.

Pure pandas/numpy, no UI. Built around the column layout seen in the SAFFIRE-III file
(RelativeSeconds, CygnusTimestamp, SaffireTimestamp + sensor channels).

Nothing here assumes physical units. The time unit of RelativeSeconds is a user choice
(see elapsed_seconds) because the source documentation has not been verified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIME_COLS = ["RelativeSeconds", "CygnusTimestamp", "SaffireTimestamp"]
SENTINEL_LIMIT = -1000.0  # radiometer readings below this are flagged, not trusted
MAX_RELATIVE = 1e9        # drops absurd timestamps (the file has a -63 trillion outlier)

CHANNEL_LABELS = {
    "P@sfsHk.Radiometer1value": "Radiometer 1",
    "P@sfsHk.Radiometer2value": "Radiometer 2",
    "P@sfsHk.Radiometer3value": "Radiometer 3",
    "P@sfsHk.Radiometer4value": "Radiometer 4",
    "P@sfsHk.Anemometer1": "Anemometer 1",
    "P@sfsHk.Anemometer2": "Anemometer 2",
    "P@sfsHk.FanTachometer": "Fan tachometer",
    "P@sfsHk.SmokeWireVoltage": "Smoke wire voltage",
    "P@sfsHk.SmokeWireCurrent": "Smoke wire current",
    "P@sfsHk.Sample1IgniterVoltage": "Sample 1 igniter voltage",
    "P@sfsHk.Sample1IgniterCurrent": "Sample 1 igniter current",
    "P@sfsHk.Sample2IgniterVoltage": "Sample 2 igniter voltage",
    "P@sfsHk.Sample2IgniterCurrent": "Sample 2 igniter current",
    "P@PCM_ETS_TEMP_SIG": "Temperature signal",
    "P@OPS_PCM_PRES_AVG_BY_TEMP": "Air pressure signal",
    "P@CDH_F1PCM_SMK_DET_SCAT_SENS_P": "Smoke detector F1 scatter",
    "P@CDH_F2PCM_SMK_DET_SCAT_SENS_P": "Smoke detector F2 scatter",
    "P@CDH_F2PCM_SMK_DET_OBS_SENS_P": "Smoke detector F2 obscuration",
}
LABEL_TO_CHANNEL = {v: k for k, v in CHANNEL_LABELS.items()}


def channel_label(col: str) -> str:
    return CHANNEL_LABELS.get(col, col)


# ----------------------------------------------------------------- cleaning
def clean_telemetry(raw: pd.DataFrame, max_relative: float = MAX_RELATIVE) -> tuple[pd.DataFrame, dict]:
    """Coerce time columns, drop rows with invalid timelines, sort by RelativeSeconds.

    Sensor values are kept as originals. Sentinel values are masked later, per use,
    so nothing is silently rewritten. Returns (clean_df, report)."""
    if "RelativeSeconds" not in raw.columns:
        raise ValueError("Missing required column 'RelativeSeconds'. Is this the SAFFIRE telemetry table?")
    df = raw.copy()
    for c in TIME_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = df["RelativeSeconds"].between(0, max_relative)
    if "CygnusTimestamp" in df.columns:
        keep &= df["CygnusTimestamp"] > 0
    cleaned = df.loc[keep].sort_values("RelativeSeconds").reset_index(drop=True)

    report = {
        "rows_raw": int(len(raw)),
        "rows_kept": int(len(cleaned)),
        "rows_dropped": int(len(raw) - len(cleaned)),
        "relative_min": float(cleaned["RelativeSeconds"].min()) if len(cleaned) else None,
        "relative_max": float(cleaned["RelativeSeconds"].max()) if len(cleaned) else None,
        "duplicate_timestamps": int(cleaned["RelativeSeconds"].duplicated().sum()) if len(cleaned) else 0,
    }
    return cleaned, report


def elapsed_seconds(df: pd.DataFrame, unit: str) -> pd.Series:
    """Seconds since the first valid row. unit is 'milliseconds' or 'seconds' (user choice)."""
    if unit not in ("milliseconds", "seconds"):
        raise ValueError("unit must be 'milliseconds' or 'seconds'")
    rel = df["RelativeSeconds"].astype(float)
    div = 1000.0 if unit == "milliseconds" else 1.0
    return (rel - rel.min()) / div


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """Columns that are not time fields and contain at least one numeric value."""
    out = []
    for c in df.columns:
        if c in TIME_COLS or c == "t_s":
            continue
        if pd.to_numeric(df[c], errors="coerce").notna().any():
            out.append(c)
    return out


def numeric_series(df: pd.DataFrame, col: str, mask_sentinel: bool = True) -> pd.Series:
    """Numeric version of a channel. Radiometer readings below SENTINEL_LIMIT become NaN."""
    s = pd.to_numeric(df[col], errors="coerce")
    if mask_sentinel and "Radiometer" in col:
        s = s.mask(s < SENTINEL_LIMIT)
    return s


# ------------------------------------------------------------------ quality
def quality_report(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or sensor_columns(df)
    rows = []
    n = len(df)
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        valid = v.dropna()
        below = int((v < SENTINEL_LIMIT).sum())
        rows.append(
            {
                "channel": c,
                "label": channel_label(c),
                "missing_pct": round(100 * v.isna().sum() / n, 2) if n else 0.0,
                "unique_values": int(valid.nunique()),
                "min": valid.min() if len(valid) else np.nan,
                "median": valid.median() if len(valid) else np.nan,
                "max": valid.max() if len(valid) else np.nan,
                f"below_{int(SENTINEL_LIMIT)}": below,
                "status": "constant: review" if valid.nunique() <= 1 else "varies",
            }
        )
    return pd.DataFrame(rows)


def sampling_summary(t_s: pd.Series) -> dict:
    dt = np.diff(np.asarray(t_s, dtype=float))
    dt = dt[dt > 0]
    if len(dt) == 0:
        return {"median_dt_s": None, "duration_s": float(t_s.max() - t_s.min()) if len(t_s) else 0.0}
    return {
        "median_dt_s": float(np.median(dt)),
        "p95_dt_s": float(np.percentile(dt, 95)),
        "duration_s": float(t_s.max() - t_s.min()),
    }


# -------------------------------------------------------- event detection
def detect_on_intervals(t, x, threshold: float | None = None, min_duration: float = 1.0) -> list[dict]:
    """Find intervals where a channel (e.g. igniter current) is 'on'.

    Default threshold is halfway between the 5th and 99.5th percentile of the channel.
    If a burst is too short to move the 99.5th percentile the default fails: pass a threshold."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = ~(np.isnan(t) | np.isnan(x))
    t, x = t[ok], x[ok]
    if len(x) < 3:
        return []
    if threshold is None:
        lo, hi = np.percentile(x, [5, 99.5])
        if hi - lo <= 1e-12:
            return []
        threshold = lo + 0.5 * (hi - lo)
    on = x > threshold
    edges = np.diff(on.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if on[0]:
        starts = [0] + starts
    if on[-1]:
        ends = ends + [len(x) - 1]
    out = []
    for s, e in zip(starts, ends):
        dur = float(t[e] - t[s])
        if dur >= min_duration:
            out.append({"start_s": float(t[s]), "end_s": float(t[e]), "duration_s": dur})
    return out


def first_response(t, x, onset: float, pre: float = 30.0, horizon: float = 300.0,
                   k: float = 5.0, m: int = 3, min_effect: float = 0.05) -> dict:
    """First sustained departure of a channel from its pre-onset baseline.

    Baseline = median of the `pre` seconds before onset. A response needs |x - baseline| above
    max(k * robust sigma, min_effect * channel range) for `m` consecutive samples, within `horizon`
    seconds after onset. This describes the signal only. It is not detector certification."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = ~(np.isnan(t) | np.isnan(x))
    t, x = t[ok], x[ok]
    if len(x) == 0:
        return {"status": "no valid data"}
    p5, p95 = np.percentile(x, [5, 95])
    rng = p95 - p5
    if np.nanmax(x) - np.nanmin(x) <= 1e-12:
        return {"status": "constant channel"}
    pre_mask = (t >= onset - pre) & (t < onset)
    if pre_mask.sum() < 10:
        return {"status": "not enough baseline samples"}
    base = float(np.median(x[pre_mask]))
    sigma = float(np.median(np.abs(x[pre_mask] - base)) * 1.4826)
    thr = max(k * sigma, min_effect * rng)
    post = (t >= onset) & (t <= onset + horizon)
    tp, xp = t[post], x[post]
    if len(xp) < m:
        return {"status": "not enough samples after onset"}
    dev = np.abs(xp - base) > thr
    run = np.convolve(dev.astype(int), np.ones(m, dtype=int), mode="valid")
    hit = np.where(run >= m)[0]
    peak = float(xp[np.argmax(np.abs(xp - base))] - base)
    if len(hit) == 0:
        return {"status": "no clear response", "baseline": base, "threshold": float(thr), "peak_change": peak}
    i = hit[0]
    return {
        "status": "response",
        "delay_s": float(tp[i] - onset),
        "baseline": base,
        "threshold": float(thr),
        "peak_change": peak,
        "direction": "up" if xp[i] > base else "down",
    }


def response_table(df: pd.DataFrame, onset: float, channels: list[str], **kw) -> pd.DataFrame:
    """Run first_response for several channels at one ignition onset."""
    rows = []
    for c in channels:
        r = first_response(df["t_s"], numeric_series(df, c), onset, **kw)
        rows.append({"channel": c, "label": channel_label(c), **r})
    out = pd.DataFrame(rows)
    for col in ["delay_s", "baseline", "threshold", "peak_change", "direction"]:
        if col not in out.columns:
            out[col] = np.nan
    return out


# ------------------------------------------------------------ comparison
def compare_channels(df: pd.DataFrame, a: str, b: str) -> dict:
    """Spearman and Pearson correlation on paired valid readings, sentinels masked."""
    d = pd.DataFrame({"a": numeric_series(df, a), "b": numeric_series(df, b)}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 10:
        return {"status": "not enough paired readings", "n": int(len(d))}
    if d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return {"status": "one channel is constant", "n": int(len(d))}
    return {
        "status": "ok",
        "n": int(len(d)),
        "spearman": float(d["a"].corr(d["b"], method="spearman")),
        "pearson": float(d["a"].corr(d["b"])),
    }


def decimate(df: pd.DataFrame, max_points: int = 6000) -> pd.DataFrame:
    """Stride-decimate for plotting only. Analysis always uses the full table."""
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step]


# ------------------------------------------------------------ time unit check
def infer_time_unit(df: pd.DataFrame) -> dict:
    """Work out the unit of RelativeSeconds by comparing its steps with CygnusTimestamp steps.

    CygnusTimestamp values near 1.5e12 are Unix milliseconds (that range is 2017). If RelativeSeconds advances by
    the same amount per row, it is in milliseconds too. Returns {'unit', 'ratio', 'start_utc', 'end_utc', 'reason'}.
    'unit' is None when the check cannot decide. This is evidence from the file itself, not documentation."""
    out = {"unit": None, "ratio": None, "start_utc": None, "end_utc": None, "reason": "CygnusTimestamp is not available to compare."}
    if not {"RelativeSeconds", "CygnusTimestamp"} <= set(df.columns):
        return out
    d = df[["RelativeSeconds", "CygnusTimestamp"]].dropna().sort_values("RelativeSeconds")
    d = d[(d["CygnusTimestamp"] > 0)]
    dr, dc = d["RelativeSeconds"].diff(), d["CygnusTimestamp"].diff()
    ok = (dr > 0) & (dc > 0)
    if ok.sum() < 10:
        out["reason"] = "Too few usable rows to compare the two clocks."
        return out
    ratio = float(np.median(dr[ok] / dc[ok]))
    out["ratio"] = round(ratio, 4)
    unix_ms = 1e11 < float(d["CygnusTimestamp"].median()) < 1e13
    if unix_ms:
        lo, hi = d["CygnusTimestamp"].min(), d["CygnusTimestamp"].max()
        out["start_utc"] = pd.to_datetime(lo, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M")
        out["end_utc"] = pd.to_datetime(hi, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M")
    if unix_ms and 0.9 <= ratio <= 1.1:
        out["unit"] = "milliseconds"
        out["reason"] = (f"RelativeSeconds advances by {ratio:g} per CygnusTimestamp millisecond, so it is in milliseconds "
                         "(the Cygnus clock reads as Unix milliseconds).")
    elif unix_ms and 0.0009 <= ratio <= 0.0011:
        out["unit"] = "seconds"
        out["reason"] = f"RelativeSeconds advances by {ratio:g} per CygnusTimestamp millisecond, so it is in seconds."
    else:
        out["reason"] = f"The two clocks do not line up (ratio {ratio:g}), so the unit is undecided. Pick it yourself."
    return out
