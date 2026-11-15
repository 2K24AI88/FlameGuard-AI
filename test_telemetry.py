"""Run: python test_telemetry.py

Uses a SYNTHETIC table that mimics the SAFFIRE column layout, with known events,
to check the analysis code finds them. It is a test fixture, not NASA data."""
import numpy as np
import pandas as pd

import telemetry as tl


def make_fixture(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 600, 1.0)  # 1 Hz, seconds
    n = len(t)
    igniter = np.where((t >= 100) & (t < 160), 3.0, 0.02) + rng.normal(0, 0.01, n)
    radio = 5 + rng.normal(0, 0.05, n)
    radio[t >= 120] += np.clip((t[t >= 120] - 120) * 0.5, 0, 15)  # responds 20 s after onset
    smoke = 1 + rng.normal(0, 0.02, n)
    smoke[t >= 145] += 4.0                                         # responds 45 s after onset
    flat = np.full(n, 7.0)
    noise_only = 3 + rng.normal(0, 0.05, n)
    radio_bad = radio.copy()
    radio_bad[rng.choice(n, 40, replace=False)] = -7e5             # sentinel values
    df = pd.DataFrame(
        {
            "RelativeSeconds": t * 1000.0,                          # ms, as the real file seems to be
            "CygnusTimestamp": 1.7e12 + t * 1000.0,
            "SaffireTimestamp": t,
            "P@sfsHk.Sample1IgniterCurrent": igniter,
            "P@sfsHk.Radiometer1value": radio_bad,
            "P@CDH_F2PCM_SMK_DET_SCAT_SENS_P": smoke,
            "P@sfsHk.FanTachometer": flat,
            "P@sfsHk.Anemometer1": noise_only,
        }
    )
    junk = df.iloc[:3].copy()
    junk["RelativeSeconds"] = [-6.3e13, np.nan, 5e12]
    junk["CygnusTimestamp"] = [0, 5, 5]
    return pd.concat([junk, df], ignore_index=True)


raw = make_fixture()
df, rep = tl.clean_telemetry(raw)
assert rep["rows_dropped"] == 3 and rep["rows_kept"] == 600, rep
assert df["RelativeSeconds"].is_monotonic_increasing
df["t_s"] = tl.elapsed_seconds(df, "milliseconds")
assert abs(df["t_s"].max() - 599) < 1e-9
assert abs(tl.elapsed_seconds(df, "seconds").max() - 599000) < 1e-6  # wrong unit changes the scale 1000x

# sensor columns, sentinel handling
cols = tl.sensor_columns(df)
assert "P@sfsHk.Radiometer1value" in cols and "RelativeSeconds" not in cols
assert (tl.numeric_series(df, "P@sfsHk.Radiometer1value").dropna() > -1000).all()
q = tl.quality_report(df)
assert q.set_index("channel").loc["P@sfsHk.FanTachometer", "status"] == "constant: review"
assert q.set_index("channel").loc["P@sfsHk.Radiometer1value", "below_-1000"] == 40

# ignition detection
iv = tl.detect_on_intervals(df["t_s"], tl.numeric_series(df, "P@sfsHk.Sample1IgniterCurrent"))
assert len(iv) == 1 and abs(iv[0]["start_s"] - 100) <= 1 and abs(iv[0]["duration_s"] - 60) <= 2, iv
assert tl.detect_on_intervals(df["t_s"], tl.numeric_series(df, "P@sfsHk.FanTachometer")) == []

# response delays: radiometer starts rising 20 s after onset (needs to cross the threshold), smoke 45 s
onset = iv[0]["start_s"]
res = tl.response_table(
    df, onset,
    ["P@sfsHk.Radiometer1value", "P@CDH_F2PCM_SMK_DET_SCAT_SENS_P", "P@sfsHk.FanTachometer", "P@sfsHk.Anemometer1"],
)
r = res.set_index("channel")
assert r.loc["P@CDH_F2PCM_SMK_DET_SCAT_SENS_P", "status"] == "response"
assert abs(r.loc["P@CDH_F2PCM_SMK_DET_SCAT_SENS_P", "delay_s"] - 45) <= 2, r
rad = r.loc["P@sfsHk.Radiometer1value"]
assert rad["status"] == "response" and 20 <= rad["delay_s"] <= 30, rad  # slow ramp crosses threshold a few s late
assert r.loc["P@sfsHk.FanTachometer", "status"] == "constant channel"
assert r.loc["P@sfsHk.Anemometer1", "status"] != "response", "noise alone must not count as a response"

# correlation
c = tl.compare_channels(df, "P@sfsHk.Radiometer1value", "P@CDH_F2PCM_SMK_DET_SCAT_SENS_P")
assert c["status"] == "ok" and c["spearman"] > 0.5 and c["n"] == 600 - 40
assert tl.compare_channels(df, "P@sfsHk.FanTachometer", "P@sfsHk.Anemometer1")["status"] == "one channel is constant"

# decimate keeps the end points' neighbourhood and caps size
assert len(tl.decimate(df, 100)) <= 100

# missing key column is a clear error
try:
    tl.clean_telemetry(pd.DataFrame({"a": [1]}))
    raise SystemExit("expected ValueError")
except ValueError:
    pass

print("telemetry tests passed")
print(res[["label", "status", "delay_s"]].to_string(index=False))
