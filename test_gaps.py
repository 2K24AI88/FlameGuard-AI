"""Run: python test_gaps.py   (BASS-II matrix, research-gap counts, time-unit check, Model Lab safeguards)"""
import numpy as np
import pandas as pd

import flame_core as fc
import gaps as gp
import telemetry as tl

# --- BASS-II matrix is the converted workbook, values untouched
b2 = gp.load_bass2()
assert len(b2) == 129 and b2["investigator_sheet"].nunique() == 6
assert b2["o2_initial_pct"].between(13.6, 22.2).all()
f = b2.set_index("test_no")
assert f.loc["F1", "co_initial_ppm"] == 0 and f.loc["F1", "co_final_ppm"] == 2
assert f.loc["F2", "co_initial_ppm"] == 105 and f.loc["F2", "co_final_ppm"] == 82   # CO FELL in F2: final CO alone misleads
assert b2["co_final_ppm"].isna().sum() == 9

# --- gap counts are computed, add up, and never invent partial-gravity tests
meas = fc.load_measurements()
tests = gp.condition_tests(meas, b2)
gm, pm = gp.gravity_matrix(tests), gp.pressure_matrix(tests)
assert gm.loc["Microgravity"].sum() == len(tests[tests["gravity"] == "Microgravity"])
assert gm.loc["Moon gravity (0.16 g)"].sum() == 0 and gm.loc["Mars gravity (0.38 g)"].sum() == 0
assert pm.values.sum() == gm.loc["Microgravity"].sum()                 # every micro test lands in exactly one pressure cell
micro_meas_no_p = int(((meas["gravity"].str.contains("micro")) & meas["pressure_kpa"].isna()).sum())
assert pm[gp.NO_P].sum() == len(b2) + micro_meas_no_p                  # BASS-II has no pressure column; Saffire-I/II rows state none
mh = gp.moon_habitat_summary(tests)
assert mh["partial_gravity_tests"] == 0
assert (mh["both"]["o2"] >= gp.MOON_O2_MIN).all() and (mh["both"]["pressure_kpa"] <= gp.MOON_P_MAX_KPA).all()
assert set(mh["both"]["experiment"]) <= {"Saffire VI"} and len(mh["both"]) >= 1   # Saffire VI VI-2/VI-3 sit in that atmosphere
assert abs(gp.MOON_P_MAX_KPA - 70.3) < 0.05
d = gp.describe_gap(tests, "Moon gravity (0.16 g)", "30% or more")
assert d["n"] == 0 and d["partial_gravity_notes"] and "template" not in d["question"].lower()
d2 = gp.describe_gap(tests, "Microgravity", "30% or more", gp.P_BANDS[0][0])
assert d2["n"] == len(mh["both"]) and len(d2["related"]) >= 1
near = gp.nearest_tests(tests[tests["pressure_kpa"].notna()], 30.0, 54.0, k=2)
assert near.iloc[0]["experiment"] == "Saffire VI" and near.iloc[0]["within_tolerance"]

# --- time unit: inferred from the file's own clocks, not guessed
t = np.arange(0, 120_000, 500)
ms = pd.DataFrame({"RelativeSeconds": t, "CygnusTimestamp": 1_496_691_300_000 + t})
assert tl.infer_time_unit(ms)["unit"] == "milliseconds" and tl.infer_time_unit(ms)["start_utc"].startswith("2017-06-05")
sec = pd.DataFrame({"RelativeSeconds": t / 1000.0, "CygnusTimestamp": 1_496_691_300_000 + t})
assert tl.infer_time_unit(sec)["unit"] == "seconds"
assert tl.infer_time_unit(pd.DataFrame({"RelativeSeconds": t}))["unit"] is None
odd = pd.DataFrame({"RelativeSeconds": t * 7, "CygnusTimestamp": 1_496_691_300_000 + t})
assert tl.infer_time_unit(odd)["unit"] is None                       # clocks disagree: undecided, not guessed

# --- Model Lab safeguards
n = 600
rng = np.random.default_rng(1)
tm = np.arange(n) * 500 + 1_496_691_300_000
walk = np.cumsum(rng.normal(size=n))
df = pd.DataFrame({"CygnusTimestamp": tm, "sensor_a": walk + rng.normal(0, 0.1, n), "sensor_b": rng.normal(size=n),
                   "P@GNCImuSensor.imu[0].currentMeas_Bdy.acc.elements[2]": rng.normal(0, 1e-4, n), "radio": np.r_[np.full(500, -4002.0), rng.uniform(0, 100, 100)]})
assert fc.looks_time_ordered(df) and not fc.looks_time_ordered(df.drop(columns="CygnusTimestamp"))
assert fc.suggest_target(df) == "sensor_a"                           # never the accelerometer or the clock
imu = "P@GNCImuSensor.imu[0].currentMeas_Bdy.acc.elements[2]"
w = fc.target_warnings(df, imu, ["CygnusTimestamp", "sensor_b"], "random")
assert any("motion" in x for x in w) and any("clock" in x for x in w) and any("time order" in x for x in w)
assert not fc.target_warnings(df, "sensor_a", ["sensor_b"], "chronological")
insp = fc.inspect_columns(df).set_index("column")
assert "no data" in insp.loc["radio", "note"] and "clock" in insp.loc["CygnusTimestamp", "note"]
masked, counts = fc.mask_sentinels(df)
assert counts == {"radio": 500} and masked["CygnusTimestamp"].equals(df["CygnusTimestamp"]) and masked["radio"].isna().sum() == 500
# an unlearnable target must be reported as not beating the baseline
r = fc.train_eval(df, ["sensor_b"], imu, "regression", split="chronological", time_col="CygnusTimestamp")
assert not r["beats_baseline"] and "not" in r["verdict"].lower() and r["split"] == "chronological"
# random folds flatter a drifting series; the chronological test shows the truth
df2 = pd.DataFrame({"t": np.arange(n), "x": walk + rng.normal(0, 0.1, n), "y": walk})
rnd = fc.train_eval(df2, ["x"], "y", "regression", split="random")["metrics"]["random forest R2"]
chr_ = fc.train_eval(df2, ["x"], "y", "regression", split="chronological", time_col="t")["metrics"]["random forest R2"]
assert rnd > chr_ + 0.2, (rnd, chr_)
try:
    fc.train_eval(df2, ["x"], "y", "regression", split="sideways")
    raise SystemExit("expected ValueError")
except ValueError:
    pass
print("gap tests passed", {"random R2": rnd, "chronological R2": chr_})
