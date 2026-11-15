"""Run: python test_evidence.py"""
import pandas as pd

import evidence as ev
import flame_core as fc

meas = ev.load_measured()
cat = fc.load_catalog()
ins = pd.read_csv(ev.DATA_DIR / "insights.csv")

# every row is sourced and ids are unique
assert meas["id"].is_unique
for col in ["source", "source_url", "source_locator", "verification", "metric", "catalog_id"]:
    assert meas[col].notna().all() and (meas[col].astype(str).str.len() > 0).all(), col
assert set(meas["catalog_id"]) <= set(cat["id"]), set(meas["catalog_id"]) - set(cat["id"])

# the headline oxygen margin is computed from the rows, and matches what the poster states (2.4)
m = ev.oxygen_margin(meas)
assert m == {"ground_o2": 16.0, "micro_o2": 13.6, "margin": 2.4}, m

# SAFFIRE-I values agree with the NASA table (420 s concurrent, 70 s opposed, 20 cm/s)
s1 = meas.set_index("id")
assert s1.loc["sf1-s1", "value"] == 420 and s1.loc["sf1-s1", "flow_direction"] == "concurrent"
assert s1.loc["sf1-s2", "value"] == 70 and s1.loc["sf1-s2", "flow_direction"] == "opposed"
assert (s1.loc[["sf1-s1", "sf1-s2"], "flow_cm_s"] == 20).all()

# condition rows are microgravity only and have both coordinates
cr = ev.condition_rows(meas)
assert cr["o2_pct"].notna().all() and cr["flow_cm_s"].notna().all()
assert not cr["gravity"].str.contains("1 g").any()

# nearest evidence: a scenario at 21% O2 and 1 cm/s sits on the BASS-II thin PMMA points
near = ev.nearest_evidence(meas, 21.0, 1.0, k=3)
assert (near["catalog_id"] == "bass-ii").all() and near["within_tolerance"].all(), near
# a scenario with no nearby data is flagged
far = ev.nearest_evidence(meas, 35.0, 200.0, k=3)
assert not far["within_tolerance"].any()
gaps = ev.coverage_gaps(meas, 35.0, 200.0, None)
assert any("No measured microgravity condition" in g for g in gaps)
assert any("Moon" in g for g in gaps)
assert not any("No measured microgravity condition" in g for g in ev.coverage_gaps(meas, 21.0, 1.0, None))

# pressure gap logic
assert any("pressure" in g.lower() for g in ev.coverage_gaps(meas, 26.9, 20.0, 400.0))
assert not any("within 100 hPa" in g for g in ev.coverage_gaps(meas, 26.9, 20.0, 761.0))

# counts and corpus
cc = ev.add_measured_counts(cat, meas)
assert cc.set_index("id").loc["bass-ii", "measured_points"] == 10
assert cc.set_index("id").loc["same", "measured_points"] == 0
corpus = ev.build_corpus(cat, ins, meas)
assert len(corpus) == len(cat) + len(ins) + len(meas)
r = fc.Retriever(corpus)
top = r.search("oxygen margin ground test PMMA rod", k=3)
assert not top.empty and top.iloc[0]["kind"] in ("insight", "measured"), top[["name", "kind", "match"]]

# brief states evidence and refuses to make a safety call
brief = ev.build_brief(21.0, 1.0, None, near, ev.coverage_gaps(meas, 21.0, 1.0, None), m)
assert "Not covered" in brief and "does not predict a fire" in brief and "2.4" in brief
assert "safe" not in brief.lower().replace("safety decisions", "")

# presets stay inside plausible ranges
for p in ev.PRESETS.values():
    assert 10 < p["o2"] <= 100 and p["flow"] > 0   # 100% is the spacesuit preset
    assert p["pressure"] is None or 200 < p["pressure"] < 1100

print("evidence tests passed")
print(near[["experiment", "material", "o2_pct", "flow_cm_s", "d_o2", "d_flow", "within_tolerance"]].to_string(index=False))
