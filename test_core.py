"""Run: python test_core.py"""
import pandas as pd

import flame_core as fc

cat = fc.load_catalog()
assert len(cat) >= 15 and cat["id"].is_unique
assert set(fc.SCORE_COLS) <= set(cat.columns)
vals = pd.unique(cat[fc.SCORE_COLS].to_numpy().ravel())
assert set(vals) <= {0, 0.5, 1, 0.0, 1.0}, vals
assert cat["source_url"].str.startswith("http").all()
assert cat["source_read"].isin(["read", "overview only", "name only"]).all()
# name-only entries must not claim to have read their source
assert (cat.loc[cat["source_read"] == "name only", "description"].str.contains("details not read")).all()
assert (cat.loc[cat["source_read"] == "overview only", "description"].str.contains("overview only")).all()

# facts / insights / measurements integrity
facts = fc.load_facts()
assert facts["id"].is_unique and facts["statement"].str.len().min() > 30
assert facts["source_url"].str.startswith("http").all()
ids = set(cat["id"])
for cell in facts["experiment_ids"]:
    for e in cell.split(";"):
        assert e in ids, f"fact refers to unknown experiment '{e}'"
ins = fc.load_insights()
fact_ids = set(facts["id"])
for cell in ins["fact_ids"]:
    for f in cell.split(";"):
        assert f in fact_ids, f"insight cites unknown fact {f}"
meas = fc.load_measurements()
assert meas["test_id"].is_unique and meas["catalog_id"].notna().all() and set(meas["catalog_id"]) <= ids
row = meas.set_index("test_id")
assert row.loc["1-1", "burn_s"] == 420 and row.loc["1-2", "burn_s"] == 70   # Saffire-I Table I
assert row.loc["1-1", "direction"] == "concurrent" and row.loc["1-2", "direction"] == "opposed"
assert abs(row.loc["1-1", "o2_num"] - 21.6) < 1e-9
assert row.loc["2-1", "burn_s"] != row.loc["2-1", "burn_s"]  # NaN: not stated, not invented
lim = fc.load_limits()
assert not lim["id"].str.startswith("sf1-").any() and set(lim["catalog_id"]) <= ids

# ranking: changing weights must change the order; scores are 0-100 and sorted
none = {c: 0 for c in fc.SCORE_COLS}
a = fc.rank_experiments(cat, {**none, "smoke_detection": 1})
b = fc.rank_experiments(cat, {**none, "suppression_extinction": 1})
assert a["id"].tolist() != b["id"].tolist()
assert a["score"].is_monotonic_decreasing and a["score"].between(0, 100).all()
assert a.iloc[0]["id"] in ("saffire-iv-vi", "same", "same-r"), a.iloc[0]["id"]
assert b.iloc[0]["id"] in ("bass", "bass-ii", "flex"), b.iloc[0]["id"]
assert fc.rank_experiments(cat, {**none, "partial_gravity": 1})["score"].max() == 0  # honest gap

# retrieval over catalog and facts; nonsense returns nothing
r = fc.Retriever(cat)
assert r.search("smoke detector particles", k=3).iloc[0]["id"] in ("same", "same-r")
assert r.search("zzzz qqqq", k=3).empty
fs = fc.FactSearch(facts)
top = fs.search("do light scattering smoke detectors work for small particles", k=3)
assert set(top["id"]) & {"F20", "F21"}, top[["id", "match"]]
assert fs.search("zzzz qqqq").empty

# brief uses only stored content and cites sources
brief = fc.make_brief(["saffire", "same"], cat, facts, meas, lim)
assert "420" in brief and "https://ntrs.nasa.gov" in brief and "Not for flight" in brief
assert "Saffire-I to III" in brief and "SAME" in brief

# evidence composer and comparison
assert fc.evidence_strength(3, 1) == "Strong" and fc.evidence_strength(3, 0) == "Moderate"
assert fc.evidence_strength(3, 1, 1) == "Moderate"  # a caveat on a matched finding blocks Strong
assert fc.evidence_strength(2, 5) == "Moderate" and fc.evidence_strength(1, 9) == "Limited"
card = fc.answer_card("do smoke detectors work for small particles", fs, cat, meas, lim)
assert card["strength"] in ("Moderate", "Limited") and len(card["hits"]) >= 1
assert any("Moon or Mars" in g for g in card["gaps"]), card["gaps"]  # the gap is always surfaced
# loosely related findings must not inflate the rating: the ground-test question rests on NASA summaries
g = fc.answer_card("Are standard ground flammability tests enough for spacecraft materials?", fs, cat, meas, lim)
assert "F44" in set(g["hits"]["id"]) and g["strength"] != "Strong", (g["strength"], list(g["hits"]["id"]))
assert (g["hits"]["match"] >= 0.5 * g["hits"]["match"].max()).all()
p = fc.answer_card("Does lower pressure slow flame spread?", fs, cat, meas, lim)
assert "F03" in set(p["hits"]["id"]) and p["strength"] != "Strong", (p["strength"], list(p["hits"]["id"]))
# unconfirmed rows never count as support
fake_lim = lim.assign(catalog_id="bass-ii")
assert fc.answer_card("how does airflow affect extinction of solid fuel", fs, cat, meas, fake_lim)["n_measured"] == 0
none_card = fc.answer_card("zzzz qqqq", fs, cat, meas)
assert none_card["strength"] is None and none_card["gaps"]
cmpd = fc.compare_experiments("saffire", "same", cat, facts, meas)
assert cmpd["measured"][0] > 0 and cmpd["measured"][1] == 0
assert any("no measured test rows" in w for w in cmpd["warnings"])
assert cmpd["criteria"]["differs"].any() and len(cmpd["criteria"]) == len(fc.SCORE_COLS)
try:
    fc.compare_experiments("saffire", "nope", cat, facts, meas)
    raise SystemExit("expected ValueError")
except ValueError:
    pass

# verification labels are honest and visible
vs = fc.verification_summary(cat, meas, lim)
assert vs["catalog_read"] + vs["catalog_overview"] + vs["catalog_name_only"] == vs["catalog_total"]
assert vs["rows_confirmed"] + vs["rows_two_readings"] + vs["rows_read_once"] + vs["rows_unconfirmed"] == vs["rows_total"]
assert vs["rows_confirmed"] == 0  # nothing is marked human-confirmed until the author does it
assert fc.verification_status("Confirmed by author 2026-10-30, Table 2 p.7") == "Confirmed by author"
assert vs["rows_two_readings"] == 5 and vs["rows_read_once"] == 11, vs
assert (meas.loc[meas["status"] == "Two readings agree", "uncertain_fields"] != "").all()  # no cross-checked row hides its caveat
assert (lim["status"] == "Unconfirmed").all()  # BASS-II table was not re-read, so it must not look verified
todo = fc.confirmation_checklist(meas, lim)
assert len(todo) == vs["rows_two_readings"] + vs["rows_unconfirmed"] and not (todo["status"] == "Read once, matches table").any()
assert fc.verification_status("matches NASA table; author to re-check") == "Read once, matches table"
assert fc.verification_status("extracted from fetched report text; author must confirm") == "Unconfirmed"
assert fc.verification_status(float("nan")) == "Unconfirmed"
brief2 = fc.make_brief(["saffire-iv-vi"], cat, facts, meas, lim)
assert "Two readings agree" in brief2 or "Status" in brief2

# real example questions land on the right sourced findings
for q, want in [("Can ground flammability tests be trusted for spacecraft materials?", {"F44"}),
                ("do smoke detectors catch small particles", {"F20", "F21", "F13"}),
                ("what is missing for Moon and Mars", {"F15"}),
                ("does lower pressure slow flame spread", {"F03"})]:
    got = set(fs.search(q, k=4)["id"])
    assert got & want, (q, got)

# gap table
assert fc.evidence_gap_table(cat).values.sum() == len(cat)

# ML: classification beats majority baseline on demo data; leakage flag works
demo = fc.make_demo_data()
res = fc.train_eval(demo, ["o2_fraction", "co2_fraction", "pressure_kpa", "droplet_mm"], "extinction", "classification")
m = res["metrics"]
assert m["random forest accuracy"] > m["majority-class accuracy"] + 0.05, m
assert res["importances"].iloc[0]["feature"] in ("o2_fraction", "co2_fraction"), res["importances"]
reg = fc.train_eval(demo, ["o2_fraction", "co2_fraction", "pressure_kpa", "droplet_mm"], "burn_rate_mm2_s", "regression")
assert reg["metrics"]["random forest R2"] > 0.5, reg["metrics"]
assert fc.leakage_warnings(["o2_fraction", "burn_time_s"], "extinction")  # burn_time flagged
assert not fc.leakage_warnings(["o2_fraction"], "extinction")

# too little data is rejected
try:
    fc.train_eval(demo.head(10), ["o2_fraction"], "extinction", "classification")
    raise SystemExit("expected ValueError")
except ValueError:
    pass

print("all tests passed")
print(m)
print(reg["metrics"])
print(res["importances"])
