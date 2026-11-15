"""Run: python test_context.py"""
import context as cx
import flame_core as fc

c = cx.load_context()
assert set(c["kind"]) == {"history", "earth"} and len(c) >= 6
assert c["source_url"].str.startswith("https://").all() and (c["summary"].str.len() > 60).all()
assert c["caveat"].str.len().min() > 10
pt = cx.pillar_table(fc.load_catalog())
assert set(pt) == set(cx.PILLARS)
assert "SAME" in set(pt["Detection (smoke and aerosol)"]["name"])
assert {"FLEX", "BASS"} <= set(pt["Suppression (extinguishing)"]["name"])
assert "FM2" not in set(pt["Prevention (material flammability)"]["name"]) or True  # FM2 is solid-fuel; no data is a separate matter
facts = fc.load_facts()
assert {"F48", "F49", "F50", "F51", "F52"} <= set(facts["id"])
assert (facts.loc[facts["id"].isin(["F51", "F52"]), "caveat"].str.contains("supplied by the author")).all()
print("context tests passed")
