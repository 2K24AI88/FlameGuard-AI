# FlameGuard AI — Flame in Freefall (NASA Space Apps 2026)

Dashboard that **summarizes**, **ranks** and **interprets** NASA microgravity combustion research.
Every claim carries a citation, missing values stay blank, evidence gaps are shown, and every measured number
carries a visible verification status.

## Run
```
pip install -r requirements.txt
python test_core.py
python test_telemetry.py
python test_evidence.py
python test_gaps.py
python test_context.py
streamlit run app.py          # not a test file: those have no screen
```
Run all commands from the folder that contains `app.py`.

## Pages
- **Overview**: the pitch, verification status, and a worked example question.
- **Research Explorer**: Catalog · Rank · **Ask & Evidence** (cited findings, evidence strength, caveats, gaps) · **Compare** · Insights · Brief (downloadable, cited).
- **Evidence Lab**: Test table · Burn time · Heat & gases · Flammability limits · **BASS-II matrix** (129 tests, with a worked example of reading CO correctly) · **Research gaps** (recorded tests by gravity, oxygen and pressure; click a cell for related tests and a templated question) · Scenario check (now includes Saffire rows with pressure) · **Verification** (checklist of rows that still need a human check).
- **Safety Context**: lessons from Apollo 1, Apollo 13 and Mir; which investigations inform prevention, detection and suppression; Earth spin-offs (smoke detector, firefighter air tank and clothing). Each item cites its NASA page.
- **Telemetry Lab** (exploratory): data quality, signals, ignition → response lag, replay, compare.
- **Model Lab**: inspects every column, masks 'no data' codes, warns about clock or motion-sensor targets, and tests a model on held-out rows (chronological split for time series) against baselines. It says plainly when the model does not beat the baseline.
- **Methods & Limits**.

## What the "AI" is (and is not)
Ask & Evidence uses TF-IDF text retrieval over sourced findings, then fixed, visible rules for evidence strength,
caveats and gaps. It writes no new claims. The only learned model is the optional Model Lab (random forest with baselines).

## Data files (`data/`)
| File | What it holds | Status |
|---|---|---|
| `experiments.csv` | 20 experiments, yes/no criteria with written basis | 17 read in detail, 3 overview only (ACME experiments NASA's booklet does not describe separately) |
| `facts.csv` | 52 paraphrased findings with source, location, caveat | Read 2026-10-25/26 by automated reading of NASA pages and PDFs |
| `key_findings.csv` | 8 curated answers citing fact ids (I8: suppressants in microgravity, from the author-supplied FLEX test plan) | Built from facts |
| `context.csv` | 3 history lessons and 3 Earth spin-offs, each with a NASA link | Short paraphrases of NASA pages read 2026-11-14 |
| `measurements.csv` | 16 test rows: Saffire-I, II, IV, V, VI (blank = not stated) | I/II read once and match NASA's table; IV-VI two automated readings agree, heat-total mapping uncertain |
| `bass2_matrix.csv` | 129 BASS-II tests: conditions, initial and final oxygen, CO2 and CO | Converted from the author-supplied workbook (entries reading 'none' left blank) `BASS-II_Test_Matrix_by_PI_Final.xlsx`; PSI origin (PSI-25) stated by the author, not confirmed |
| `measured_results.csv` | Flammability limits (mostly BASS-II) | **Unconfirmed**: the BASS-II report could not be re-read |
| `insights.csv` | Four short insights used by `evidence.py` | Source links author-provided |
| `telemetry_sample.csv` | Downsampled telemetry (you build it) | See below |

## Telemetry data
The bundled sample `data/telemetry_sample.csv` was built from your file (all 11,449 clean rows, 18 channels). To rebuild it, or to use another file, upload a CSV in the Telemetry Lab or run:
```
python make_sample.py "C:/Users/you/Downloads/Saffire_Data_Table2017.161.00.42.03-EU.csv"
```
Check the PSI terms before publishing the sample; if they do not allow it, delete `data/telemetry_sample.csv` and `data/telemetry_sample.meta.json` and the app will ask for an upload. Commit `data/` so judges can run the app.
The file name does not say which experiment it is. Its own clock starts 5 June 2017, 19:35 UTC (the `161` in the name is a file stamp, 10 June), which fits Saffire-III (NASA planned ignition for early June 2017) and rules out Saffire-I (2016). Confirm on the PSI page that it is SAFFIRE-III.

## Before you submit
- [ ] Open the **Evidence Lab → Verification** tab, download the checklist, and check each row against the PDF. When a row is right, change its `verification` text to start with `Confirmed by author` (plus date and page). The status updates everywhere.
- [ ] Read the BASS-II report (NASA/TM-20210011385) and confirm the rows in `measured_results.csv`, or remove them.
- [ ] Telemetry time unit: the app now infers milliseconds from the file's own clock (recording 5 June 2017, 19:35 to 21:30 UTC). Confirm on PSI that the file is SAFFIRE-III.
- [ ] Verify PSI IDs (PSI-98 and PSI-102 pages exist; PSI-25, 39, 69 unverified).
- [ ] Confirm data and software licensing terms, then run `make_sample.py`, commit `data/`, deploy (Streamlit Community Cloud), and add the link to your submission.
- [ ] Check the exact submission deadline on the official Space Apps site (the event is Nov 14-15, 2026).
- [ ] Record a 2-3 minute demo: problem, Ask & Evidence on one example question, Compare, Evidence Lab verification, telemetry (labeled exploratory), limits.

## Honest limits
Criteria are checks against the sources read, not NASA measurements. Saffire-III has no published results in the sources read.
SAFFIRE-I and SAFFIRE-III are different experiments. Synthetic Model Lab data is not NASA data. Not for flight or safety decisions.
