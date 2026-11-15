"""Build a small, shareable telemetry sample for the app.

Usage:  python make_sample.py "C:/Users/you/Downloads/Saffire_Data_Table2017.161.00.42.03-EU.csv"
Writes data/telemetry_sample.csv and data/telemetry_sample.meta.json.
Check the PSI terms of use before you publish the sample."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import telemetry as tl

MAX_ROWS = 20000

if len(sys.argv) != 2:
    raise SystemExit(__doc__)
src = Path(sys.argv[1])
raw = pd.read_csv(src, low_memory=False)
df, rep = tl.clean_telemetry(raw)
keep = [c for c in tl.CHANNEL_LABELS if c in df.columns]
if not keep:
    raise SystemExit("None of the expected channels were found. Check the file.")
step = max(1, int(np.ceil(len(df) / MAX_ROWS)))
out = df[["RelativeSeconds", "CygnusTimestamp"] + keep].iloc[::step]
Path("data").mkdir(exist_ok=True)
out.to_csv("data/telemetry_sample.csv", index=False)
meta = {"source_file": src.name, "rows_clean": rep["rows_kept"], "rows_sample": int(len(out)),
        "downsample_step": step, "channels": keep}
Path("data/telemetry_sample.meta.json").write_text(json.dumps(meta, indent=2))
print(f"Wrote {len(out):,} rows (every {step}th of {rep['rows_kept']:,}). Channels kept: {len(keep)}")
