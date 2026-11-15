"""Safety context: history lessons, Earth spin-offs, and which experiments inform prevention, detection and suppression.

History and spin-off rows are short paraphrases of NASA pages (data/context.csv), each with its link.
The pillar table is computed from the catalog criteria (author's checks against the sources read)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# Transparent rule: which catalog criterion places an experiment under which safety pillar.
PILLARS = {
    "Prevention (material flammability)": "solid_materials",
    "Detection (smoke and aerosol)": "smoke_detection",
    "Suppression (extinguishing)": "suppression_extinction",
}


def load_context(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "context.csv")


def pillar_table(catalog: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Experiments with a criterion score above zero for each pillar, strongest first."""
    out = {}
    for pillar, col in PILLARS.items():
        d = catalog[catalog[col] > 0].sort_values([col, "name"], ascending=[False, True])
        out[pillar] = d[["name", "platform", "topic", col, "source_read"]].rename(columns={col: "criterion score (1 = yes, 0.5 = partly)"})
    return out
