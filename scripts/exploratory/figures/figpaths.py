"""Paths for the whole-figure scripts in this folder (imported first by each of them).

DATA_ROOT is the repository root: inputs are read from ``results/`` and ``data/processed/``.
OUT_ROOT is ``plots/exploratory``; each script writes ``OUT_ROOT/figures/<name>.{svg,pdf,png}``. Set
``MOTINF_FIG_OUT`` to write elsewhere (for example to compare a render with the committed figures).

Data preparation and the statistics drawn in the figures come from ``motinf.exploratory.figure_data`` and the other
analysis modules (the package installed by ``environment.yml``).
"""
import os
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = Path(os.environ.get("MOTINF_FIG_OUT", DATA_ROOT / "plots" / "exploratory")).resolve()
