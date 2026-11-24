"""Source data of a figure: the plotted values, one CSV per panel.

Each figure script creates a ``SourceData``, adds one or more tables per panel next to the code that draws the panel,
and calls ``write()``. The tables go to ``plots/exploratory/figures/source_data/<figure>/<figure><panel>[_<part>].csv``,
with ``index.csv`` listing panel, file and description. Values are those plotted: one row per participant (and
condition) where a panel shows participants, otherwise the plotted summary or curve. The tests marked by stars are
in ``results/`` and ``paper_tables.json`` (``PAPER_MAP.md``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NEUTRAL = 0.0


class SourceData:
    def __init__(self, figure: str, out_root: str | Path):
        self.figure = figure
        self.dir = Path(out_root) / "figures" / "source_data" / figure
        self.items: list[tuple[str, str, str, pd.DataFrame]] = []

    def add(self, panel: str, table, description: str, part: str | None = None) -> None:
        """One table for ``panel`` (``part`` distinguishes several tables of one panel)."""
        name = f"{self.figure}{panel}" + (f"_{part}" if part else "")
        if any(n == name for _, n, _, d in self.items if d is not None):
            raise ValueError(f"source data {name} added twice")
        df = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
        self.items.append((panel, name, description, df.reset_index(drop=df.index.name is None)))

    def note(self, panel: str, description: str) -> None:
        """A panel without data (a schematic): listed in ``index.csv`` with no file."""
        self.items.append((panel, "", description, None))

    def add_levels(self, panel: str, wide: pd.DataFrame, value: str, description: str, *, levels=None,
                   change: str | None = "diff", part: str | None = None, condition: str = "incentive") -> None:
        """Participants × conditions (``wide``, as drawn by ``dist.levels_panel``) in long form, with the change
        from neutral drawn by ``dist.effects_panel``: ``diff`` (value − neutral), ``logdiff`` (log value − log
        neutral) or ``None``."""
        levels = list(wide.columns) if levels is None else list(levels)
        w = wide[levels].dropna()
        long = w.rename_axis(index="participant", columns=condition).stack().rename(value).reset_index()
        if change is not None and NEUTRAL in levels:
            base = long["participant"].map(w[NEUTRAL])
            v = long[value]
            col = f"{value}_change_from_neutral" if change == "diff" else f"log_{value}_change_from_neutral"
            long[col] = (v - base) if change == "diff" else (np.log(v) - np.log(base))
            long.loc[long[condition] == NEUTRAL, col] = np.nan
        self.add(panel, long, description, part)

    def write(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        for old in self.dir.glob("*.csv"):
            old.unlink()
        rows = []
        for panel, name, description, df in self.items:
            if df is not None:
                df.to_csv(self.dir / f"{name}.csv", index=False, lineterminator="\n")
            rows.append({"panel": panel, "file": f"{name}.csv" if df is not None else "", "description": description})
        pd.DataFrame(rows).to_csv(self.dir / "index.csv", index=False, lineterminator="\n")
        return self.dir
