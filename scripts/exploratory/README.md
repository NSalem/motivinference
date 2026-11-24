# Analyses that were not pre-registered

- `fits/`: model fits and simulation studies (full model, fast/slow RT-split fits, recovery and identifiability);
  the `fit` stage.
- `stats/`: the participant-level tables, from the trial data and the fits (`stats` stage, the variance partition
  included), and the statistics computed from those tables alone (`tables` stage: `run_table_stats.py`, then
  `run_paper_tables.py`, which collects every reported test into
  `results/exploratory/merged/paper_tables/paper_tables.json`).
- `figures/`: the paper's figures, each drawn whole at its printed size (`fig1.py` … `fig5.py`, `sfig1.py` …
  `sfig7.py`; house style and conventions in `figures/README.md`). The data preparation and statistics they draw
  are in `src/motinf/exploratory/figure_data.py`.

`run_pipeline.py` at the repository root runs all of the above in order (`--stage fit stats tables figures demo
verify`, `--dry-run`).

Scripts read cleaned data from `data/processed/<exp>/` and pre-registered outputs from `results/prereg/<exp>/`,
and write to `results/exploratory/<exp>/...` and `plots/exploratory/figures/`. Run them from the repository
root; each docstring lists its inputs, outputs and the paper item it produces. See `PAPER_MAP.md` at the
repository root for the full map.
