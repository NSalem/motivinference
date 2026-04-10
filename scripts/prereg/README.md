# Preregistered Analysis Boundary

This folder documents and enforces the preregistered analysis path.

## Entry Scripts

**Full pipeline (recommended):** from the repository root run

`python scripts/prereg/run_prereg_pipeline.py --exp-name exp1`

It runs, in order: `run_cleaning.py` → `run_behavior_main.py` → `fit_noise_per_cond.py` → `fit_incentive_effects.py` → `run_model_stats.py` (each under `scripts/prereg/`), using `subprocess` and your current interpreter.

Optional flags: `--bads-nit` (default `10`), `--max-nsubs` (default `0` = all participants). These are forwarded to both fit scripts only.

Optional: `scripts/prereg/run_prereg_pipeline.sh` runs the same five steps from bash (same flags; does not invoke `run_prereg_pipeline.py`).

**Individual steps** (if you do not use the pipeline driver):

- `scripts/prereg/run_cleaning.py`
- `scripts/prereg/run_behavior_main.py`
- `scripts/prereg/fit_noise_per_cond.py`
- `scripts/prereg/fit_incentive_effects.py`
- `scripts/prereg/run_model_stats.py`

Core pipeline logic lives in `src/motinf/prereg/`. The fit scripts import `motinf.prereg.model` (and related `motinf.prereg` modules). `run_model_stats.py` reads the saved `.npz` outputs from the fit scripts and uses `pingouin` plus optional `groupBMC` to build CSV summaries and `model_stats_summary.md`.

## Source Namespace

- `src/motinf/prereg/`

This namespace contains prereg-focused modules and API entry functions.
It is self-contained for prereg workflows (no imports from non-prereg
`motinf.`* modules).

## Output Conventions

- Behavioral outputs: `results/prereg/<exp_name>/behavior/`
- Model fitting outputs: `results/prereg/<exp_name>/model_fit/`
- QC summaries: `results/prereg/<exp_name>/qc/`

