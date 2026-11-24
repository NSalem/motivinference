# Preregistered Analysis Boundary

This folder documents and enforces the preregistered analysis path.

## Deviations from the frozen code

This folder and `src/motinf/prereg/` hold the code frozen at pre-registration (git tag `prereg`) with two changes,
listed with their effect on every pre-registered number in `PREREG_DEVIATIONS.md` at the repository root:

1. Random starting points for the 10 BADS restarts (`src/motinf/prereg/model.py`); the models were refitted.
2. The exact Greenhouse-Geisser epsilon for every repeated-measures ANOVA (`src/motinf/prereg/behavior.py`,
   `run_model_stats.py`, and the new `src/motinf/prereg/rm_anova_exact.py`): pingouin's tables are kept and gain the
   columns `eps_exact` and `p_GG_exact`, and `results/prereg/<exp>/behavior/rm_anova_exact_gg.csv` lists the exact
   results. These are the p values reported in the paper.

`tests/test_prereg.py` checks that no other file differs from the tag and that Table 1 of the paper is this
pipeline's output.

## Entry Scripts

**Full pipeline (recommended):** from the repository root run

`python scripts/prereg/run_prereg_pipeline.py --exp-name exp1`

It runs, in order: `run_cleaning.py` → `run_behavior_main.py` → `fit_noise_per_cond.py` → `fit_incentive_effects.py` → `run_model_stats.py` (each under `scripts/prereg/`), using `subprocess` and your current interpreter.

Optional flags: `--bads-nit` (default `10`), `--max-nsubs` (default `0` = all participants). These are forwarded to both fit scripts only.

**Individual steps** (if you do not use the pipeline driver):

- `scripts/prereg/run_cleaning.py`
- `scripts/prereg/run_behavior_main.py`
- `scripts/prereg/fit_noise_per_cond.py`
- `scripts/prereg/fit_incentive_effects.py`
- `scripts/prereg/run_model_stats.py`

Core pipeline logic lives in `src/motinf/prereg/`. The fit scripts import `motinf.prereg.model` (and related `motinf.prereg` modules). `run_model_stats.py` reads the saved `.npz` outputs from the fit scripts and uses `pingouin` and `groupBMC` to build CSV summaries and `model_stats_summary.md`.

## Source Namespace

- `src/motinf/prereg/`

This namespace contains prereg-focused modules and API entry functions.
It is self-contained for prereg workflows (no imports from non-prereg
`motinf.*` modules).

## Output Conventions

- Behavioral outputs: `results/prereg/<exp_name>/behavior/` (`behav_stats.md`; exact-GG ANOVAs in `rm_anova_exact_gg.csv`)
- Model fitting outputs: `results/prereg/<exp_name>/model_fit/`
- QC summaries: `results/prereg/<exp_name>/qc/`

## Statistical tests by script

- `run_cleaning.py`
  - No inferential tests; data cleaning and QC summaries.

- `run_behavior_main.py` (via `src/motinf/prereg/behavior.py`)
  - Repeated-measures ANOVA (`pingouin.rm_anova`) on:
    - `sensitivity ~ incentive`
    - `choice_var_log ~ seqlen * incentive`
    - `estim_sd ~ incentive`
    - with, for every effect, the exact Greenhouse-Geisser epsilon and GG p (`rm_anova_exact`, deviation 2).
  - Paired post-hoc tests (`pingouin.pairwise_tests`) with Holm correction.
  - Paired t-tests (`pingouin.ttest`) on absolute-incentive contrasts (`|incentive| = 1` vs `0`) with preregistered directional alternatives where specified.

- `fit_noise_per_cond.py`
  - No inferential tests; participant-level model fitting (BADS) for per-condition models with LL/AICc/BIC outputs.

- `fit_incentive_effects.py`
  - No inferential tests; participant-level model fitting (BADS, with priors) for incentive-channel models with LL/AICc/BIC outputs.

- `run_model_stats.py`
  - Group Bayesian Model Comparison (`GroupBMC`) for:
    - per-condition models (run separately for each incentive),
    - incentive-effects models.
  - Repeated-measures ANOVA (`pingouin.rm_anova`) on `log(sd_inf) ~ incentive`, with the exact GG epsilon and p
    (deviation 2).
  - Paired one-sided t-test (`pingouin.ttest`) on `log(sd_inf)` for pooled `|incentive| = 1` vs `0` (`alternative='less'`).

