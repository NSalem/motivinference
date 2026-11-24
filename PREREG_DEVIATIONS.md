# Deviations from the pre-registered analysis code

The analysis code for Experiment 2 was frozen before its data were collected: git tag `prereg` (commit fb8c144,
10 April 2026), archived at https://doi.org/10.5281/zenodo.19490623. The pre-registered pipeline in this repository
(`src/motinf/prereg/`, `scripts/prereg/`) is that code plus the two changes below, both reported in the paper's
Methods ("Pre-registered analyses"). Apart from `scripts/prereg/README.md`, no other file in those folders differs
from the tag: `tests/test_prereg.py` checks every file against the hashes in `checks/prereg_frozen.sha256`. In the
public repository, which is built on top of the tag, `git diff prereg -- src/motinf/prereg scripts/prereg` shows the
whole change.

| | Files |
|---|---|
| Deviation 1 | `src/motinf/prereg/model.py` |
| Deviation 2 | `src/motinf/prereg/behavior.py`, `scripts/prereg/run_model_stats.py`, `src/motinf/prereg/rm_anova_exact.py` (new) |
| Documentation only | `scripts/prereg/README.md` |

## Deviation 1: random starting points for the optimizer

**Change.** `fit_model_BADS` started every BADS restart from the model's `x0` whenever the model defined one, which
every pre-registered model does, so the 10 restarts repeated the same optimization. With more than one restart, each
now starts from a point drawn uniformly within the plausible bounds (Table 4 of the paper), as in the full-model fits;
`x0` is used only for a single run. A restart that raises an exception is dropped, and the best of the others is kept.

**Why.** A bug: the code was meant to run 10 independent optimizations and keep the best.

**Effect.** Both experiments were refitted in November 2026 (single-noise models per incentive condition and
single-modulation models; seed 42, 10 restarts; in the environment of `environment.yml`). In the single-noise refit, 12
of the 19,440 restarts crashed inside PyBADS (a degenerate initial Gaussian process: `AttributeError: 'float' object has
no attribute 'item'`) and were dropped; each of these fits kept the best of its other 9 restarts:

- Experiment 1: S061 (neutral, inference-noise model), S068 (neutral, selection), S078 (gain, selection), S086
  (neutral, inference);
- Experiment 2: S059 (neutral, inference), S063 (neutral, selection), S074 (gain, selection), S080 (neutral,
  inference), S115 (neutral, sensory), S117 (neutral, sensory), S143 (gain, sensory), S147 (gain, selection).

No restart crashed in the single-modulation refit. Every analysis in the paper, pre-registered or not, uses the
refitted fits. The fits as registered are kept in `results/prereg/<exp>/model_fit/as_registered/`. The table below
gives every pre-registered number they change; no conclusion changes.

## Deviation 2: exact Greenhouse–Geisser correction for every ANOVA

**Change.** The frozen code reports pingouin's `rm_anova`. In one-way designs, pingouin applies the Greenhouse–Geisser
correction only when Mauchly's test rejects sphericity; in the two-way design (sequence length × incentive), its ε for
the interaction is inexact when both factors have more than two levels (pingouin warns about this). The paper reports,
for every repeated-measures ANOVA, the GG-corrected *P* with the exact ε of each effect, computed by
`src/motinf/prereg/rm_anova_exact.py` (tested against pingouin and against an independent construction in
`tests/test_rm_anova_exact.py`). `behavior.py` and `run_model_stats.py` keep pingouin's tables unchanged and add two
columns, `eps_exact` and `p_GG_exact`; `results/prereg/<exp>/behavior/rm_anova_exact_gg.csv` lists the exact results
of the three behavioural ANOVAs.

**Why.** One rule for all ANOVAs, and a correct ε for the interaction.

**Effect.** Four *P* values change at the reported precision, none of them across 0.05: the H2 interaction and the
reproduction error SD ANOVA, in both experiments (the rows of the table where only the last column is bold).

## Effect on every pre-registered number

Columns: the frozen code on the fits as registered (with the frozen *P*-value rule); after deviation 1 (refitted fits,
same rule); after both deviations, as reported in the paper (Table 1 and Methods). Bold: differs from the first column
at the reported precision. All three columns are computed by the pre-registered code's own functions in the
environment of `environment.yml` (pingouin 0.6.1).

<!-- table:start (checks/prereg_deviations.py) -->
| Test | Frozen code, fits as registered | Deviation 1: random starts | Deviations 1 + 2 (reported) |
|---|---|---|---|
| **Experiment 1 (discovery)** | | | |
| H1: evidence sensitivity, incentive | F(2, 166) = 7.59, *P* = 0.001 | F(2, 166) = 7.59, *P* = 0.001 | F(2, 166) = 7.59, *P* = 0.001 |
| H2 ANOVA: log choice variability, sequence length | F(2, 166) = 185.06, *P* < 0.001 | F(2, 166) = 185.06, *P* < 0.001 | F(2, 166) = 185.06, *P* < 0.001 |
| H2 ANOVA: log choice variability, incentive | F(2, 166) = 9.21, *P* = 0.002 | F(2, 166) = 9.21, *P* = 0.002 | F(2, 166) = 9.21, *P* = 0.002 |
| H2: log choice variability, length × incentive | F(4, 332) = 1.24, *P* = 0.290 | F(4, 332) = 1.24, *P* = 0.290 | **F(4, 332) = 1.24, *P* = 0.296** |
| H3: evidence sensitivity, incentive vs neutral (one-tailed) | t(83) = 3.29, *P* < 0.001 | t(83) = 3.29, *P* < 0.001 | t(83) = 3.29, *P* < 0.001 |
| H4A: log σinf, incentive | F(2, 166) = 9.54, *P* = 0.001 | **F(2, 166) = 9.55, *P* = 0.001** | **F(2, 166) = 9.55, *P* = 0.001** |
| H4B: log σinf, incentive vs neutral (one-tailed) | t(83) = −3.29, *P* < 0.001 | t(83) = −3.29, *P* < 0.001 | t(83) = −3.29, *P* < 0.001 |
| Reproduction error SD, incentive (pre-specified) | F(2, 166) = 2.72, *P* = 0.069 | F(2, 166) = 2.72, *P* = 0.069 | **F(2, 166) = 2.72, *P* = 0.073** |
| Reproduction error SD, incentive vs neutral (one-tailed, pre-specified) | t(83) = −2.09, *P* = 0.020 | t(83) = −2.09, *P* = 0.020 | t(83) = −2.09, *P* = 0.020 |
| Single-noise BMC: inference-model frequency, loss / neutral / gain (pre-specified) | 96% / 92% / 99%; PXP > 0.99 | 96% / 92% / 99%; PXP > 0.99 | 96% / 92% / 99%; PXP > 0.99 |
| Single-modulation BMC: inference-modulation frequency (pre-specified) | 82%; PXP > 0.99 | **79%; PXP > 0.99** | **79%; PXP > 0.99** |
| **Experiment 2 (pre-registered)** | | | |
| H1: evidence sensitivity, incentive | F(2, 262) = 9.07, *P* < 0.001 | F(2, 262) = 9.07, *P* < 0.001 | F(2, 262) = 9.07, *P* < 0.001 |
| H2 ANOVA: log choice variability, sequence length | F(2, 262) = 242.11, *P* < 0.001 | F(2, 262) = 242.11, *P* < 0.001 | F(2, 262) = 242.11, *P* < 0.001 |
| H2 ANOVA: log choice variability, incentive | F(2, 262) = 6.53, *P* = 0.006 | F(2, 262) = 6.53, *P* = 0.006 | F(2, 262) = 6.53, *P* = 0.006 |
| H2: log choice variability, length × incentive | F(4, 524) = 0.51, *P* = 0.621 | F(4, 524) = 0.51, *P* = 0.621 | **F(4, 524) = 0.51, *P* = 0.689** |
| H3: evidence sensitivity, incentive vs neutral (one-tailed) | t(131) = 4.18, *P* < 0.001 | t(131) = 4.18, *P* < 0.001 | t(131) = 4.18, *P* < 0.001 |
| H4A: log σinf, incentive | F(2, 262) = 7.47, *P* = 0.002 | **F(2, 262) = 7.48, *P* = 0.002** | **F(2, 262) = 7.48, *P* = 0.002** |
| H4B: log σinf, incentive vs neutral (one-tailed) | t(131) = −3.20, *P* < 0.001 | t(131) = −3.20, *P* < 0.001 | t(131) = −3.20, *P* < 0.001 |
| Reproduction error SD, incentive (pre-specified) | F(2, 262) = 0.48, *P* = 0.619 | F(2, 262) = 0.48, *P* = 0.619 | **F(2, 262) = 0.48, *P* = 0.618** |
| Reproduction error SD, incentive vs neutral (one-tailed, pre-specified) | t(131) = −0.72, *P* = 0.236 | t(131) = −0.72, *P* = 0.236 | t(131) = −0.72, *P* = 0.236 |
| Single-noise BMC: inference-model frequency, loss / neutral / gain (pre-specified) | 90% / 93% / 93%; PXP > 0.99 | 90% / 93% / 93%; PXP > 0.99 | 90% / 93% / 93%; PXP > 0.99 |
| Single-modulation BMC: inference-modulation frequency (pre-specified) | 99%; PXP > 0.99 | 99%; PXP > 0.99 | 99%; PXP > 0.99 |
<!-- table:end -->

Regenerate the table with `python checks/prereg_deviations.py` (data bundle needed);
`python checks/prereg_deviations.py --check`, also run by `tests/test_prereg.py`, fails when it is out of date. The
values behind it are in `checks/prereg_deviations.csv`.
