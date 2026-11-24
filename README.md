# Incentives buy precision in human inferences: analysis code

Analysis code for Salem-Garcia, Lebreton & Wyart, *Incentives buy precision in human inferences*. In two online
experiments (Experiment 1, N = 84; Experiment 2, N = 132, pre-registered on AsPredicted, #287024), participants
performed a Stimulus Reproduction task and a Sequence Categorization task under monetary gain, loss and neutral
incentives. The code cleans the data, fits the noisy-inference models, computes every statistic of the paper and its
supplement, and draws every figure.

## Contents

| Path | Contents |
|---|---|
| `run_pipeline.py` | Runs everything, stage by stage: `fit`, `stats`, `tables`, `figures`, `demo`, `verify` (`--dry-run` lists the steps). |
| `scripts/prereg/`, `src/motinf/prereg/` | The pre-registered pipeline: cleaning, behaviour, single-noise and single-modulation model fits, model statistics. It is the code frozen at pre-registration (git tag `prereg`, archived at https://doi.org/10.5281/zenodo.19490623) with the two deviations of `PREREG_DEVIATIONS.md`. |
| `scripts/exploratory/`, `src/motinf/exploratory/` | Every analysis that was not pre-registered: `fits/` (full model, fast / slow fits, recovery studies), `stats/` (statistics, sample description, `paper_tables.json`), `figures/` (one script per figure of the paper and supplement). |
| `checks/` | The demo, the comparison of outputs with the recorded ones, and the table of `PREREG_DEVIATIONS.md`. |
| `tests/` | Unit tests (`pytest`). |
| `results/`, `plots/exploratory/figures/` | The outputs as reported: statistics (tables) and figures, with each figure's plotted values in `plots/exploratory/figures/source_data/` (one CSV per panel), and the quality checks of the cleaning and the simulation studies. The model fits (`.npz`) come with the data bundle. |
| `PAPER_MAP.md` | Which script and output give each table, figure and number of the paper. |
| `PREREG_DEVIATIONS.md` | The two deviations from the frozen pre-registered code, and the effect of each on every pre-registered number. |
| `data/README.md` | The data bundle: its contents, where it unpacks, how closely a refit reproduces it; the variables of the cleaned trial tables are in `data/data_clean_variables.md`. |

## System requirements

- Tested on Ubuntu 22.04 (under WSL2 on Windows 11), Intel Core i7-13700H (20 threads), 16 GB RAM, with conda.
  Other Linux and macOS systems were not tested; on Windows, use WSL.
- Python 3.11 and the packages pinned in `environment.yml`. Main packages: NumPy, SciPy, pandas, statsmodels,
  pingouin, matplotlib, PyBADS and PyVBMC (model fitting), groupBMC (group Bayesian model selection).
- No special hardware. The statistics and figures take minutes; the model fits and simulation studies take days of
  CPU time (the full-model fits were run on a computing cluster).
- The figures use Arial. Without it they are drawn with Liberation Sans, which has the same character widths, and no
  PDF is written (`scripts/exploratory/figures/README.md`).

## Installation

Download the code (from its Zenodo record, https://doi.org/10.5281/zenodo.19490622, or a release or clone of the
GitHub repository) and, for levels 2 and 3 below, the data bundle (https://doi.org/10.5281/zenodo.24503922;
`data/README.md`). Unzip the code, then the data bundle inside the folder that holds this
README; that folder is named after the source (`NSalem-motivinference-<commit>` from Zenodo, `motivinference-<version>`
from a GitHub release). `unzip` works, or `python -m zipfile -e <zip> <folder>`:

```bash
unzip <code>.zip                          # creates the folder that holds this README
cd <that folder>
unzip <path>/motivinference-data.zip      # optional: the data bundle (data/README.md)
```

Then, in that folder:

```bash
conda env create -f environment.yml
conda activate motinf
```

This installs the package versions used for the paper and the `motinf` package itself (`pip install -e .`), in a few
minutes. Every stage of the pipeline runs in this one environment.

## Demo

```bash
python run_pipeline.py --stage demo
```

Refits the pre-registered single-noise models (sensory, inference and selection noise, each condition separately) to
the first two participants of Experiment 1 with the seed and settings of the full run, and compares the 18 fits with
the bundled ones (`checks/demo_refit.py`). Expected output: a table of 18 fits and `PASS` (log-likelihoods within
0.01 of the bundled ones, parameters within 1%). In the environment of `environment.yml`, with 10 or more CPU cores,
they are identical; with fewer they differ in the last digits and still pass (`data/README.md` says why). Run time:
about 30 seconds (1 minute on one core). The demo needs the data bundle (below).

## Reproducing the paper

Three levels, from a few minutes with nothing to download to refitting every model.

1. **Every test statistic and every figure, from this repository alone.** The participant-level tables of `results/`
   are in the repository; the `tables` stage recomputes from them the tests and group model comparisons and every
   test statistic of the paper and supplement (`paper_tables.json`), and the `figures` stage every figure, with its
   source data (about 2 minutes on the machine above):

   ```bash
   python run_pipeline.py --stage tables figures verify
   ```

   `verify` compares the outputs with the recorded ones. Files identical to the record pass the byte check (on the
   tested machine and environment, all of them); tables and numbers that differ in their bytes must agree within a
   relative tolerance of 1e-6 (`checks/compare_outputs.py`), compared with the versions shipped here, which the first
   run keeps in `build/recorded_outputs/` before overwriting them. Figures that differ are listed, not compared: their
   bytes depend on fonts and library versions. The model fits are recorded too but come with the data bundle; without
   it they are reported, not checked.
2. **The participant-level tables too, from the trial data and the fits, without refitting.** Download the data
   bundle and unpack it at the repository root (`data/README.md`): it holds the de-identified raw and cleaned trial
   data and the model fits, nothing that is already here (the quality checks and the simulation studies are in
   `results/`). Then (about 5 minutes):

   ```bash
   python run_pipeline.py --stage stats tables figures verify
   ```

   The `stats` stage writes the participant-level tables to `results/` (behaviour, bias-noise partition, model
   parameters and predictions, biases, response times, questionnaire regressions, sample description).
3. **The model fits and simulation studies** (days of CPU time), from the trial data of the data bundle:

   ```bash
   python run_pipeline.py --stage fit
   ```

   Seeds are fixed (42) throughout. `PAPER_MAP.md` lists the settings of each fit, and `data/README.md` says how
   closely a refit reproduces the recorded ones.

```bash
pytest tests
```

runs the unit tests (some need the data bundle and are skipped without it; about 20 seconds).

## How the analyses are organised

The paper reports results at three levels, which map onto the code as follows.

- **Confirmatory**: the tests of hypotheses H1–H4 listed in the pre-registration, in Experiment 2 (Table 1). They are
  computed by the pre-registered pipeline (`scripts/prereg/`, `src/motinf/prereg/`).
- **Pre-specified**: analyses run by the same frozen code that are not hypothesis tests (the Stimulus Reproduction
  error SD ANOVA and its one-tailed test, pairwise tests, the group Bayesian model selection among single-noise and
  among single-modulation models). Same code; the Methods say where each is reported.
- **Exploratory**: everything else (pooled sample, effect sizes and Bayes factors, the full model, the bias-noise
  partition, response times, biases, questionnaires): `scripts/exploratory/`, `src/motinf/exploratory/`.

The pre-registered code differs from the frozen code in two places, both described in the Methods, and
`PREREG_DEVIATIONS.md` lists every pre-registered number each one changed. The frozen code is the `prereg` tag of
https://github.com/NSalem/motivinference (archived at https://doi.org/10.5281/zenodo.19490623); with a copy of it in
`<tag>/`, `diff -r <tag>/src/motinf/prereg src/motinf/prereg` (and the same for `scripts/prereg`) shows the two
changes (and an updated `scripts/prereg/README.md`). `checks/prereg_frozen.sha256`
records the hash of every frozen file, and `tests/test_prereg.py` checks against it that nothing else changed, and
that Table 1 is what the pre-registered pipeline writes.

## Glossary

| In the code | In the paper |
|---|---|
| `estim`, `est`, Estimation | Stimulus Reproduction task |
| `categ`, Integration; `inf` in file names (`data_clean_inf.csv`) | Sequence Categorization task |
| `sd_ang`, `sd_inf`, `sd_sel` | sensory, inference and selection noise (σang, σinf, σsel) |
| `motiv_ang`, `motiv_inf`, `motiv_sel` | incentive effects δ = log σ£0 − log σ±£1 |
| `incentive` = −1 / 0 / 1 | loss / neutral / gain (£−1 / £0 / £+1) |
| `seqlen` | sequence length (4, 8 or 12 gratings) |
| `sumllr_noisy` | summed evidence of the displayed sequence (Σ log LR) |
| `optim` | choice consistent with the evidence |
| `noise_per_cond`; models `ang`, `inf`, `sel` | single-noise models, fitted in each condition separately |
| `inc_effects`; models `ang_v`, `inf_v`, `sel_v` | single-modulation models |
| `full_model`, `fixed_ang_from_estim` | full model (σang and δang fixed from the Stimulus Reproduction task) |
| `knockout` | full model without one incentive effect (lesioned model) |
| `variance_partition`, `rho` | split of the variance into bias and noise from repeated trials (ρ) |
| `exp1`, `exp2` | Experiments 1 and 2; the pre-registration calls Experiment 2 `exp1rep` |
| `merged` | both experiments pooled (participant ids prefixed `exp1__` / `exp2__`) |

## Checking a number

Take H3 in Experiment 2 (Table 1: incentives increase evidence sensitivity).

1. `PAPER_MAP.md` gives the source: `paper_tables.json`, key `table1` → `exp2` → `sensitivity` → `abs` (t, p, dz and
   its CI, Bayes factor), computed by `src/motinf/exploratory/paper_tables.py`.
2. The pre-registered pipeline writes the same test to `results/prereg/exp2/behavior/behav_stats.md`
   (`ttest_sensitivity_abs_1_gt_0`); `tests/test_prereg.py` checks that the two agree.
3. To recompute it: `python run_pipeline.py --stage tables verify`, which needs nothing outside this repository.

## Licence

- Code (`src/`, `scripts/`, `checks/`, `tests/`, `run_pipeline.py`): MIT (`LICENSE`).
- Data and results in this repository (`data/`, `results/`, the figures in `plots/`) and the data bundle on Zenodo:
  CC BY 4.0 (`LICENSE-data`).

Versions up to the pre-registration tag `prereg` were released under CC0 1.0 and remain so.

## Citation

Please cite the code or the data you use (also in `CITATION.cff`):

- Code: Salem-Garcia, N. Analysis code for: Incentives buy precision in human inferences, version 1.0.0. Zenodo,
  https://doi.org/10.5281/zenodo.19490622.
- Data: Salem-Garcia, N., Lebreton, M. & Wyart, V. Data for: Incentives buy precision in human inferences. Zenodo,
  https://doi.org/10.5281/zenodo.24503922.
