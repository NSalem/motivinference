# Data

The trial data and the model fits are not stored in this repository. Download the data bundle, a separate Zenodo
record under CC BY 4.0 (https://doi.org/10.5281/zenodo.24503922), and unzip it at the repository root, the folder
that holds the main `README.md` (`unzip <path>/motivinference-data.zip` there). It holds only files that this
repository does not, so unpacking it overwrites nothing; it adds:

| Path | Contents |
|---|---|
| `data/raw/<exp>/trials/S###/` | Task output of each session (one CSV per session, PsychoPy). Participants are identified by `S###` codes; Prolific participant, session and study identifiers were removed. |
| `data/raw/<exp>/questionnaire/questionnaire_data.csv` | Questionnaire responses and scores (BIS/BAS; SPSRQ-RC in Experiment 2), age, sex, handedness. Identifiers, browser data and free-text answers were removed. |
| `data/interim/<exp>/all_trials_{inf,est}.csv` | All trials before cleaning (Sequence Categorization / Stimulus Reproduction task). |
| `data/processed/<exp>/data_clean_{inf,est}.csv` | Trials after cleaning; variables are described in `data_clean_variables.md`. |
| `results/prereg/<exp>/model_fit/*.npz` | Pre-registered fits: single-noise models per condition (`fit_noise_per_cond.npz`) and single-modulation models (`fit_inc_effects.npz`). `as_registered/`: the same fits as the frozen code produced them, before deviation 1 (`PREREG_DEVIATIONS.md`). |
| `results/exploratory/<exp>/modeling/fit_files/*.npz` | Full model (`fit_full_model.npz`) and inference-only model fitted to fast and slow trials (`fit_inf_rt_split.npz`). |

The other outputs of the cleaning and of the fits are in this repository:

| Path | Contents |
|---|---|
| `results/prereg/<exp>/qc/` | Quality checks of the cleaning: criteria per participant, exclusion summary. |
| `results/exploratory/exp2/modeling/full_model_fit/` | Record of the Experiment 2 full-model run (settings, one row per participant). |
| `results/exploratory/exp1/modeling/{model_identifiability,model_motiv_identifiability,recovery_noise_per_cond}/` | Model recovery for the single-noise and single-modulation models, parameter recovery for the inference-only model (Supp. Figs 2 and 4b). |
| `results/exploratory/merged/modeling/param_recovery_full_model/` | Parameter recovery for the full model (Supp. Fig. 5). |
| `data/participant_flow_prolific.csv` | Totals from the Prolific submission records: participants who did not complete session 1 or failed its attention check, completed session 1, did not complete session 2, completed both. The records themselves hold Prolific IDs and are not distributed; this table was compiled from them by hand. |

`exp1` is Experiment 1 (N = 84 after exclusions) and `exp2` Experiment 2, the pre-registered experiment (N = 132;
`exp1rep` in the pre-registration). `DATA_BUNDLE_MANIFEST.sha256` in the bundle lists a checksum for every file
(`sha256sum -c DATA_BUNDLE_MANIFEST.sha256` at the repository root, after unpacking), and `DATA_BUNDLE_LICENSE.txt`
holds its licence (CC BY 4.0, as `LICENSE-data`).

With the bundle in place, `python run_pipeline.py --stage stats tables figures verify` recomputes and checks every
statistic and figure; without it, `--stage tables figures verify` recomputes them from the participant-level tables
of `results/`. `data/interim/`, `data/processed/` and `results/prereg/<exp>/qc/` are re-created from `data/raw/` by the
frozen cleaning (`scripts/prereg/run_cleaning.py`) in the environment of `environment.yml`, byte for byte; the
fits by the `fit` stage of `run_pipeline.py`.

## Refitting

The `fit` stage of `run_pipeline.py` regenerates every fit of the bundle except `as_registered/`, which the frozen
code made, and the simulation studies above (seed 42; the settings of each are in `PAPER_MAP.md`). On Linux, in the
environment of `environment.yml`, it reproduces the recorded files bit for bit. The pre-registered fits need at least 10 CPUs for that:
the frozen code runs the 10 restarts of a fit in as many processes as there are CPUs, up to 10, and restarts that share
a process draw from one random stream, so with fewer CPUs the fits differ slightly (about as much as with another
seed). The other fits give the same result on any number of CPUs. Another build of NumPy and SciPy changes the fits
slightly too: the log-likelihoods of the pre-registered single-noise fits by up to 2e-4.
