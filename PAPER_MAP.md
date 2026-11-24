# Paper → code map

Where each table, figure and number of the paper and its supplement comes from. Paths are relative to the repository
root. `<exp>` is `exp1` or `exp2`; `merged` = both experiments pooled (participant ids prefixed `exp1__` / `exp2__`).
Every test statistic and figure is recomputed from the participant-level tables in `results/`, with nothing to
download, by

```bash
python run_pipeline.py --stage tables figures verify
```

and the participant-level tables themselves from the trial data and the fits (data bundle, `data/README.md`) by
`--stage stats`. Outputs of the `tables` stage are written by `scripts/exploratory/stats/run_table_stats.py`
(`src/motinf/exploratory/table_stats.py`) and `run_paper_tables.py`; every other output under `results/` by the
`stats` stage. The pre-registered analyses are computed by the pre-registered code (`scripts/prereg/`,
`src/motinf/prereg/`); everything else by `scripts/exploratory/` and `src/motinf/exploratory/`.

Abbreviations: `PT` = `results/exploratory/merged/paper_tables/paper_tables.json`, written by
`scripts/exploratory/stats/run_paper_tables.py` (functions in `src/motinf/exploratory/paper_tables.py`, which holds
the conventions: contrasts, dz with exact CI, Holm correction, Bayes factors, exact Greenhouse-Geisser correction).
`MS` = `results/exploratory/merged/modeling/stats`; `BX` = `results/exploratory/merged/behavior/stats/behav_extra/params`
(its tables have one row per experiment where the paper reports the experiments separately); `SD` =
`results/exploratory/merged/sample_description`.

## Tables

In `PT`, `<sample>` is `exp1`, `exp2` or `merged`; each measure holds the omnibus ANOVA (`omnibus`), the contrasts
(`gain_neutral`, `loss_neutral`, `gain_loss`, `pooled`), and descriptives (`desc`).

| Item | Source (`PT` keys) | Computed by |
|---|---|---|
| Table 1 (pre-registered tests H1–H4) | `table1` → `<exp>` → `sensitivity` (H1: `omnibus`; H3: `abs`), `choice_var` (H2: `seqlen`, `incentive`, `seqlen_x_incentive`), `log_sd_inf` (H4: `omnibus`, `abs`), `estim_sd` | `omnibus`, `seq_omnibus`, `abs_effect`. The pre-registered pipeline writes the same tests: `results/prereg/<exp>/behavior/{rm_anova_exact_gg.csv,behav_stats.md}`, `results/prereg/<exp>/model_fit/stats/{anova_sd_inf_incentive.csv,ttest_log_sd_inf_abs_1_vs_0.csv}`; `tests/test_prereg.py` checks that they agree. Effect of the two deviations on each number: `PREREG_DEVIATIONS.md` |
| Table 2 (sensitivity, choices consistent with the evidence, inference noise, reproduction error SD) | `table2` → `<sample>` → `sensitivity`, `categ_optim`, `log_sd_inf`, `estim_sd` | `measure` |
| Table 3 (bias and noise from repeated trials) | `table2` → `<sample>` → `categ_noise_log`, `categ_bias`, `estim_noise_log`, `estim_bias`; omnibus tests of the correlation between repetitions and of consistency: `supp` → `partition` | `measure`; partitions from `scripts/exploratory/stats/run_variance_partition.py` |
| Table 4 (model parameters: bounds, priors) | pre-registered fits: `param_config_categ` in `src/motinf/prereg/model.py`; full model: `param_config_categ`, `priors_categ` in `src/motinf/exploratory/model.py` and `src/motinf/exploratory/full_model_fitting.py` | |
| Supp. Table 1 (choice variability on the natural scale; reproduction error SD on the log scale) | `supp` → `choice_var_natural`, `estim_sd_log` | `choice_var_natural`, `omnibus` |
| Supp. Table 2 (each experiment, and Experiment 1 vs 2) | `table2` → `exp1`, `exp2`; `supp` → `heterogeneity` | `measure`, `experiment_difference` |
| Supp. Tables 3 and 5 (questionnaire and demographic predictors of the inference-only and full-model parameters) | `supp` → `questionnaire` | `questionnaire_rlm`, from `scripts/exploratory/stats/run_questionnaire_model_correlations.py` |
| Supp. Table 4 (full-model parameters predicting the choice-variance midpoint and slope) | `supp` → `choice_var_prediction` | `choice_var_prediction_rlm`, from the regressions of `table_stats.param_behavior_rlm` |
| Supp. Table 6 (common biases) | `supp` → `biases` | `bias_measures`, from `scripts/exploratory/stats/run_bias_stats.py` |
| Supp. Table 7 (median response time) | `table2` → `<sample>` → `categ_rt`, `estim_rt` | `measure` |
| Supp. Table 8 (response time against evidence) | `supp` → `rt_evidence` | `rt_evidence`, from `scripts/exploratory/stats/run_response_times.py` |
| Supp. Table 9 (noise in fast and slow trials) | `supp` → `fast_slow`, `fast_slow_by_incentive` | `build_all`, from `run_response_times.py` and the fast / slow fits |
| Supp. Table 10 (incentive effects on response time vs on noise) | `supp` → `rt_corr` | `build_all` |

## Figures

Each figure is drawn whole, at its printed size, by one script in `scripts/exploratory/figures/`, schematics included;
it writes `plots/exploratory/figures/<name>.{svg,pdf,png}` (the PDF only when Arial is installed) and the plotted
values, one CSV per panel, to `plots/exploratory/figures/source_data/<name>/` (described in its `index.csv`). The
docstring of each script lists its panels and inputs; house style: `scripts/exploratory/figures/README.md`.

| Figure | Script | Inputs |
|---|---|---|
| 1 | `fig1.py` | a–d: drawn; e, g: `results/prereg/<exp>/behavior/data_mean_inc.csv`; f: `BX/psychometric_consistency_{bins,curves}.csv` (the curves average a probit fit per participant and condition, `behavior_stats.psychometric_consistency`, run by `run_behavior_stats.py`); stars of e, g: `PT` → `table2` → `merged` |
| 2 | `fig2.py` | a: drawn (simulated); b: `MS/model_fit_stats/params/groupbmc_noise_per_incentive.csv`; c: `results/prereg/<exp>/model_fit/stats/sd_inf_long.csv`; d: `results/prereg/<exp>/behavior/data_mean_inc_seqlen.csv`, `BX/model_pred_per_seqlen_simple_inf.csv`; stars of c: `PT` → `table2` → `merged` |
| 3 | `fig3.py` | `MS/full_model_stats/params/{params_wide,ttests_motiv_vs0,variance_partition_params,r2_var_red_natural_abs_obs_vs_knockout_model}.csv`, `MS/full_model_stats/knockout/{knockout_predictions,knockout_var_red_natural_by_seqlen_long}.csv` |
| 4 | `fig4.py` | a: drawn (simulated, ρ set to the observed means at £0); b, c: `results/exploratory/<exp>/variance_partition/{variance_partition_estim,var_partition_categ}.csv`; stars: `PT` → `table2` → `merged` (Table 3) |
| 5 | `fig5.py` | `results/exploratory/merged/behavior/stats/rt_analysis/*.csv`; stars of a: `PT` → `table2` → `merged` (Supp. Table 7) |
| Supp. 1 | `sfig1.py` | `PT` → `table2` |
| Supp. 2 | `sfig2.py` | a: `results/exploratory/exp1/modeling/model_identifiability/confusion_matrix_percent.csv`; b: `results/exploratory/exp1/modeling/recovery_noise_per_cond/recovery_runs.csv` |
| Supp. 3 | `sfig3.py` | `MS/questionnaire_model_correlations/simple_model/core/{,demographics/}rlm/rlm_coefs_long.csv` |
| Supp. 4 | `sfig4.py` | a, c: `MS/model_fit_stats/params/{groupbmc_incentive_effects,r2_var_red_natural_abs_obs_vs_inc_eff_model}.csv`; b: `results/exploratory/exp1/modeling/model_motiv_identifiability/confusion_matrix_percent.csv`; d: `MS/model_fit_stats/params/var_red_by_seqlen_single_modulation.csv` (predictions per participant in `pred_choice_var_single_modulation.csv`, both from `run_inference_model_stats.py`) |
| Supp. 5 | `sfig5.py` | `results/exploratory/merged/modeling/param_recovery_full_model/parameter_recovery_runs.csv` |
| Supp. 6 | `sfig6.py` | `MS/full_model_stats/correlations/{param_behavior_main,param_behavior_diff,param_behavior_correlations,rlm_coefs}.csv` (r, P and model R² of a and c: `param_behavior_correlations`, tables stage) |
| Supp. 7 | `sfig7.py` | `MS/bias_stats/params/` |

## Numbers quoted in the text

Test statistics (F, t, p, dz with CI, Bayes factors, regression coefficients) are in `PT` (tables above). The other
numbers:

| Number | Source |
|---|---|
| **Methods** | |
| Participants who did not complete session 1 (13, 21) or failed the attention check (0, 1), completed both sessions (102, 156), did not complete session 2 (4, 1) | `SD/participant_flow.csv` (from the Prolific records: `data/participant_flow_prolific.csv`) |
| Participants with the data of both sessions (100, 148), excluded by the quality checks (16, 16; 16%, 11%) and per criterion; final samples (84, 132) | `SD/participant_flow.csv`, `SD/exclusion_criteria.csv`, `SD/sample_description.txt`; excluded ids in `results/prereg/<exp>/qc/exclusion_summary.txt` |
| Sex and age of the final samples | `SD/demographics.csv` (`scripts/exploratory/stats/run_sample_description.py`) |
| Trials removed by the response-time cut among retained participants | `BX/rt_trial_exclusions.csv` |
| Participants excluded from the choice-variability analyses (18, 35; N = 163) | `n_participants` of `results/exploratory/<exp>/behavior/stats/behav_extra/params/rm_anova_choice_var_natural_seqlen_x_incentive.csv` and of `BX/rm_anova_choice_var_natural_seqlen_x_incentive.csv` |
| Power analysis: effect of Experiment 1 (0.218 standardized by the average SD; dz = 0.36) | `PT` → `text` → `power_d_av_exp1` (the pre-registered output: `results/prereg/exp1/behavior/behav_stats.md`, `ttest_sensitivity_abs_1_gt_0`, `cohen_d`); `PT` → `table1` → `exp1` → `sensitivity` → `abs` → `dz` |
| **Results** | |
| Reproduction error SD, both experiments (8.34 ± 0.25°) | `PT` → `text` → `estim_sd_pooled` |
| Accuracy (78%, 75%), ideal observer (89%), choices consistent with the evidence (79.3%, 77.1%) | `BX/accuracy_vs_ideal_observer.csv` |
| Single-noise models: estimated frequency of the inference model and PXP | `PT` → `bms` → `<sample>` (`noise_freq_min`, `noise_freq_max`, `noise_pxp_min`); per condition: `results/prereg/<exp>/model_fit/stats/groupbmc_noise_per_incentive.csv`, `MS/model_fit_stats/params/groupbmc_noise_per_incentive.csv` |
| R² of the inference-only model to choice variability (mean, median) | `BX/r2_choice_var_full_grid_natural_obs_vs_simple_inf.csv` |
| Correlation of the incentive effects between tasks | `MS/model_fit_stats/correlations/corr_between_tasks.csv` |
| Single-modulation models: frequency of the inference-modulation model and PXP; error of the predicted incentive effect | `PT` → `bms` → `<sample>` (`mod_freq`, `mod_pxp`); `MS/model_fit_stats/params/r2_var_red_natural_abs_obs_vs_inc_eff_model.csv` |
| Full model: incentive effects δ against 0; correlation between δinf and δsel | `MS/full_model_stats/params/ttests_motiv_vs0.csv`; `MS/full_model_stats/correlations/corr_motiv_params.csv` |
| Shares of the choice variance at £0 (79%, 4%, 17%) and of the incentive effect (12%: 9% inference, 3% selection) | `MS/full_model_stats/params/variance_partition_summary.csv` |
| Models without one incentive effect (knockouts): error of the predicted incentive effect | `MS/full_model_stats/params/r2_var_red_natural_abs_obs_vs_knockout_model.csv` |
| Inference-only vs full model (group BMC, PXP) | `MS/full_vs_inference_only/groupbmc_full_vs_infv_vs_noise_per_cond.csv` (`scripts/exploratory/stats/run_full_vs_inference_only_bmc.py`) |
| Slowing under incentives (11 ms, 1.3%; 94 ms, 4.0%) | `PT` → `table2` → `merged` → `categ_rt`, `estim_rt` → `pooled` (`mean_diff` × `scale`, in ms; % of `desc` → `neutral` → `mean`) |

## Model fits and simulation studies (`--stage fit`)

Inputs of the stats stage. The fits (`.npz`) come with the data bundle; the outputs of the cleaning checks and of the
simulation studies are in this repository. All use seed 42; which environment produced each file is in
`data/README.md`.

| Fit | Script (settings used for the paper) | Output |
|---|---|---|
| Cleaning, behaviour, single-noise and single-modulation fits (pre-registered) | `scripts/prereg/run_prereg_pipeline.py --exp-name <exp>` (BADS, 10 random starts) | `data/interim/<exp>/`, `data/processed/<exp>/`, `results/prereg/<exp>/` |
| Full model | `scripts/exploratory/fits/fit_full_model.py --exp-name <exp> --vbmc-after-bads` (BADS from 10 random starts, then VBMC; priors of Table 4) | `results/exploratory/<exp>/modeling/fit_files/fit_full_model.npz` |
| Inference-only model on fast and slow trials (Fig. 5c, Supp. Table 9) | `scripts/exploratory/fits/fit_inf_per_cond_rt_split.py --exp-name <exp>` | `results/exploratory/<exp>/modeling/fit_files/fit_inf_rt_split.npz` |
| Model recovery, single-noise models (Supp. Fig. 2a) | `scripts/exploratory/fits/run_identifiability_single_noise.py --exp-name exp1 --n-reps 20` | `results/exploratory/exp1/modeling/model_identifiability/` |
| Model recovery, single-modulation models (Supp. Fig. 4b) | `scripts/exploratory/fits/run_identifiability_single_modulation.py --exp-name exp1 --n-reps 20` | `results/exploratory/exp1/modeling/model_motiv_identifiability/` |
| Parameter recovery, inference-only model (Supp. Fig. 2b) | `scripts/exploratory/fits/run_recovery_inference_only.py --exp-name exp1` | `results/exploratory/exp1/modeling/recovery_noise_per_cond/` |
| Parameter recovery, full model (Supp. Fig. 5) | `scripts/exploratory/fits/run_recovery_full_model.py --exp-name merged --bads-nit 10 --vbmc-after-bads` (one simulated dataset per participant of both experiments) | `results/exploratory/merged/modeling/param_recovery_full_model/` |
