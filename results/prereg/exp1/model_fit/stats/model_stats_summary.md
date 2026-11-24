# Preregistered model-fit statistics

Experiment: `exp1`

## Group BMC — per-condition fits (by incentive)

|   incentive | model   |   n |   frequency_mean |   frequency_var |        xp |       pxp |
|------------:|:--------|----:|-----------------:|----------------:|----------:|----------:|
|          -1 | ang     |  84 |         0.00452  |       5.232e-05 | 2.093e-28 | 2.174e-11 |
|          -1 | inf     |  84 |         0.9613   |       0.000433  | 1         | 1         |
|          -1 | sel     |  84 |         0.03422  |       0.0003843 | 1.249e-24 | 2.174e-11 |
|           0 | ang     |  84 |         0.004616 |       5.343e-05 | 7.63e-27  | 5.128e-08 |
|           0 | inf     |  84 |         0.9212   |       0.0008442 | 1         | 1         |
|           0 | sel     |  84 |         0.0742   |       0.0007987 | 3.472e-20 | 5.128e-08 |
|           1 | ang     |  84 |         0.004513 |       5.224e-05 | 1.445e-29 | 4.154e-12 |
|           1 | inf     |  84 |         0.9904   |       0.0001103 | 1         | 1         |
|           1 | sel     |  84 |         0.005061 |       5.855e-05 | 1.899e-29 | 4.154e-12 |

## Inferential tests on `log(sd_inf)` (inf-only parameters, per condition)

### Repeated-measures ANOVA (`log(sd_inf)` ~ incentive)

| Source    |     SS |   DF |      MS |       F |       p_unc |   p_GG_corr |      np2 |     eps |   sphericity |   W_spher |     p_spher |   eps_exact |   p_GG_exact |
|:----------|-------:|-----:|--------:|--------:|------------:|------------:|---------:|--------:|-------------:|----------:|------------:|------------:|-------------:|
| incentive | 0.7245 |    2 | 0.3622  |   9.548 |   0.0001188 |    0.001216 |   0.1032 |   0.628 |            0 |    0.4076 |   1.047e-16 |       0.628 |     0.001216 |
| Error     | 6.298  |  166 | 0.03794 | nan     | nan         |  nan        | nan      | nan     |          nan |  nan      | nan         |     nan     |   nan        |

### Paired one-sided t-test (`log(sd_inf)` at |incentive| 1 vs 0; mean within ±1 pooled; `alternative='less'`)

| test                              |      T |   dof | alternative   |     p_val | CI95          |   cohen_d |   power |
|:----------------------------------|-------:|------:|:--------------|----------:|:--------------|----------:|--------:|
| log(sd_inf): abs incentive 1 vs 0 | -3.289 |    83 | less          | 0.0007386 | [ -inf -0.06] |    0.2537 |  0.7458 |

## Group BMC — incentive-effects models

| model   |   n |   frequency_mean |   frequency_var |        xp |       pxp |
|:--------|----:|-----------------:|----------------:|----------:|----------:|
| ang_v   |  84 |         0.004396 |       5.089e-05 | 2.595e-22 | 0.0001044 |
| inf_v   |  84 |         0.7901   |       0.001928  | 1         | 0.9998    |
| sel_v   |  84 |         0.2055   |       0.001898  | 6.265e-11 | 0.0001044 |

