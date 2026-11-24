# Preregistered model-fit statistics

Experiment: `exp2`

## Group BMC — per-condition fits (by incentive)

|   incentive | model   |   n |   frequency_mean |   frequency_var |        xp |       pxp |
|------------:|:--------|----:|-----------------:|----------------:|----------:|----------:|
|          -1 | ang     | 132 |         0.002965 |       2.206e-05 | 3.926e-39 | 7.294e-11 |
|          -1 | inf     | 132 |         0.898    |       0.0006835 | 1         | 1         |
|          -1 | sel     | 132 |         0.09903  |       0.0006659 | 1.674e-23 | 7.294e-11 |
|           0 | ang     | 132 |         0.002969 |       2.209e-05 | 7.355e-40 | 8.527e-14 |
|           0 | inf     | 132 |         0.9252   |       0.0005163 | 1         | 1         |
|           0 | sel     | 132 |         0.07181  |       0.0004974 | 2.495e-27 | 8.527e-14 |
|           1 | ang     | 132 |         0.002954 |       2.198e-05 | 6.739e-40 | 6.412e-13 |
|           1 | inf     | 132 |         0.9264   |       0.000509  | 1         | 1         |
|           1 | sel     | 132 |         0.07067  |       0.0004901 | 1.67e-27  | 6.412e-13 |

## Inferential tests on `log(sd_inf)` (inf-only parameters, per condition)

### Repeated-measures ANOVA (`log(sd_inf)` ~ incentive)

| Source    |    SS |   DF |      MS |       F |       p_unc |   p_GG_corr |       np2 |      eps |   sphericity |   W_spher |     p_spher |   eps_exact |   p_GG_exact |
|:----------|------:|-----:|--------:|--------:|------------:|------------:|----------:|---------:|-------------:|----------:|------------:|------------:|-------------:|
| incentive | 0.335 |    2 | 0.1675  |   7.481 |   0.0006925 |    0.001671 |   0.05402 |   0.8098 |            0 |    0.7652 |   2.787e-08 |      0.8098 |     0.001671 |
| Error     | 5.866 |  262 | 0.02239 | nan     | nan         |  nan        | nan       | nan      |          nan |  nan      | nan         |    nan      |   nan        |

### Paired one-sided t-test (`log(sd_inf)` at |incentive| 1 vs 0; mean within ±1 pooled; `alternative='less'`)

| test                              |      T |   dof | alternative   |     p_val | CI95          |   cohen_d |   power |
|:----------------------------------|-------:|------:|:--------------|----------:|:--------------|----------:|--------:|
| log(sd_inf): abs incentive 1 vs 0 | -3.204 |   131 | less          | 0.0008518 | [ -inf -0.03] |    0.1335 |  0.4528 |

## Group BMC — incentive-effects models

| model   |   n |   frequency_mean |   frequency_var |        xp |       pxp |
|:--------|----:|-----------------:|----------------:|----------:|----------:|
| ang_v   | 132 |          0.00291 |       2.165e-05 | 1.122e-41 | 2.055e-08 |
| inf_v   | 132 |          0.9863  |       0.0001012 | 1         | 1         |
| sel_v   | 132 |          0.01084 |       8e-05     | 2.856e-39 | 2.055e-08 |

