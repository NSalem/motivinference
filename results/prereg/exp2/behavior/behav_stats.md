# Behavioral statistics (prereg)

## rm_anova_sensitivity_incentive

|    | Source    |    SS |   DF |    MS |       F |   p_unc |     np2 |    eps |   eps_exact |   p_GG_exact |
|----|-----------|-------|------|-------|---------|---------|---------|--------|-------------|--------------|
|  0 | incentive | 0.094 |    2 | 0.047 |   9.073 |       0 |   0.065 |   0.98 |        0.98 |            0 |
|  1 | Error     | 1.352 |  262 | 0.005 | nan     |     nan | nan     | nan    |      nan    |          nan |

### post-hoc

|    | Contrast   |   A |   B | Paired   | Parametric   |      T |   dof | alternative   |   p_unc |   p_corr | p_adjust   |   BF10 |   cohen |
|----|------------|-----|-----|----------|--------------|--------|-------|---------------|---------|----------|------------|--------|---------|
|  0 | incentive  |  -1 |   0 | True     | True         |  3.693 |   131 | two-sided     |   0     |    0.001 | holm       | 55.953 |   0.164 |
|  1 | incentive  |  -1 |   1 | True     | True         | -0.375 |   131 | two-sided     |   0.708 |    0.708 | holm       |  0.104 |  -0.016 |
|  2 | incentive  |   0 |   1 | True     | True         | -3.623 |   131 | two-sided     |   0     |    0.001 | holm       | 44.438 |  -0.18  |

## ttest_sensitivity_abs_1_gt_0

|        | test                              |     T |   dof | alternative   |   p_val | CI95        |   cohen_d |   power |
|--------|-----------------------------------|-------|-------|---------------|---------|-------------|-----------|---------|
| T_test | sensitivity: abs incentive 1 vs 0 | 4.185 |   131 | greater       |       0 | [0.02  inf] |     0.175 |   0.639 |

## rm_anova_choice_var_log_seqlen_x_incentive

|    | Source             |      SS |   ddof1 |   ddof2 |      MS |       F |   p_unc |   p_GG_corr |   np2 |   eps |   eps_exact |   p_GG_exact |
|----|--------------------|---------|---------|---------|---------|---------|---------|-------------|-------|-------|-------------|--------------|
|  0 | seqlen             | 297.757 |       2 |     262 | 148.878 | 242.112 |   0     |       0     | 0.649 | 0.846 |       0.846 |        0     |
|  1 | incentive          |   6.564 |       2 |     262 |   3.282 |   6.533 |   0.002 |       0.006 | 0.048 | 0.693 |       0.693 |        0.006 |
|  2 | seqlen * incentive |   0.499 |       4 |     524 |   0.125 |   0.512 |   0.727 |       0.621 | 0.004 | 0.561 |       0.813 |        0.689 |

## rm_anova_estim_sd_incentive

|    | Source    |      SS |   DF |    MS |       F |   p_unc |     np2 |     eps |   eps_exact |   p_GG_exact |
|----|-----------|---------|------|-------|---------|---------|---------|---------|-------------|--------------|
|  0 | incentive |   2.831 |    2 | 1.416 |   0.481 |   0.619 |   0.004 |   0.999 |       0.999 |        0.618 |
|  1 | Error     | 771.055 |  262 | 2.943 | nan     | nan     | nan     | nan     |     nan     |      nan     |

### post-hoc

|    | Contrast   |   A |   B | Paired   | Parametric   |      T |   dof | alternative   |   p_unc |   p_corr | p_adjust   |   BF10 |   cohen |
|----|------------|-----|-----|----------|--------------|--------|-------|---------------|---------|----------|------------|--------|---------|
|  0 | incentive  |  -1 |   0 | True     | True         | -0.292 |   131 | two-sided     |   0.771 |    1     | holm       |  0.101 |  -0.015 |
|  1 | incentive  |  -1 |   1 | True     | True         |  0.663 |   131 | two-sided     |   0.509 |    1     | holm       |  0.12  |   0.036 |
|  2 | incentive  |   0 |   1 | True     | True         |  0.972 |   131 | two-sided     |   0.333 |    0.998 | holm       |  0.153 |   0.051 |

## ttest_estim_sd_abs_1_vs_0

|        | test                           |      T |   dof | alternative   |   p_val | CI95        |   cohen_d |   power |
|--------|--------------------------------|--------|-------|---------------|---------|-------------|-----------|---------|
| T_test | estim_sd: abs incentive 1 vs 0 | -0.723 |   131 | less          |   0.236 | [-inf 0.17] |     0.034 |   0.104 |

## rm_anova_exact_gg

|    | dv             | Source             |      SS |   ddof1 |   ddof2 |      MS |       F |   p_unc |   p_GG_corr |   np2 |   eps |   n_participants |
|----|----------------|--------------------|---------|---------|---------|---------|---------|---------|-------------|-------|-------|------------------|
|  0 | sensitivity    | incentive          |   0.094 |       2 |     262 |   0.047 |   9.073 |   0     |       0     | 0.065 | 0.98  |              132 |
|  1 | choice_var_log | seqlen             | 297.757 |       2 |     262 | 148.878 | 242.112 |   0     |       0     | 0.649 | 0.846 |              132 |
|  2 | choice_var_log | incentive          |   6.564 |       2 |     262 |   3.282 |   6.533 |   0.002 |       0.006 | 0.048 | 0.693 |              132 |
|  3 | choice_var_log | seqlen * incentive |   0.499 |       4 |     524 |   0.125 |   0.512 |   0.727 |       0.689 | 0.004 | 0.813 |              132 |
|  4 | estim_sd       | incentive          |   2.831 |       2 |     262 |   1.416 |   0.481 |   0.619 |       0.618 | 0.004 | 0.999 |              132 |
