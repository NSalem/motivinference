# Behavioral statistics (prereg)

## rm_anova_sensitivity_incentive

|    | Source    |    SS |   DF |    MS |       F |   p_unc |   p_GG_corr |     np2 |     eps |   sphericity |   W_spher |   p_spher |   eps_exact |   p_GG_exact |
|----|-----------|-------|------|-------|---------|---------|-------------|---------|---------|--------------|-----------|-----------|-------------|--------------|
|  0 | incentive | 0.104 |    2 | 0.052 |   7.589 |   0.001 |       0.001 |   0.084 |   0.864 |            0 |     0.842 |     0.001 |       0.864 |        0.001 |
|  1 | Error     | 1.142 |  166 | 0.007 | nan     | nan     |     nan     | nan     | nan     |          nan |   nan     |   nan     |     nan     |      nan     |

### post-hoc

|    | Contrast   |   A |   B | Paired   | Parametric   |      T |   dof | alternative   |   p_unc |   p_corr | p_adjust   |   BF10 |   cohen |
|----|------------|-----|-----|----------|--------------|--------|-------|---------------|---------|----------|------------|--------|---------|
|  0 | incentive  |  -1 |   0 | True     | True         |  2.979 |    83 | two-sided     |   0.004 |    0.008 | holm       |  7.196 |   0.203 |
|  1 | incentive  |  -1 |   1 | True     | True         | -0.506 |    83 | two-sided     |   0.614 |    0.614 | holm       |  0.136 |  -0.025 |
|  2 | incentive  |   0 |   1 | True     | True         | -3.16  |    83 | two-sided     |   0.002 |    0.007 | holm       | 11.682 |  -0.227 |

## ttest_sensitivity_abs_1_gt_0

|        | test                              |     T |   dof | alternative   |   p_val | CI95        |   cohen_d |   power |
|--------|-----------------------------------|-------|-------|---------------|---------|-------------|-----------|---------|
| T_test | sensitivity: abs incentive 1 vs 0 | 3.289 |    83 | greater       |   0.001 | [0.02  inf] |     0.218 |    0.63 |

## rm_anova_choice_var_log_seqlen_x_incentive

|    | Source             |      SS |   ddof1 |   ddof2 |     MS |       F |   p_unc |   p_GG_corr |   np2 |   eps |   eps_exact |   p_GG_exact |
|----|--------------------|---------|---------|---------|--------|---------|---------|-------------|-------|-------|-------------|--------------|
|  0 | seqlen             | 197.77  |       2 |     166 | 98.885 | 185.061 |   0     |       0     | 0.69  | 0.94  |       0.94  |        0     |
|  1 | incentive          |  13.592 |       2 |     166 |  6.796 |   9.212 |   0     |       0.002 | 0.1   | 0.597 |       0.597 |        0.002 |
|  2 | seqlen * incentive |   1.328 |       4 |     332 |  0.332 |   1.238 |   0.295 |       0.29  | 0.015 | 0.434 |       0.715 |        0.296 |

## rm_anova_estim_sd_incentive

|    | Source    |      SS |   DF |    MS |       F |   p_unc |     np2 |     eps |   eps_exact |   p_GG_exact |
|----|-----------|---------|------|-------|---------|---------|---------|---------|-------------|--------------|
|  0 | incentive |  15.05  |    2 | 7.525 |   2.718 |   0.069 |   0.032 |   0.939 |       0.939 |        0.073 |
|  1 | Error     | 459.534 |  166 | 2.768 | nan     | nan     | nan     | nan     |     nan     |      nan     |

### post-hoc

|    | Contrast   |   A |   B | Paired   | Parametric   |      T |   dof | alternative   |   p_unc |   p_corr | p_adjust   |   BF10 |   cohen |
|----|------------|-----|-----|----------|--------------|--------|-------|---------------|---------|----------|------------|--------|---------|
|  0 | incentive  |  -1 |   0 | True     | True         | -1.984 |    83 | two-sided     |   0.051 |    0.152 | holm       |  0.776 |  -0.133 |
|  1 | incentive  |  -1 |   1 | True     | True         | -0.02  |    83 | two-sided     |   0.984 |    0.984 | holm       |  0.12  |  -0.001 |
|  2 | incentive  |   0 |   1 | True     | True         |  1.839 |    83 | two-sided     |   0.069 |    0.152 | holm       |  0.601 |   0.134 |

## ttest_estim_sd_abs_1_vs_0

|        | test                           |      T |   dof | alternative   |   p_val | CI95          |   cohen_d |   power |
|--------|--------------------------------|--------|-------|---------------|---------|---------------|-----------|---------|
| T_test | estim_sd: abs incentive 1 vs 0 | -2.094 |    83 | less          |    0.02 | [ -inf -0.11] |     0.136 |    0.34 |

## rm_anova_exact_gg

|    | dv             | Source             |      SS |   ddof1 |   ddof2 |     MS |       F |   p_unc |   p_GG_corr |   np2 |   eps |   n_participants |
|----|----------------|--------------------|---------|---------|---------|--------|---------|---------|-------------|-------|-------|------------------|
|  0 | sensitivity    | incentive          |   0.104 |       2 |     166 |  0.052 |   7.589 |   0.001 |       0.001 | 0.084 | 0.864 |               84 |
|  1 | choice_var_log | seqlen             | 197.77  |       2 |     166 | 98.885 | 185.061 |   0     |       0     | 0.69  | 0.94  |               84 |
|  2 | choice_var_log | incentive          |  13.592 |       2 |     166 |  6.796 |   9.212 |   0     |       0.002 | 0.1   | 0.597 |               84 |
|  3 | choice_var_log | seqlen * incentive |   1.328 |       4 |     332 |  0.332 |   1.238 |   0.295 |       0.296 | 0.015 | 0.715 |               84 |
|  4 | estim_sd       | incentive          |  15.05  |       2 |     166 |  7.525 |   2.718 |   0.069 |       0.073 | 0.032 | 0.939 |               84 |
