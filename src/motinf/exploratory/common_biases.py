"""Common response biases (Supplementary Figure 7, Supplementary Table 6), used by
``scripts/exploratory/stats/run_bias_stats.py``.

Data preparation (cardinal / serial estimation bias, choice-history columns, merging of cohorts):

- ``bias_per_ori_to_long_df``, ``aggregate_mean_abs_dev_from_long``, ``cardinal_estimation_mean_angles``,
  ``add_estimation_serial_columns``, ``prep_inf_for_categ_biases`` (estimation trials are merged by
  ``cohorts.merge_data_clean_est``).

Bias models:

- ``get_ksr_curves``: kernel-smoothed (local linear) estimation bias against angle, per participant x incentive.
- ``fit_sequential_bias``: evidence weight per sample position (L2 logistic regression of choice on the
  sample evidence aligned to the last sample), per participant x incentive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from statsmodels.nonparametric.kernel_regression import KernelReg as kr


def mean_abs_dev(x: np.ndarray | pd.Series, axis: int = 0) -> np.floating | np.ndarray:
    """Mean of absolute values along ``axis``."""
    arr = np.asarray(x, dtype=float)
    return np.mean(np.abs(arr), axis=axis)


def bias_per_ori_to_long_df(
    bias_per_ori: np.ndarray,
    incentives: np.ndarray,
    participants: np.ndarray,
    x_grid: np.ndarray,
    *,
    task: str,
) -> pd.DataFrame:
    """Flatten ``bias_per_ori`` (n_inc, n_sub, n_x) to a long DataFrame."""
    n_incentive, n_participant, n_angle = bias_per_ori.shape
    inc_vals = np.repeat(np.asarray(incentives, dtype=float), n_participant * n_angle)
    pids = np.tile(np.repeat(np.asarray(participants).astype(str), n_angle), n_incentive)
    angles = np.tile(x_grid, n_incentive * n_participant)
    bias = bias_per_ori.reshape(-1)
    return pd.DataFrame(
        {
            "task": task,
            "incentive": inc_vals,
            "participant": pids,
            "angle": angles,
            "bias": bias,
        }
    )


def aggregate_mean_abs_dev_from_long(df_curves: pd.DataFrame) -> pd.DataFrame:
    """Per participant × incentive: mean of |bias| across angle samples."""
    g = df_curves.groupby(["task", "incentive", "participant"], observed=True)["bias"].apply(
        lambda s: float(mean_abs_dev(s.to_numpy(), axis=0))
    )
    out = g.reset_index(name="mean_abs_dev")
    return out


def cardinal_estimation_mean_angles(df_est: pd.DataFrame) -> pd.DataFrame:
    """Group-mean deviations per (participant, noisy angle, incentive); add degrees."""
    data_mean_ang = (
        df_est.groupby(["participant", "estimang_noisy", "incentive"], observed=True)[["dev"]]
        .mean()
        .reset_index()
    )
    data_mean_ang["estimang_deg"] = np.rad2deg(data_mean_ang["estimang_noisy"])
    data_mean_ang["dev_deg"] = np.rad2deg(data_mean_ang["dev"])
    return data_mean_ang


def add_estimation_serial_columns(df_est: pd.DataFrame) -> pd.DataFrame:
    """Add ``rel_prev_ang``, ``rel_prev_resp``, ``dev_demean``, ``dev_demean_deg`` per participant."""
    df_ref = df_est.copy()
    wrap_angle = np.pi
    for isub in np.unique(df_ref["participant"].to_numpy()):
        m = df_ref["participant"] == isub
        df_sub = df_ref.loc[m].reset_index(drop=True)
        df_sub["prev_ang"] = df_sub["estimang_noisy"].shift(1, fill_value=0.0)
        df_sub["prev_resp"] = df_sub["resp_estim"].shift(1, fill_value=0.0)
        df_sub["rel_prev_ang"] = (
            (df_sub["estimang_noisy"] - df_sub["prev_ang"] + wrap_angle / 2) % wrap_angle
        ) - wrap_angle / 2
        df_sub["dev_demean"] = df_sub["dev_noisy"] - np.mean(df_sub["dev_noisy"].to_numpy())
        df_sub["rel_prev_resp"] = (
            (df_sub["resp_estim"] - df_sub["prev_resp"] + wrap_angle / 2) % wrap_angle
        ) - wrap_angle / 2
        df_ref.loc[m, "rel_prev_ang"] = df_sub["rel_prev_ang"].to_numpy()
        df_ref.loc[m, "rel_prev_resp"] = df_sub["rel_prev_resp"].to_numpy()
        df_ref.loc[m, "dev_demean"] = df_sub["dev_demean"].to_numpy()
    # The serial-bias KSR uses the participant-demeaned deviation in degrees.
    df_ref["dev_demean_deg"] = np.rad2deg(df_ref["dev_demean"].astype(float)).round(1)
    return df_ref


def prep_inf_for_categ_biases(df_inf: pd.DataFrame) -> pd.DataFrame:
    """Sort trials and add session-aware ``prev_resp_side``."""
    sort_cols = [c for c in ("participant", "session", "trialN") if c in df_inf.columns]
    if len(sort_cols) < 3:
        raise KeyError("df_inf must contain participant, session, trialN for categ bias prep")
    df = df_inf.sort_values(sort_cols, kind="mergesort").copy()
    prev_resp_side = df.groupby(["participant", "session"], sort=False)["resp_side"].shift(1)
    df["prev_resp_side"] = prev_resp_side
    return df


# ---- bias models ----

N_SEQ_POSITIONS = 12  # longest sequence; position 11 = last sample before the choice


def _ksr_bias_curve(x, y, x_values, bw, fallback_bw=[15], verbose: bool = True):
    """Local linear kernel regression of the mean deviation per unique angle, evaluated at ``x_values``.

    If the fit has NaNs or the bandwidth falls outside [5, 45], it is refitted with ``fallback_bw``.
    """
    unique_angles = np.unique(x)
    avg_deviation_per_angle = [np.nanmean(y[x == angle]) for angle in unique_angles]

    kreg_bias = kr(endog=[avg_deviation_per_angle], exog=[unique_angles], var_type='c', reg_type='ll', bw=bw)
    yhat = kreg_bias.fit(x_values)[0]
    yhat_obs = kreg_bias.fit(x)[0]
    selected_bw = kreg_bias.bw
    if np.isnan(yhat).any() or np.isnan(yhat_obs).any() or selected_bw < 5 or selected_bw > 45:
        if verbose:
            print(f"NaNs or out-of-range bandwidth {selected_bw}; switching to fallback bandwidth {fallback_bw}")
        kreg_bias = kr(endog=[avg_deviation_per_angle], exog=[unique_angles], var_type='c', reg_type='ll', bw=[fallback_bw])
        yhat = kreg_bias.fit(x_values)[0]
    if verbose:
        print("bw:", kreg_bias.bw)
    return yhat


def get_ksr_curves(
    data,
    x_var_name="estimang_noisy",
    y_var_name="dev",
    x_vals_out=np.linspace(-np.pi, np.pi, 181),
    pad_size=90,
    bw=[10],
    verbose: bool = True,
):
    """Smoothed bias per participant x incentive.

    Angles are wrapped by ``pad_size`` on both ends of the 0-180 range before smoothing. Returns
    ``{"bias_per_ori": array (n_incentives, n_participants, len(x_vals_out))}`` with incentives and
    participants in sorted order.
    """
    participants = np.unique(data['participant'])
    incentives = np.unique(data['incentive'])
    bias_per_ori = np.zeros([len(incentives), len(participants), len(x_vals_out)])
    for nsub, participant in enumerate(participants):
        for ninc, iinc in enumerate(incentives):
            sel = (data['incentive'] == iinc) & (data['participant'] == participant)
            x = data[x_var_name][sel]
            y = data[y_var_name][sel]
            x_padded = np.concatenate([x[x > (180 - pad_size)] - 180, x, x[x < pad_size] + 180])
            y_padded = np.concatenate([y[x > (180 - pad_size)], y, y[x < pad_size]])
            bias_per_ori[ninc, nsub, :] = _ksr_bias_curve(x_padded, y_padded, x_vals_out, bw, verbose=verbose)
    return {'bias_per_ori': bias_per_ori}


def _last_aligned_evidence(df, base_col_prefix="seqllr_noisy_", seqlen_col="seqlen", max_len=N_SEQ_POSITIONS):
    """Trials x positions matrix of sample evidence, right-aligned so the last sample is in the last column
    (earlier positions of shorter sequences are 0)."""
    cols = [f"{base_col_prefix}{i}" for i in range(1, max_len + 1)]
    arr = np.vstack([df[c].to_numpy() for c in cols]).T
    out = np.zeros((len(df), max_len))
    seqlen = df[seqlen_col].to_numpy().astype(int)
    for L in np.unique(seqlen):
        idx = np.where(seqlen == L)[0]
        out[idx, -L:] = arr[idx, :L]
    return out


def _choice01(y):
    y = pd.Series(y)
    if set(y.unique()) == {-1, 1}:
        y = (y == 1).astype(int)
    else:
        y = y.astype(int).clip(0, 1)
    return y


def fit_sequential_bias(df, C=0.5):
    """Logistic-regression weight of each sample position on choice, per participant x incentive (long table:
    participant, incentive, position 0-11, beta)."""
    participants = df['participant'].unique()
    incentives = np.sort(df['incentive'].unique())
    coefs_all = np.zeros((len(participants), len(incentives), N_SEQ_POSITIONS))
    for nsub, isub in enumerate(participants):
        for ninc, iinc in enumerate(incentives):
            df_sub_inc = df[(df['participant'] == isub) & (df['incentive'] == iinc)]
            clf = LogisticRegression(penalty='l2', C=C, fit_intercept=True, max_iter=1000)
            clf.fit(_last_aligned_evidence(df_sub_inc), _choice01(df_sub_inc['choice']))
            coefs_all[nsub, ninc, :] = clf.coef_[0]

    rows = []
    for isub, participant in enumerate(participants):
        for iinc, incentive in enumerate(incentives):
            for pos in range(N_SEQ_POSITIONS):
                rows.append({
                    'participant': participant,
                    'incentive': incentive,
                    'position': pos,
                    'beta': coefs_all[isub, iinc, pos],
                })
    return pd.DataFrame(rows)
