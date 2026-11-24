"""Pre-registered single-modulation models (``fit_inc_effects.npz``): loading and predictions.

``scripts/prereg/fit_incentive_effects.py`` fits three models per participant (``ang_v``, ``inf_v``,
``sel_v``). Each has all three noise sources (``sd_ang``, ``sd_inf``, ``sd_sel``) free and lets only one of
them take a separate value in incentivized conditions (``sd_X_v``). This module loads those fits (per
experiment or pooled), predicts each model's choice variability per participant x incentive x sequence
length with the pre-registered generative model (``motinf.prereg.model.psychofun``,
``choicevar = spost**2``), and summarises the predicted incentive effect by sequence length.

Used for Supplementary Figure 4d (``scripts/exploratory/stats/run_inference_model_stats.py`` writes the predictions,
``scripts/exploratory/figures/sfig4.py`` draws them) and for the AICc of ``inf_v`` in
``scripts/exploratory/stats/run_full_vs_inference_only_bmc.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motinf.exploratory.choice_var_seqlen_preprocess import var_red_natural_long_by_seqlen
from motinf.exploratory.cohorts import participant_prefixed
# NOTE: this uses the *preregistered* model implementation (motinf.prereg.model), not
# motinf.exploratory.model — fit_inc_effects.npz was produced by
# scripts/prereg/fit_incentive_effects.py, which fits (and therefore must be evaluated) with
# motinf.prereg.model.psychofun/param_config_categ, a separate copy of the generative model with
# its own bounds table. The exploratory module's get_ppost_fast additionally applies
# clip_param(...) bounds-clipping on every read; mixing the two risks silently re-clipping an
# already-in-bounds prereg-fitted parameter against a different bounds table.
from motinf.prereg.model import df_to_pack, param_config_categ, psychofun

# The three single-modulation models of fit_inc_effects.npz: the incentive acts on sensory, inference or selection noise
SINGLE_MODULATION_MODELS = ("ang_v", "inf_v", "sel_v")


@dataclass(frozen=True)
class CategSingleIncentiveFit:
    """Parsed contents of a ``fit_inc_effects.npz`` bundle."""

    participants: list[str]
    model_names: list[str]
    specs: dict[str, dict[str, Any]]  # model_name -> spec dict (parnames, fixedpars_keys, label)
    pars_categ: dict[str, dict[str, np.ndarray]]  # model_name -> {parname: (n_participants,) array}
    aicc: dict[str, np.ndarray]  # model_name -> (n_participants,)
    bic: dict[str, np.ndarray]
    ll: dict[str, np.ndarray]
    use_priors_categ: bool


def load_categ_single_incentive_fit(npz_path: Path) -> CategSingleIncentiveFit:
    """Load and reshape a ``fit_inc_effects.npz`` bundle."""
    d = np.load(npz_path, allow_pickle=True)
    participants = [str(p) for p in d["participants"]]
    n_p = len(participants)
    model_names = [str(n) for n in d["categ_model_names"]]
    specs_raw = list(d["categ_model_specs"])
    specs = {str(s["name"]): dict(s) for s in specs_raw}

    pars_categ: dict[str, dict[str, np.ndarray]] = {}
    aicc: dict[str, np.ndarray] = {}
    bic: dict[str, np.ndarray] = {}
    ll: dict[str, np.ndarray] = {}
    for mi, mname in enumerate(model_names):
        parnames = [str(pn) for pn in specs[mname]["parnames"]]
        mat: dict[str, list[float]] = {pn: [] for pn in parnames}
        for pi in range(n_p):
            row = d["pars_categ"][mi, pi]
            row = np.asarray(row, dtype=float).ravel() if row is not None else None
            for j, pn in enumerate(parnames):
                mat[pn].append(
                    float(row[j]) if row is not None and row.size == len(parnames) else float("nan")
                )
        pars_categ[mname] = {pn: np.asarray(v, dtype=float) for pn, v in mat.items()}
        aicc[mname] = np.asarray(d["aicc_categ"][mi], dtype=float)
        bic[mname] = np.asarray(d["bic_categ"][mi], dtype=float)
        ll[mname] = np.asarray(d["ll_categ"][mi], dtype=float)

    return CategSingleIncentiveFit(
        participants=participants,
        model_names=model_names,
        specs=specs,
        pars_categ=pars_categ,
        aicc=aicc,
        bic=bic,
        ll=ll,
        use_priors_categ=bool(d["use_priors_categ"]),
    )


def load_categ_single_incentive_fit_merged(
    root: Path,
    segments: list[str],
    *,
    npz_relpath: str = "model_fit/fit_inc_effects.npz",
) -> CategSingleIncentiveFit:
    """Load and concatenate ``fit_inc_effects.npz`` across cohort segments (e.g. exp1 + exp2).

    Participant ids are prefixed ``{segment}__{id}`` (matching
    :func:`motinf.exploratory.cohorts.participant_prefixed`) when more than one segment is
    given, so they line up with ``merge_inf_trials_for_choice_var`` / ``merge_behavior_seqlen``
    called with the same segment list. All segments must share the same ``categ_model_names`` /
    per-model ``parnames`` (true for exp1 vs exp2's ``fit_inc_effects.npz``, both produced by the
    same ``fit_incentive_effects_prereg`` script).
    """
    prefix_pids = len(segments) > 1
    fits = [
        load_categ_single_incentive_fit(root / "results" / "prereg" / seg / npz_relpath)
        for seg in segments
    ]
    base = fits[0]
    for seg, fit in zip(segments[1:], fits[1:]):
        if fit.model_names != base.model_names:
            raise ValueError(
                f"categ_model_names mismatch between segments: {segments[0]}={base.model_names} "
                f"vs {seg}={fit.model_names}"
            )
        for mname in base.model_names:
            if set(fit.pars_categ[mname]) != set(base.pars_categ[mname]):
                raise ValueError(
                    f"parnames mismatch for model {mname!r} between segments {segments[0]} and {seg}"
                )

    participants: list[str] = []
    for seg, fit in zip(segments, fits):
        pids = fit.participants
        participants.extend(
            [participant_prefixed(seg, p) for p in pids] if prefix_pids else pids
        )

    pars_categ: dict[str, dict[str, np.ndarray]] = {}
    aicc: dict[str, np.ndarray] = {}
    bic: dict[str, np.ndarray] = {}
    ll: dict[str, np.ndarray] = {}
    for mname in base.model_names:
        pars_categ[mname] = {
            pn: np.concatenate([fit.pars_categ[mname][pn] for fit in fits])
            for pn in base.pars_categ[mname]
        }
        aicc[mname] = np.concatenate([fit.aicc[mname] for fit in fits])
        bic[mname] = np.concatenate([fit.bic[mname] for fit in fits])
        ll[mname] = np.concatenate([fit.ll[mname] for fit in fits])

    return CategSingleIncentiveFit(
        participants=participants,
        model_names=base.model_names,
        specs=base.specs,
        pars_categ=pars_categ,
        aicc=aicc,
        bic=bic,
        ll=ll,
        use_priors_categ=base.use_priors_categ,
    )


def _pars_dict_for_participant(fit: CategSingleIncentiveFit, model_name: str, pi: int) -> dict[str, float] | None:
    """Build the ``pars`` dict ``psychofun`` expects, for one model × one participant (all parameters free)."""
    if fit.specs[model_name].get("fixedpars_keys"):
        raise ValueError(f"{model_name}: fixed parameters are not supported")
    pars: dict[str, float] = {}
    for pn, arr in fit.pars_categ[model_name].items():
        val = float(arr[pi])
        if not np.isfinite(val):
            return None
        pars[pn] = val
    return pars


def predict_choicevar_per_seqlen_categ(
    df_inf: pd.DataFrame,
    fit: CategSingleIncentiveFit,
    model_name: str,
) -> pd.DataFrame:
    """Trial-level predictions -> participant x incentive x seqlen ``pred_choice_var`` for one model.

    Uses ``motinf.prereg.model.psychofun`` (``choicevar = spost**2``) — the same generative-model
    implementation ``scripts/prereg/fit_incentive_effects.py`` used to fit these parameters — fed
    each categ model's own directly-fit ``sd_X`` / ``sd_X_v`` params. Note ``psychofun`` returns
    ``(p, spost, xpost)``, a different order than the exploratory module's ``get_ppost_fast``
    (``ppost, xpost, spost``); don't mix them up if extending this.
    """
    if model_name not in fit.model_names:
        raise KeyError(f"Unknown model {model_name!r}; available: {fit.model_names}")
    df_inf = df_inf.copy()
    df_inf["participant"] = df_inf["participant"].astype(str)
    df_inf["seqlen"] = df_inf["seqlen"].astype(int)
    df_inf["incentive"] = df_inf["incentive"].astype(float)

    pid_to_idx = {pid: i for i, pid in enumerate(fit.participants)}
    chunks: list[pd.DataFrame] = []
    for pid, df_sub in df_inf.groupby("participant"):
        pi = pid_to_idx.get(pid)
        if pi is None:
            continue
        pars = _pars_dict_for_participant(fit, model_name, pi)
        if pars is None:
            continue
        data = df_to_pack(df_sub)
        _p, spost, _xpost = psychofun(pars, data, kappa=0.5, sigstr=1, bounds_config=param_config_categ)
        out = df_sub[["participant", "incentive", "seqlen"]].copy().reset_index(drop=True)
        out["choicevar"] = np.asarray(spost, dtype=float) ** 2
        chunks.append(out)

    if not chunks:
        return pd.DataFrame(columns=["participant", "incentive", "seqlen", "pred_choice_var"])
    df_pred = pd.concat(chunks, ignore_index=True)
    return (
        df_pred.groupby(["participant", "incentive", "seqlen"], as_index=False)
        .agg(pred_choice_var=("choicevar", "mean"))
        .sort_values(["participant", "incentive", "seqlen"])
        .reset_index(drop=True)
    )


def pooled_reduction_summary_by_incentive_abs(m_nat: pd.DataFrame, model_label: str) -> pd.DataFrame:
    """Group mean ± SEM per sequence length of the incentive effect (neutral − |incentive| = 1, gain and
    loss pooled), observed (``beh_red_*``) and predicted (``pred_red_*``); one ``incentive`` = 1.0 row per
    sequence length."""
    obs_long = var_red_natural_long_by_seqlen(m_nat, "choice_var")
    pred_long = var_red_natural_long_by_seqlen(m_nat, "pred_choice_var")
    if obs_long.empty or pred_long.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for sl in sorted(set(obs_long["seqlen"]) & set(pred_long["seqlen"])):
        b = obs_long.loc[obs_long["seqlen"] == sl, "var_red"].to_numpy(float)
        p = pred_long.loc[pred_long["seqlen"] == sl, "var_red"].to_numpy(float)
        b = b[np.isfinite(b)]
        p = p[np.isfinite(p)]
        if b.size == 0 or p.size == 0:
            continue
        rows.append(
            {
                "model": model_label,
                "incentive": 1.0,
                "seqlen": int(sl),
                "beh_red_mean": float(np.mean(b)),
                "beh_red_sem": float(np.std(b, ddof=1) / np.sqrt(b.size)) if b.size > 1 else 0.0,
                "pred_red_mean": float(np.mean(p)),
                "pred_red_sem": float(np.std(p, ddof=1) / np.sqrt(p.size)) if p.size > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)
