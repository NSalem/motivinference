"""Choice-variance predictions of the inference-only model (Figure 2d, the R² of the Results).

The model's decision variable has variance ``Σ var(ℓ_i) + (n − 1) σ_inf² + σ_sel²`` (``motinf.prereg.model.psychofun``).
In the inference-only model sensory and selection noise are zero, so each sample's log-likelihood ratio ``ℓ_i`` is
known exactly (``var(ℓ_i) = 0``) and the predicted choice variability of a sequence of length n is
``(n − 1) σ_inf²``, with ``σ_inf`` fitted per participant and incentive. The single-modulation models are predicted
with the pre-registered ``psychofun`` itself (``single_modulation_fits.predict_choicevar_per_seqlen_categ``).
"""

from __future__ import annotations

import pandas as pd


def predict_inf_choicevar_per_seqlen(
    df_trials: pd.DataFrame, df_sd_inf_cell: pd.DataFrame
) -> pd.DataFrame:
    """``pred_choice_var = (seqlen − 1) · sd_inf²`` per (participant, incentive, seqlen) cell of ``df_trials`` that has a
    fitted ``sd_inf`` in ``df_sd_inf_cell`` (participant, incentive, sd_inf)."""
    keys = ["participant", "incentive", "seqlen"]
    d = df_trials[keys].drop_duplicates().merge(df_sd_inf_cell, on=["participant", "incentive"], how="inner")
    d["pred_choice_var"] = (d["seqlen"].to_numpy(float) - 1.0) * d["sd_inf"].to_numpy(float) ** 2
    return d[[*keys, "pred_choice_var"]].sort_values(keys).reset_index(drop=True)
