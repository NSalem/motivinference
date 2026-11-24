"""The choice model of the Sequence Categorization task (``motinf.exploratory.model``, ``motinf.prereg.model``).

1. Both likelihoods (the fast exploratory ``get_ppost_fast`` and the pre-registered ``psychofun``) give the choice
   probabilities of the generative process they approximate, simulated here independently: each grating's orientation
   is perturbed by sensory noise and contributes its exact log-likelihood ratio, each of the n − 1 updates of the
   accumulated evidence adds inference noise, the choice adds selection noise, and the choice is the sign. The models
   replace the sum of the per-grating evidence by a Gaussian with its exact mean and variance, so they agree with the
   simulation within its binomial error, plus at most about 0.002 (root-mean-square, in choice probability) when
   sensory noise is large and the other two small.
2. The two parametrisations of the incentive effect give the same likelihood (Methods): the pre-registered model's
   separate incentive noise σ_v equals the full model's σ·exp(−δ).
3. An exploratory BADS fit gives the same result whatever the number of worker processes, or with no process pool.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from motinf.exploratory import model as M
from motinf.prereg import model as P

KAPPA = 0.5  # concentration of the category distributions
ANGLES = [f"seqang_noisy_{i}" for i in range(1, 13)]
SOURCES = ("ang", "inf", "sel")


def _trials(n: int, seed: int) -> pd.DataFrame:
    """Sequences of 4, 8 or 12 orientations (radians, [0, π)) drawn from the two categories, random incentives."""
    rng = np.random.default_rng(seed)
    seqlen = rng.choice([4, 8, 12], n)
    mean = np.where(rng.choice([1, -1], n) == 1, 3 * np.pi / 4, np.pi / 4)
    ang = np.mod(mean[:, None] + rng.vonmises(0, KAPPA, (n, 12)) / 2, np.pi)  # von Mises on the doubled angle
    ang[np.arange(12)[None, :] >= seqlen[:, None]] = np.nan
    df = pd.DataFrame(ang, columns=ANGLES)
    df["seqlen"] = seqlen
    df["incentive"] = rng.choice([-1, 0, 1], n)
    df["choice"] = rng.choice([-1, 1], n)
    return df


def _simulated_p_choice_1(df: pd.DataFrame, pars: dict, reps: int, rng: np.random.Generator) -> np.ndarray:
    """P(choice = 1) per trial, by simulating the generative process `reps` times."""
    seq = df[ANGLES].to_numpy()
    shown = ~np.isnan(seq)
    incentive = df["incentive"].to_numpy() != 0
    sd = {k: np.where(incentive, pars[f"sd_{k}"] * np.exp(-pars[f"motiv_{k}"]), pars[f"sd_{k}"]) for k in SOURCES}
    n_updates = df["seqlen"].to_numpy() - 1
    ones = np.zeros(len(df))
    for _ in range(reps):
        perceived = seq + rng.normal(size=seq.shape) * sd["ang"][:, None]
        evidence = np.where(shown, 2 * KAPPA * np.cos(2 * (perceived - 3 * np.pi / 4)), 0.0).sum(axis=1)
        evidence += rng.normal(size=len(df)) * sd["inf"] * np.sqrt(n_updates)
        evidence += rng.normal(size=len(df)) * sd["sel"]
        ones += evidence + pars["choice_bias"] > 0
    return ones / reps


def _prereg_pars(pars: dict) -> dict:
    """The same model in the pre-registered parametrisation: σ_v = σ·exp(−δ)."""
    out = {k: v for k, v in pars.items() if not k.startswith("motiv_")}
    out.update({f"sd_{k}_v": pars[f"sd_{k}"] * np.exp(-pars[f"motiv_{k}"]) for k in SOURCES})
    return out


def _excess_rms(simulated: np.ndarray, p: np.ndarray, reps: int) -> float:
    """Root-mean-square difference between simulated and model probabilities beyond the simulation's binomial noise
    (independent of the number of simulated choices)."""
    noise = np.mean(p * (1 - p)) / reps
    return float(np.sqrt(max(np.mean((simulated - p) ** 2) - noise, 0.0)))


TOLERANCE = 0.006  # observed: at most 0.002 (Gaussian approximation of the summed evidence, σ_ang = 0.3 rad)
GRID = list(itertools.product([0.05, 0.3], [0.2, 1.0], [0.2, 1.5]))  # σ_ang (rad), σ_inf, σ_sel


@pytest.mark.parametrize("sd_ang, sd_inf, sd_sel", GRID)
def test_likelihoods_match_the_simulated_process(sd_ang: float, sd_inf: float, sd_sel: float) -> None:
    df = _trials(150, seed=1)
    pars = {"sd_ang": sd_ang, "sd_inf": sd_inf, "sd_sel": sd_sel, "motiv_ang": 0.2, "motiv_inf": 0.3,
            "motiv_sel": -0.2, "choice_bias": 0.1, "plapse_choice": 0.0}
    reps = 6000
    simulated = _simulated_p_choice_1(df, pars, reps, np.random.default_rng(2))
    fast = M.get_ppost_fast(pars, M.df_to_pack(df))[0]
    prereg = P.psychofun(_prereg_pars(pars), P.df_to_pack(df.assign(choice=1)))[0]
    for name, p in (("get_ppost_fast", fast), ("psychofun", prereg)):
        excess = _excess_rms(simulated, p, reps)
        assert excess < TOLERANCE, f"{name}: rms error beyond simulation noise = {excess:.4f}"
        assert abs(np.mean(simulated - p)) < 0.003, f"{name}: mean Δp = {np.mean(simulated - p):+.4f}"


def test_a_wrong_noise_variance_is_caught() -> None:
    """The tolerance is tight enough to catch a model whose inference noise is 10% too large (excess about 0.018)."""
    df = _trials(150, seed=1)
    pars = {"sd_ang": 0.05, "sd_inf": 1.0, "sd_sel": 0.2, "motiv_ang": 0.0, "motiv_inf": 0.0, "motiv_sel": 0.0,
            "choice_bias": 0.0, "plapse_choice": 0.0}
    reps = 6000
    simulated = _simulated_p_choice_1(df, pars, reps, np.random.default_rng(2))
    wrong = M.get_ppost_fast(dict(pars, sd_inf=1.1), M.df_to_pack(df))[0]
    assert _excess_rms(simulated, wrong, reps) > 2 * TOLERANCE


@pytest.mark.parametrize("seed", range(5))
def test_the_two_parametrisations_share_the_likelihood(seed: int) -> None:
    rng = np.random.default_rng(seed)
    df = _trials(300, seed=10 + seed)
    pars = {"sd_ang": rng.uniform(0.02, 0.4), "sd_inf": rng.uniform(0.1, 1.5), "sd_sel": rng.uniform(0.1, 3.0),
            **{f"motiv_{k}": rng.uniform(-0.5, 0.5) for k in SOURCES},
            "choice_bias": rng.uniform(-0.5, 0.5), "plapse_choice": rng.uniform(0.001, 0.05)}
    fast = M.get_ppost_fast(pars, M.df_to_pack(df))[0]
    p_observed = np.where(df["choice"] == 1, fast, 1 - fast)
    prereg = P.psychofun(_prereg_pars(pars), P.df_to_pack(df))[0]
    np.testing.assert_allclose(prereg, p_observed, rtol=1e-12)
    names = list(pars)
    ll_fast = M.get_ll_fast([pars[k] for k in names], names, M.df_to_pack(df))
    pre = _prereg_pars(pars)
    ll_prereg = P.get_ll([pre[k] for k in pre], list(pre), P.df_to_pack(df))
    assert ll_fast == pytest.approx(ll_prereg, rel=1e-12)


class _RefusedPool:
    def __init__(self, *args, **kwargs):
        raise PermissionError("process pool refused (test)")


def test_bads_fit_does_not_depend_on_the_worker_processes(monkeypatch) -> None:
    """Each restart reseeds the random generators, so the fit is the same with one worker per restart, with restarts
    sharing a worker, and with no pool (``run_fit_categ`` then runs them one after the other in this process, with the
    same inputs: ``sd_sel`` has integer bounds, which BADS treats differently from float ones); in every case the
    generators of this process are left as the pool leaves them, so the next fit draws the same starts."""
    pytest.importorskip("pybads")
    df = _trials(200, seed=3)
    np.random.seed(0)
    df["choice"] = M.simulate_choice({"sd_sel": 2.0}, df)[0]
    model = M.create_categ_model(["sd_sel"])

    def fit() -> tuple:
        np.random.seed(42)
        res = M.run_fit_categ(df, model, bads_nit=3)
        return float(res["params"][0]), float(res["ll"]), np.random.rand()

    one_per_restart = fit()
    monkeypatch.setattr(M, "BADS_WORKERS", 1)
    shared_worker = fit()
    monkeypatch.setattr(M, "ProcessPoolExecutor", _RefusedPool)
    no_pool = fit()
    assert shared_worker == one_per_restart
    assert no_pool == one_per_restart
