"""Integration-task (categorization) model used by the exploratory fits.

Parameter bounds (``param_config_categ``), the posterior over the category given the noisy sample sequence
(``get_ppost_fast``), log-likelihood and log-posterior, choice simulation, and BADS / VBMC fitting
(``fit_model_BADS``, ``fit_model_VBMC``), and the helpers the fit scripts share (``create_categ_model``,
``run_fit_categ``, ``trials_ok``, ``subset_priors``). Noise sources: estimation (``sd_ang``, on each sample angle),
inference (``sd_inf``, added per accumulated sample) and selection (``sd_sel``, at the choice); under
incentive each is scaled by ``exp(-motiv_X)``. ``fit_metrics`` gives the AIC, AICc and BIC of a fit.

:mod:`motinf.prereg.model` is the frozen pre-registered copy of the same likelihood, with separate incentive SDs
(``sd_X_v``); ``tests/test_model.py`` checks that the two give the same likelihood.
"""

from __future__ import annotations

import copy
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import gamma, norm
from threadpoolctl import threadpool_limits

try:
    from pybads import BADS
except ImportError:  # optional: only needed for BADS optimizers
    BADS = None  # type: ignore[misc, assignment]
try:
    from pyvbmc import VBMC
except ImportError:  # optional: only needed for VBMC optimizers
    VBMC = None  # type: ignore[misc, assignment]


def _require_bads() -> None:
    if BADS is None:
        raise ImportError(
            "The optional dependency `pybads` is required for BADS optimization "
            "but is not installed."
        )


def _require_vbmc() -> None:
    if VBMC is None:
        raise ImportError(
            "The optional dependency `pyvbmc` is required for VBMC optimization "
            "but is not installed."
        )


KAPPA = 0.5  # concentration of the two orientation distributions of the task (von Mises)
SIGNAL_STRENGTH = 1  # scale of the log-likelihood ratio carried by one sample
BADS_WORKERS = 10  # processes that run the restarts of one BADS fit; each restart reseeds, so fits do not depend on it

priors_categ = {
    'sd_ang': gamma(a=1.1, scale=np.deg2rad(2)),
    'sd_ang_v': gamma(a=1.1, scale=np.deg2rad(2)),
    'motiv_ang': norm(loc=0, scale=0.5),
    'sd_inf': gamma(a=2, scale=0.5),
    'sd_inf_v': gamma(a=2, scale=0.5),
    'motiv_inf': norm(loc=0, scale=0.5),
    'sd_sel': gamma(a=1.1, scale=2), #1.38 is equivalent sd_inf of 0.5. total_sd = (sqrt(4*0.5**2) + sqrt(8 * 0.5**2) + sqrt(12 * 0.5**2))/3 = 1.38
    'sd_sel_v': gamma(a=1.1, scale=2),
    'motiv_sel': norm(loc=0, scale=0.5),
}
    
param_config_categ = {
    'sd_ang': {
        'lb': np.deg2rad(0.001), 'ub': np.deg2rad(50),
        'plb': np.deg2rad(0.1), 'pub': np.deg2rad(15),
        'x0': np.deg2rad(4)
    },

    'sd_ang_v': {
        'lb': np.deg2rad(0.001), 'ub': np.deg2rad(50),
        'plb': np.deg2rad(0.1), 'pub': np.deg2rad(15),
        'x0': np.deg2rad(4)
    },

    'motiv_ang': {
        'lb': -3, 'ub': 3,
        'plb': -0.5, 'pub': 0.5,
        'x0': np.deg2rad(0)
    },

    'sd_inf': {
        'lb':1e-6,'ub':5,
       'plb':0.1,'pub':1.5,
       'x0':0.5
       },
    'sd_inf_v': {
        'lb':1e-6,'ub':5,
       'plb':0.1,'pub':1.5,
       'x0':0.5
       },
    'sd_inf_g': {
        'lb':1e-6,'ub':5,
       'plb':0.1,'pub':1.5,
       'x0':0.5
       },
    'sd_inf_l': {
        'lb':1e-6,'ub':5,
       'plb':0.1,'pub':1.5,
       'x0':0.5
       },
       
    'sd_sel':{'lb':1e-6,'ub':10,
            'plb':0.1,'pub':4,
            'x0':1
            },
    'sd_sel_v': {
        'lb':1e-6,'ub':10,
       'plb':0.1,'pub':4,
       'x0':1
       },
    'motiv_inf': {
        'lb':-3,'ub':3,
       'plb':-.5,'pub':.5,
       'x0':0
       },

    'motiv_sel': {
        'lb':-3,'ub':3,
       'plb':-.5,'pub':.5,
       'x0':0
       },

    'plapse_choice': {
        'lb': 1e-10, 'ub': 0.1,
        'plb': 0.005, 'pub': 0.05,
        'x0': 0.01
    },
    'choice_bias': {
        'lb': -5, 'ub': 5,
        'plb': -1, 'pub': 1,
        'x0': 0
    }
}


def get_bounds(param_name, bounds_config):
    """Return (lb, ub) for param_name from bounds_config or (None, None)."""
    if bounds_config is None or param_name not in bounds_config:
        return None, None
    info = bounds_config[param_name]
    return info.get("lb", None), info.get("ub", None)


def clip_param(value, param_name, bounds_config):
    """Clip scalar/array value to parameter bounds if available."""
    lb, ub = get_bounds(param_name, bounds_config)
    if lb is None or ub is None:
        return value
    return np.clip(value, lb, ub)
    
def df_to_pack(df):
    """
    Convert a participant's dataframe to a dict ready for get_ppost().
    Expected columns: seqlen, choice, incentive (optional),
    seqang_noisy_* (or seqang_*) for angles.
    """
    # --- core vectors ---
    seqlen = df["seqlen"].to_numpy(int, copy=False)
    

    choice = df["choice"].to_numpy(copy=False) if "choice" in df else None
        
    incentive = df["incentive"].to_numpy(copy=False) if "incentive" in df else None

    # --- detect and stack sequence columns ---
    seqang_cols = ['seqang_noisy_%d'%i for i in range(1,13)]
    seqang = np.column_stack([df[c].to_numpy(copy=False) for c in seqang_cols])

    # --- mask positions beyond actual seqlen with NaN ---
    N, T = seqang.shape
    cols = np.arange(T)[None, :]
    valid = cols < seqlen[:, None]
    seqang[~valid] = np.nan

    # --- return ready-to-use dict ---
    return {
        "seqang": seqang,       # 2D array (N×T)
        "seqlen": seqlen,       # 1D array (N,)
        "choice": choice,       # 1D array (N,)
        "incentive": incentive, # 1D array or None
        "N": N,
        "T": T,
    }
    
def get_ppost_fast(pars, data, kappa=KAPPA, sigstr=SIGNAL_STRENGTH, bounds_config=param_config_categ):
    """P(choice = 1), mean (``xpost``) and SD (``spost``) of the decision variable, per trial.

    ``pars`` holds the noise SDs ``sd_ang``, ``sd_inf`` and ``sd_sel`` (missing: 0) and, optionally,
    ``choice_bias`` and ``plapse_choice``. Under incentive each SD is ``sd_X * exp(-motiv_X)``, unless given as
    ``sd_X_v`` (gains and losses) or ``sd_X_g`` / ``sd_X_l`` (gains / losses only). The SDs and the lapse rate are
    clipped to their bounds in ``bounds_config``.
    """
    incentive = data['incentive']
    sd = {}
    for src in ('ang', 'inf', 'sel'):
        name = f'sd_{src}'
        base = clip_param(pars.get(name, 0.0), name, bounds_config)
        incent = clip_param(pars.get(f'{name}_v', base * np.exp(-pars.get(f'motiv_{src}', 0.0))), name, bounds_config)
        gain = clip_param(pars.get(f'{name}_g', incent), name, bounds_config)
        loss = clip_param(pars.get(f'{name}_l', incent), name, bounds_config)
        sd[src] = np.where(incentive == 1, gain, np.where(incentive == -1, loss, base))
    plapse = clip_param(pars.get('plapse_choice', 0.0), 'plapse_choice', bounds_config)
    choice_bias = pars.get('choice_bias', 0)

    seqang = data['seqang']
    sd_ang_vec = sd['ang'][:, None]
    mu_llr = 2 * kappa * sigstr * np.exp(-2 * sd_ang_vec ** 2) * np.cos(2 * (seqang - np.pi * 3 / 4))
    s2_llr = 2 * (kappa**2) * (sigstr**2) * (1 + np.exp(-8 * sd_ang_vec**2) * np.cos(4 * (seqang - np.pi * 3 / 4)))

    diff = np.clip(s2_llr - mu_llr**2, 0, None)
    sd_llr = np.sqrt(diff)

    mu_llr[np.isnan(mu_llr)] = 0
    sd_llr[np.isnan(sd_llr)] = 0

    xpost = np.nansum(mu_llr, axis=1)
    spost = np.sqrt(np.nansum(sd_llr**2, axis=1) + (data['seqlen'] - 1) * (sd['inf'] ** 2) + (sd['sel'] ** 2))
    spost = np.clip(spost, 1e-20, np.inf)

    ppost = norm.cdf(xpost+choice_bias, 0, spost)
    ppost = (1 - plapse) * ppost + plapse / 2

    if np.isnan(ppost).any():
        raise ValueError("NAN FOUND")

    return ppost, xpost, spost

def get_ll_fast(theta, parnames, data, fixedpars={}, kappa=KAPPA, sigstr=SIGNAL_STRENGTH,
                bounds_config=param_config_categ):
    pars = {'sd_ang': 0, 'sd_inf': 0, 'sd_sel': 0}
    pars.update(fixedpars)
    for i, name in enumerate(parnames):
        pars[name] = theta[i]
    
    ppost, _, _ = get_ppost_fast(pars, data, kappa, sigstr, bounds_config=bounds_config)
    choice = data['choice']
    ppostcw = np.where(choice == 1, ppost, 1 - ppost)
    ppostcw = np.clip(ppostcw, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return np.nansum(np.log(ppostcw))

def get_log_posterior_fast(theta, parnames, data, fixedpars={}, priors={}, kappa=KAPPA, sigstr=SIGNAL_STRENGTH,
                           bounds_config=param_config_categ):
    ll = get_ll_fast(theta, parnames, data, fixedpars, kappa, sigstr, bounds_config=bounds_config)
    log_priors = 0.0
    for i, name in enumerate(parnames):
        if name in priors:
            log_priors += priors[name].logpdf(theta[i])
    return ll + log_priors
     
def simulate_choice(params, df, kappa=KAPPA):
    """Simulated choices (+1 / -1) on the trials of ``df`` for the parameters ``params`` (missing noise SDs: 0), from
    NumPy's global random stream; also returns ``xpost``, ``spost`` and ``ppost`` of :func:`get_ppost_fast`."""

    pars = {
    'sd_ang': 0,
    'sd_inf': 0,
    'sd_sel': 0,
    }

    pars.update(params)

    data = df_to_pack(df)
    ppost,xpost,spost = get_ppost_fast(pars,data,kappa=kappa)

    rand = np.random.rand(len(ppost))
    choice = np.zeros(len(ppost))*np.nan
    choice[rand<ppost]=1
    choice[rand>=ppost]=-1

    assert not np.any(np.isnan(choice))
    return choice,xpost,spost,ppost


def run_bads(opt_fun, theta0, lb, ub, plb, pub, options):
    """
    General function for running BADS, to be used in paralellization
    """
    _require_bads()
    bads = BADS(opt_fun, theta0, lb, ub, plb, pub, options=options)
    res_i = bads.optimize()
    return res_i

def _run_bads_with_limits(theta0_i, lb, ub, plb, pub, options,
                          parnames, data, fixedpars, priors, kappa=KAPPA, sigstr=SIGNAL_STRENGTH, worker_id=None):
    """One BADS run from ``theta0_i`` on the negative log-posterior, single-threaded (for a process pool); restart
    ``worker_id`` seeds NumPy's and Python's random generators with 10000 + ``worker_id``."""
    if worker_id is not None:
        seed = 10000 + worker_id
        np.random.seed(seed)
        random.seed(seed)
        
    def obj(theta):
        # use your existing (pandas-based) function
        return -get_log_posterior_fast(theta, parnames, data, fixedpars, priors,
                                  kappa=kappa, sigstr=sigstr)

    with threadpool_limits(limits=1):
        # Call your existing run_bads (whatever you already have)
        res = run_bads(obj, theta0_i, lb, ub, plb, pub, options)

    return {
            'x': np.asarray(res['x']),             # NumPy arrays are picklable
            'fval': float(res['fval']),
        }    
     

def _starting_points(x0, nit, plb, pub):
    """The ``nit`` BADS starting points: drawn uniformly in [``plb``, ``pub``] when ``x0`` is None, ``x0`` for every
    restart when it is one vector, or one row of ``x0`` per restart."""
    if x0 is None:
        return [np.random.uniform(low=plb, high=pub) for _ in range(nit)]
    if np.ndim(x0) < 2:
        return [x0 for _ in range(nit)]
    if len(x0) != nit:
        raise ValueError(f"x0 has {len(x0)} rows (starting points) for {nit} restarts.")
    return [np.asarray(row, dtype=float) for row in x0]


def fit_metrics(ll, k, n):
    """AIC, AICc and BIC of a fit with log-likelihood ``ll``, ``k`` free parameters and ``n`` trials (AICc and BIC
    are NaN when ``n <= k + 1``)."""
    aic = 2 * k - 2 * ll
    if n - k - 1 <= 0:
        return aic, np.nan, np.nan
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    bic = -2 * ll + k * np.log(n)
    return aic, aicc, bic


def fit_model_BADS(df,parnames,lb,ub,plb=None,pub=None,x0=None,fixedpars={},priors={},
                   nit = 1,
                   options = {'uncertainty_handling':False,'display':'off'},
                   kappa=KAPPA, in_process=False):
    """BADS fit with ``nit`` restarts, each of which reseeds the random generators with its index
    (:func:`_run_bads_with_limits`): in a pool of up to ``BADS_WORKERS`` processes, or, with ``in_process``, one after
    the other in this process (where a pool is refused). Both give the same fit and leave this process's random state as
    they found it after the starting points were drawn."""
    if plb is None:
        plb=lb+(ub-lb)*0
    if pub is None:
        pub=lb+(ub-lb)*0.8

    ntrials = len(df)
    theta0 = _starting_points(x0, nit, plb, pub)

    nparams = len(lb)
    # Run optimization
    
    data = df_to_pack(df)
   
    res_list = []
    res_ll = []
    if in_process:
        # As in a worker: each restart gets its own copy of the inputs, as given (BADS treats an integer bound
        # differently from a float one), and reseeds; this process's generators are then restored, as a pool leaves
        # them untouched.
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            for i in range(nit):
                try:
                    res_i = _run_bads_with_limits(*copy.deepcopy((theta0[i], lb, ub, plb, pub, options, parnames,
                                                                  data, fixedpars, priors, kappa)), worker_id=i)
                    res_list.append(res_i)
                    res_ll.append(res_i['fval'])
                    print(f"Results for iteration {i}: {res_i['x'].round(4)}")
                except Exception as e:
                    print(f"Optimization iteration {i} failed: {e}")
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
    else:
        n_workers = min(nit, BADS_WORKERS)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(
                    _run_bads_with_limits,
                    theta0[i], lb, ub, plb, pub, options,
                    parnames, data, fixedpars, priors, kappa, worker_id = i
                ): i
                for i in range(nit)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    res_i = future.result()
                    res_list.append(res_i)
                    res_ll.append(res_i['fval'])
                    print(f"Results for iteration {i}: {res_i['x'].round(4)}")
                except Exception as e:
                    print(f"Optimization iteration {i} failed: {e}")

    if res_ll:
        res = res_list[np.argmin(res_ll)]#
    else:
        print('OPTIMIZATION FAILED FOR ALL ITERATIONS')
        res = {'x': np.zeros(nparams)*np.nan, 'fval': np.nan}

    print('Returned parameter vector:', np.round(res['x'],4))
    print('Negative log-likelihood at solution:', np.round(res['fval'],4))

    params_final = res['x']
    # Recompute plain log-likelihood
    ll = get_ll_fast(params_final, parnames, data, fixedpars, kappa=kappa)
    _, aicc, bic = fit_metrics(ll, nparams, ntrials)

    return {
        'params': params_final,
        'll': ll,
        'aicc': aicc,
        'bic': bic,
    }
    

def fit_model_VBMC(df,parnames,lb,ub,plb=None,pub=None,x0=None,fixedpars={},priors = {},kappa=KAPPA):

    data = df_to_pack(df)
    opt_fun = lambda theta_: get_log_posterior_fast(theta_, parnames, data,fixedpars, priors,kappa = kappa)

    if plb is None:
        plb = lb+(ub-lb)*0.1
    if pub is None:
        pub = lb+(ub-lb)*0.9

    if x0 is None:
        x0 = np.random.uniform(low=plb,high=pub)

    x0 = np.array(x0)

    _require_vbmc()
    vbmc = VBMC(opt_fun, x0, lb, ub, plb, pub)
    vp, results = vbmc.optimize()
    
    Xs = vp.sample(1000000)[0]
    pars_mean = Xs.mean(0)
    pars_std = Xs.std(0)
    pars_ci95 = np.quantile(Xs,[0.025,0.975],axis=0)

    return {
        'params': pars_mean,
        'params_sd': pars_std,
        'params_ci95': pars_ci95,
        'elbo': results['elbo'],
        }


# --------------------------------------------------------------------------------------------------------------- #
# Helpers shared by the fit scripts
# --------------------------------------------------------------------------------------------------------------- #


def trials_ok(n_trials: int, n_params: int) -> bool:
    """Enough trials for a finite AICc (``n_trials - n_params - 1 > 0``)."""
    return n_trials >= n_params + 2


def subset_priors(prior_map: dict, parnames: list) -> dict:
    """The priors of ``parnames`` that ``prior_map`` defines."""
    return {k: prior_map[k] for k in parnames if k in prior_map}


def create_categ_model(parnames: list) -> dict:
    """Bounds (``lb``, ``ub``), plausible bounds (``plb``, ``pub``) and starting point (``x0``) of the
    categorization parameters ``parnames``, from ``param_config_categ``."""
    lb, ub, plb, pub, x0 = [], [], [], [], []
    for param in parnames:
        if param not in param_config_categ:
            raise ValueError(f"Categorization parameter '{param}' not in param_config_categ.")
        cfg = param_config_categ[param]
        lb.append(cfg["lb"])
        ub.append(cfg["ub"])
        plb.append(cfg["plb"])
        pub.append(cfg["pub"])
        x0.append(cfg["x0"])
    return {"parnames": parnames, "lb": lb, "ub": ub, "plb": plb, "pub": pub, "x0": x0}


def fit_model_BADS_serial(df, parnames, lb, ub, plb=None, pub=None, x0=None, fixedpars=None, priors=None, nit=1,
                          options=None, kappa=KAPPA):
    """As :func:`fit_model_BADS`, with the ``nit`` restarts run one after the other in this process, for a fit inside a
    pool's worker (no nested pool). A restart whose objective is NaN is never the one kept. The bounds go to BADS as
    float arrays, so a model with an integer bound (``sd_sel``) can come out slightly differently than from
    :func:`fit_model_BADS`, which passes them as given."""
    if fixedpars is None:
        fixedpars = {}
    if priors is None:
        priors = {}
    if options is None:
        options = {"uncertainty_handling": False, "display": "off"}

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    plb = lb + (ub - lb) * 0 if plb is None else np.asarray(plb, dtype=float)
    pub = lb + (ub - lb) * 0.8 if pub is None else np.asarray(pub, dtype=float)

    ntrials = len(df)
    data = df_to_pack(df)
    theta0 = _starting_points(x0, nit, plb, pub)

    res_list = []
    res_ll = []
    for i in range(nit):
        res_i = _run_bads_with_limits(theta0_i=theta0[i], lb=lb, ub=ub, plb=plb, pub=pub, options=options,
                                      parnames=parnames, data=data, fixedpars=fixedpars, priors=priors, kappa=kappa,
                                      worker_id=i)
        res_list.append(res_i)
        res_ll.append(res_i["fval"])

    res = res_list[np.nanargmin(res_ll)]
    params_final = res["x"]
    ll = get_ll_fast(params_final, parnames, data, fixedpars=fixedpars, kappa=kappa)
    _, aicc, bic = fit_metrics(ll, len(lb), ntrials)
    return {"params": params_final, "ll": ll, "aicc": aicc, "bic": bic}


def run_fit_categ(df, model: dict, fixedpars: dict | None = None, priors: dict | None = None, bads_nit: int = 1):
    """BADS fit of ``model`` (:func:`create_categ_model`) to the categorization trials ``df``: ``bads_nit`` restarts in
    a process pool (:func:`fit_model_BADS`), or one after the other where the pool is refused."""
    kwargs = dict(df=df, parnames=model["parnames"], lb=model["lb"], ub=model["ub"], plb=model["plb"],
                  pub=model["pub"], fixedpars={} if fixedpars is None else fixedpars,
                  priors={} if priors is None else priors, nit=bads_nit,
                  options={"uncertainty_handling": False, "display": "off"}, kappa=0.5)
    state = np.random.get_state()
    try:
        return fit_model_BADS(**kwargs)
    except PermissionError:
        np.random.set_state(state)  # draw the same starting points again, for the restarts run one after the other
        return fit_model_BADS(**kwargs, in_process=True)
