"""Effect of each deviation from the frozen pre-registered code on every pre-registered number.

Three versions of each number, per experiment:

- frozen: the code frozen at pre-registration, on the fits as registered (all 10 BADS restarts started from the
  model's x0; ``results/prereg/<exp>/model_fit/as_registered/``), with the frozen p-value rule (pingouin: in one-way
  ANOVAs the Greenhouse-Geisser correction only when Mauchly's test rejects sphericity; pingouin's epsilon for the
  two-way interaction);
- deviation 1: the same code and rule, on the refitted fits (random starting points);
- reported: deviation 1 plus deviation 2 (GG-corrected p with the exact epsilon for every ANOVA): the paper.

The numbers are computed with the pre-registered code's own functions (``motinf.prereg.behavior`` and
``scripts/prereg/run_model_stats.py``). The reported column is checked against ``paper_tables.json``.

Usage (repository root, data bundle unpacked):

    python checks/prereg_deviations.py           # rewrite the table in PREREG_DEVIATIONS.md and prereg_deviations.csv
    python checks/prereg_deviations.py --check   # exit code 1 if either is out of date (skipped without the bundle)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg

ROOT = Path(__file__).resolve().parents[1]

from motinf.exploratory.paper_format import p_value
from motinf.prereg import behavior as B
from motinf.prereg.rm_anova_exact import rm_anova_exact

DOC = ROOT / "PREREG_DEVIATIONS.md"
CSV = ROOT / "checks" / "prereg_deviations.csv"
START, END = "<!-- table:start (checks/prereg_deviations.py) -->", "<!-- table:end -->"
EXPS = {"exp1": "Experiment 1 (discovery)", "exp2": "Experiment 2 (pre-registered)"}
MINUS = "−"


def _load_model_stats():
    spec = importlib.util.spec_from_file_location("prereg_run_model_stats", ROOT / "scripts/prereg/run_model_stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MS = _load_model_stats()


# ------------------------------------------------------------------ p-value rules
def _norm(table: pd.DataFrame) -> pd.DataFrame:
    return table.rename(columns=lambda c: c.replace("-", "_")).set_index("Source")


def frozen_anova(table: pd.DataFrame, source: str) -> dict:
    """pingouin's row with the frozen rule: GG p when pingouin reports one, else uncorrected."""
    r = _norm(table).loc[source]
    df1 = r["ddof1"] if "ddof1" in r.index else r["DF"]
    p = r["p_GG_corr"] if "p_GG_corr" in r.index and pd.notna(r["p_GG_corr"]) else r["p_unc"]
    return {"F": float(r["F"]), "df1": int(df1), "p": float(p)}


def exact_anova(data: pd.DataFrame, dv: str, within, source: str) -> dict:
    r = rm_anova_exact(data, dv, within).set_index("Source").loc[source]
    return {"F": float(r["F"]), "df1": int(r["ddof1"]), "df2": int(r["ddof2"]), "p": float(r["p_GG_corr"])}


def ttest(table: pd.DataFrame) -> dict:
    r = table.rename(columns=lambda c: c.replace("-", "_")).iloc[0]
    return {"t": float(r["T"]), "df": int(round(float(r["dof"]))), "p": float(r["p_val"])}


# ------------------------------------------------------------------ the numbers
def behaviour(exp: str) -> dict:
    """H1-H3 and the Stimulus Reproduction tests: no fits involved, so frozen == deviation 1."""
    base = ROOT / "data" / "processed" / exp
    df_inf, df_est = pd.read_csv(base / "data_clean_inf.csv"), pd.read_csv(base / "data_clean_est.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v, vl = B._build_choice_variability_tables(df_inf)
        df_m, df_m_len = B._build_mean_tables(df_inf, df_est, v, vl)
        an = B._run_analyses(df_m, df_m_len)
    n = df_m["participant"].nunique()
    out = {}
    for key, table, data, dv, within, source in (
        ("H1", "rm_anova_sensitivity_incentive", df_m, "sensitivity", ["incentive"], "incentive"),
        ("H2_length", "rm_anova_choice_var_log_seqlen_x_incentive", df_m_len, "choice_var_log", ["seqlen", "incentive"], "seqlen"),
        ("H2_incentive", "rm_anova_choice_var_log_seqlen_x_incentive", df_m_len, "choice_var_log", ["seqlen", "incentive"], "incentive"),
        ("H2_interaction", "rm_anova_choice_var_log_seqlen_x_incentive", df_m_len, "choice_var_log", ["seqlen", "incentive"], "seqlen * incentive"),
        ("repro_anova", "rm_anova_estim_sd_incentive", df_m, "estim_sd", ["incentive"], "incentive"),
    ):
        fr, ex = frozen_anova(an[table], source), exact_anova(data, dv, within, source)
        fr["df2"] = ex["df2"]
        out[key] = {"kind": "F", "frozen": fr, "dev1": fr, "reported": ex}
    for key, table in (("H3", "ttest_sensitivity_abs_1_gt_0"), ("repro_ttest", "ttest_estim_sd_abs_1_vs_0")):
        t = ttest(an[table])
        out[key] = {"kind": "t", "frozen": t, "dev1": t, "reported": t}
    out["_n"] = n
    return out


def model_numbers(npz_dir: Path) -> dict:
    """H4 and the two group BMCs, computed by run_model_stats.py's functions on one set of fits."""
    per_cond = MS._npz_to_dict(np.load(npz_dir / "fit_noise_per_cond.npz", allow_pickle=True))
    df_sd = MS._extract_sd_inf_long(per_cond)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = pg.rm_anova(data=df_sd, dv="sd_inf_log", within=["incentive"], subject="participant", detailed=True,
                        effsize="np2")
    fr = frozen_anova(a, "incentive")
    ex = exact_anova(df_sd, "sd_inf_log", "incentive", "incentive")
    fr["df2"] = ex["df2"]
    out = {"H4A_frozen_rule": fr, "H4A_exact": ex, "H4B": ttest(MS._paired_ttest_log_sd_inf_abs_1_vs_0(df_sd))}

    names = MS._extract_model_names_from_specs(per_cond["categ_model_specs"])
    parts = np.asarray(per_cond["participants"])
    freq, pxp = [], []
    for ji, _ in enumerate(np.asarray(per_cond["incentives"], dtype=float)):  # loss, neutral, gain
        mat, _, _ = MS._drop_incomplete_subjects(MS._build_per_condition_aicc_matrix_for_incentive(per_cond, names, ji), parts)
        m = MS._extract_groupbmc_metrics(MS._run_group_bmc(-0.5 * mat), len(names))
        freq.append(float(m["frequency_mean"][names.index("inf")]))
        pxp.append(float(m["pxp"][names.index("inf")]))
    out["bmc_single_noise"] = {"freq": freq, "pxp": pxp}

    inc = MS._npz_to_dict(np.load(npz_dir / "fit_inc_effects.npz", allow_pickle=True))
    names_ie = MS._extract_model_names_from_specs(inc["categ_model_specs"])
    mat, _, _ = MS._drop_incomplete_subjects(np.asarray(inc["aicc_categ"], dtype=float), np.asarray(inc["participants"]))
    m = MS._extract_groupbmc_metrics(MS._run_group_bmc(-0.5 * mat), len(names_ie))
    i = names_ie.index("inf_v")
    out["bmc_single_modulation"] = {"freq": [float(m["frequency_mean"][i])], "pxp": [float(m["pxp"][i])]}
    return out


def collect() -> dict:
    res = {}
    for exp in EXPS:
        beh = behaviour(exp)
        mf = ROOT / "results" / "prereg" / exp / "model_fit"
        old, new = model_numbers(mf / "as_registered"), model_numbers(mf)
        rows = {k: v for k, v in beh.items() if not k.startswith("_")}
        rows["H4A"] = {"kind": "F", "frozen": old["H4A_frozen_rule"], "dev1": new["H4A_frozen_rule"], "reported": new["H4A_exact"]}
        rows["H4B"] = {"kind": "t", "frozen": old["H4B"], "dev1": new["H4B"], "reported": new["H4B"]}
        for k in ("bmc_single_noise", "bmc_single_modulation"):
            rows[k] = {"kind": "bmc", "frozen": old[k], "dev1": new[k], "reported": new[k]}
        res[exp] = rows
    return res


# ------------------------------------------------------------------ checks against the paper's numbers
def check_against_paper(res: dict) -> None:
    pt = json.loads((ROOT / "results/exploratory/merged/paper_tables/paper_tables.json").read_text())
    for exp, rows in res.items():
        t1 = pt["table1"][exp]
        pairs = [
            (rows["H1"]["reported"]["p"], t1["sensitivity"]["omnibus"]["p_gg"]),
            (rows["H1"]["reported"]["F"], t1["sensitivity"]["omnibus"]["F"]),
            (rows["H2_interaction"]["reported"]["p"], t1["choice_var"]["seqlen_x_incentive"]["p_gg"]),
            (rows["H2_length"]["reported"]["F"], t1["choice_var"]["seqlen"]["F"]),
            (rows["H2_incentive"]["reported"]["p"], t1["choice_var"]["incentive"]["p_gg"]),
            (rows["H3"]["reported"]["t"], t1["sensitivity"]["abs"]["t"]),
            (rows["H3"]["reported"]["p"], t1["sensitivity"]["abs"]["p"]),
            (rows["H4A"]["reported"]["p"], t1["log_sd_inf"]["omnibus"]["p_gg"]),
            (rows["H4B"]["reported"]["t"], t1["log_sd_inf"]["abs"]["t"]),
            (rows["H4B"]["reported"]["p"], t1["log_sd_inf"]["abs"]["p"]),
            (rows["repro_anova"]["reported"]["p"], t1["estim_sd"]["omnibus"]["p_gg"]),
            (rows["repro_ttest"]["reported"]["p"], t1["estim_sd"]["abs"]["p"]),
            (rows["bmc_single_modulation"]["reported"]["freq"][0], pt["bms"][exp]["mod_freq"]),
        ]
        for a, b in pairs:
            assert abs(a - b) <= 1e-9 * max(1.0, abs(b)), f"{exp}: reported {a} != paper_tables {b}"


# ------------------------------------------------------------------ formatting
def fp(p: float) -> str:
    """The paper's rule (``paper_format.p_value``) with its relation: ``< 0.001`` or ``= 0.073``."""
    s = p_value(p)
    return s if s.startswith("<") else "= " + s


def fnum(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace("-", MINUS)


def fmt(kind: str, v: dict) -> str:
    if kind == "F":
        return f"F({v['df1']}, {v['df2']}) = {fnum(v['F'])}, *P* {fp(v['p'])}"
    if kind == "t":
        return f"t({v['df']}) = {fnum(v['t'])}, *P* {fp(v['p'])}"
    freq = " / ".join(f"{100 * f:.0f}%" for f in v["freq"])
    lo = min(v["pxp"])
    return f"{freq}; PXP {'> 0.99' if lo > .99 else fp(lo).replace('= ', '')}"


LABELS = [
    ("H1", "H1: evidence sensitivity, incentive"),
    ("H2_length", "H2 ANOVA: log choice variability, sequence length"),
    ("H2_incentive", "H2 ANOVA: log choice variability, incentive"),
    ("H2_interaction", "H2: log choice variability, length × incentive"),
    ("H3", "H3: evidence sensitivity, incentive vs neutral (one-tailed)"),
    ("H4A", "H4A: log σinf, incentive"),
    ("H4B", "H4B: log σinf, incentive vs neutral (one-tailed)"),
    ("repro_anova", "Reproduction error SD, incentive (pre-specified)"),
    ("repro_ttest", "Reproduction error SD, incentive vs neutral (one-tailed, pre-specified)"),
    ("bmc_single_noise", "Single-noise BMC: inference-model frequency, loss / neutral / gain (pre-specified)"),
    ("bmc_single_modulation", "Single-modulation BMC: inference-modulation frequency (pre-specified)"),
]


def table_md(res: dict) -> str:
    lines = ["| Test | Frozen code, fits as registered | Deviation 1: random starts | Deviations 1 + 2 (reported) |",
             "|---|---|---|---|"]
    for exp, title in EXPS.items():
        lines.append(f"| **{title}** | | | |")
        for key, label in LABELS:
            r = res[exp][key]
            cells = [fmt(r["kind"], r[c]) for c in ("frozen", "dev1", "reported")]
            shown = [cells[0]] + [f"**{c}**" if c != cells[0] else c for c in cells[1:]]
            lines.append(f"| {label} | " + " | ".join(shown) + " |")
    return "\n".join(lines)


def table_csv(res: dict) -> pd.DataFrame:
    rows = []
    for exp in EXPS:
        for key, _ in LABELS:
            r = res[exp][key]
            stats = ("freq", "pxp") if r["kind"] == "bmc" else (("F", "p") if r["kind"] == "F" else ("t", "p"))
            for s in stats:
                vals = [r[c][s] for c in ("frozen", "dev1", "reported")]
                if r["kind"] == "bmc":
                    for j in range(len(vals[0])):
                        rows.append({"experiment": exp, "test": key, "statistic": f"{s}[{j}]",
                                     "frozen": vals[0][j], "deviation_1": vals[1][j], "reported": vals[2][j]})
                else:
                    rows.append({"experiment": exp, "test": key, "statistic": s,
                                 "frozen": vals[0], "deviation_1": vals[1], "reported": vals[2]})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="only check that the table and the CSV are current")
    args = ap.parse_args()

    needed = [ROOT / "data" / "processed" / e / "data_clean_inf.csv" for e in EXPS] + [
        ROOT / "results" / "prereg" / e / "model_fit" / d / "fit_noise_per_cond.npz"
        for e in EXPS for d in ("as_registered", ".")]
    if args.check and not all(p.is_file() for p in needed):
        print("PREREG_DEVIATIONS.md: not checked (needs the trial data and the fits of the data bundle)")
        return 0
    res = collect()
    check_against_paper(res)
    doc = DOC.read_text(encoding="utf-8")
    a, b = doc.index(START) + len(START), doc.index(END)
    new_doc = doc[:a] + "\n" + table_md(res) + "\n" + doc[b:]
    csv_text = table_csv(res).to_csv(index=False, float_format="%.10g", lineterminator="\n")
    if args.check:
        ok = new_doc == doc and CSV.exists() and CSV.read_text(encoding="utf-8") == csv_text
        print("PREREG_DEVIATIONS.md and checks/prereg_deviations.csv are " + ("current" if ok else "OUT OF DATE"))
        return 0 if ok else 1
    DOC.write_text(new_doc, encoding="utf-8", newline="\n")
    CSV.write_text(csv_text, encoding="utf-8", newline="\n")
    print(table_md(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
