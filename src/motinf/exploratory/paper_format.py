"""How the paper prints numbers: one function per rule (``checks/prereg_deviations.py`` uses :func:`p_value`).

Each rule takes the source values of a printed number (a list, in the order of its sources) and returns the string
the paper should print. A rule is written ``name`` or ``name(arg, ...)``; the arguments are strings (decimals, a
threshold, a unit).

Conventions (Nature Human Behaviour's house style): a leading zero on every number below one, a true minus sign
(U+2212) and no negative zero; p values with three decimals, ``< 0.001`` below 0.001; effect sizes, F and t with two
decimals; Bayes factors as BF10 or BF01 = 1/BF10, whichever is at least 1, with one decimal below 10, none from 10 to
999, and as ``m.m×10ⁿ`` from 1,000; stars from the p value given (Holm-corrected where the table says so):
* < 0.05, ** < 0.01, *** < 0.001.

:func:`normalize` removes what only differs in typography (hyphen-minus vs minus sign, spaces), so a check can tell
a different value from a different spelling of the same value.
"""

from __future__ import annotations

import math
import re
from typing import Callable

MINUS = "−"
SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty")


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def num(v: float, nd: int = 2) -> str:
    """Fixed decimals, true minus, no negative zero: ``num(-0.001)`` -> ``"0.00"``."""
    s = f"{v:.{nd}f}"
    if float(s) == 0:
        s = f"{0:.{nd}f}"
    return s.replace("-", MINUS)


def signed(v: float, nd: int = 0) -> str:
    """Like :func:`num`, with ``+`` before a positive value."""
    s = num(v, nd)
    return s if s.startswith(MINUS) or float(s) == 0 else "+" + s


def num_floor(v: float, nd: int = 2) -> str:
    """Like :func:`num`, but ``< 0.01`` (for nd = 2) when the value rounds to zero: for η²ₚ and R²."""
    s = num(v, nd)
    if float(s.replace(MINUS, "-")) == 0:
        return "< 0." + "0" * (nd - 1) + "1"
    return s


def p_value(p: float) -> str:
    """``< 0.001`` below 0.001, else three decimals (``1.000`` for p = 1)."""
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def sci(v: float, nd: int = 1) -> str:
    """``m.m×10ⁿ`` with a superscript exponent."""
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10 ** e
    if float(f"{m:.{nd}f}") >= 10:  # 9.96e3 -> 1.0e4
        e += 1
        m /= 10
    return f"{num(m, nd)}×10{str(e).translate(SUPERSCRIPT)}"


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def bf_number(b: float) -> str:
    """One decimal below 10, none from 10 to 999, ``m.m×10ⁿ`` from 1,000."""
    if b < 10:
        return num(b, 1)
    if b < 999.5:
        return str(int(round(b)))
    return sci(b, 1)


def bf(bf10: float) -> str:
    """``BF₁₀ = x`` or ``BF₀₁ = x`` (BF₀₁ = 1/BF₁₀), whichever is at least 1."""
    return f"BF₁₀ = {bf_number(bf10)}" if bf10 >= 1 else f"BF₀₁ = {bf_number(1 / bf10)}"


def plain(v: float) -> str:
    """Shortest decimal form (``0.001``, ``50``), ``10⁻⁶`` for a negative power of ten, true minus."""
    if v != 0 and abs(v) < 1e-3:
        e = math.log10(abs(v))
        if abs(e - round(e)) < 1e-9:
            return ("" if v > 0 else MINUS) + "10" + str(int(round(e))).translate(SUPERSCRIPT)
    s = f"{v:.6g}"
    return s.replace("-", MINUS)


def _degrees(v: float) -> float:
    return round(math.degrees(v), 9)


# --------------------------------------------------------------------------- #
# Rules: name -> f(values, *args)
# --------------------------------------------------------------------------- #

def _one(vals):
    (v,) = vals
    return v


RULES: dict[str, Callable[..., str]] = {
    # numbers
    "num": lambda vals, nd="2": num(_one(vals), int(nd)),
    "signed": lambda vals, nd="0": signed(_one(vals), int(nd)),
    "num_floor": lambda vals, nd="2": num_floor(_one(vals), int(nd)),
    "int": lambda vals: num(_one(vals), 0),
    "int_comma": lambda vals: f"{int(round(_one(vals))):,}",
    "word": lambda vals: WORDS[int(round(_one(vals)))],
    "pct": lambda vals, nd="0": num(100 * _one(vals), int(nd)),
    "df": lambda vals: ", ".join(str(int(round(v))) for v in vals),
    "ci": lambda vals, nd="2": f"[{num(vals[0], int(nd))}, {num(vals[1], int(nd))}]",
    "num_n": lambda vals, nd="3": f"{num(vals[0], int(nd))} ({int(round(vals[1]))})",  # value (n)
    # p values, stars, Bayes factors
    "p": lambda vals: p_value(_one(vals)),
    "p_stars": lambda vals: p_value(_one(vals)) + stars(_one(vals)),
    "p_sci": lambda vals, nd="2": sci(_one(vals), int(nd)),
    "num_stars": lambda vals, nd="2": num(vals[0], int(nd)) + stars(vals[1]),  # value, p
    "bf": lambda vals: bf(_one(vals)),
    "bf10": lambda vals: bf_number(_one(vals)),  # the number of a BF₁₀ printed as "BF₁₀ = x"
    "bf01": lambda vals: bf_number(1 / _one(vals)),
    # arithmetic on the sources
    "diff_int": lambda vals: num(vals[0] - vals[1], 0),  # a - b
    "diff_word": lambda vals: WORDS[int(round(vals[0] - vals[1]))],
    "pct_diff": lambda vals, nd="0": num(100 * (vals[0] - vals[1]), int(nd)),  # 100 (a - b)
    "pct_complement": lambda vals, nd="0": num(100 * (1 - _one(vals)), int(nd)),  # 100 (1 - a)
    "ratio_pct": lambda vals, nd="0": num(100 * vals[0] / vals[1], int(nd)),  # 100 a / b
    "scaled": lambda vals, nd="0": num(vals[0] * vals[1], int(nd)),  # a * scale
    "scaled_signed": lambda vals, nd="0": signed(vals[0] * vals[1], int(nd)),
    "scaled_pct_of": lambda vals, nd="1": num(100 * vals[0] * vals[1] / vals[2], int(nd)),  # 100 a scale / b
    "scaled_pct_of_signed": lambda vals, nd="1": signed(100 * vals[0] * vals[1] / vals[2], int(nd)),
    "abs_max": lambda vals, nd="2": num(max(abs(v) for v in vals), int(nd)),
    "p_max": lambda vals: p_value(max(vals)),
    "p_min_floor": lambda vals, nd="2": f"{math.floor(min(vals) * 10 ** int(nd)) / 10 ** int(nd):.{nd}f}",
    "range": lambda vals, nd="0", scale="1": "–".join(num(float(scale) * v, int(nd)) for v in (min(vals), max(vals))),
    "range_abs": lambda vals, nd="2": " to ".join(num(v, int(nd)) for v in
                                                 (min(vals, key=abs), max(vals, key=abs))),
    "bf01_range": lambda vals: "–".join(bf_number(1 / v) for v in (max(vals), min(vals))),
    # how many of k p values print differently after a change: values = k before, then the same k after
    "changed_p": lambda vals, style="int": _count(sum(p_value(a) != p_value(b) for a, b in
                                                      zip(vals[:len(vals) // 2], vals[len(vals) // 2:])), style),
    # bounds: printed as the bound when every value is beyond it
    "gt": lambda vals, bound: bound if min(vals) > float(bound) else f"not > {bound} ({min(vals):.4g})",
    # model settings (Table 4)
    "bounds": lambda vals: f"[{plain(vals[0])}, {plain(vals[1])}]",
    "bounds_deg": lambda vals: f"[{plain(_degrees(vals[0]))}, {plain(_degrees(vals[1]))}]",
    "gamma": lambda vals, unit="": _gamma(_one(vals), unit),
    "normal": lambda vals: _normal(_one(vals)),
}


def _count(n: int, style: str) -> str:
    return WORDS[n] if style == "word" else str(n)


def _gamma(dist, unit: str) -> str:
    assert dist.dist.name == "gamma", dist.dist.name
    a, scale = dist.kwds["a"], dist.kwds["scale"]
    if unit == "deg":
        scale = _degrees(scale)
    return f"Gamma({plain(a)}, {plain(scale)})"


def _normal(dist) -> str:
    assert dist.dist.name == "norm", dist.dist.name
    return f"N({plain(dist.kwds['loc'])}, {plain(dist.kwds['scale'])})"


_RULE = re.compile(r"^(\w+)(?:\((.*)\))?$")


def apply(rule: str, values: list) -> str:
    """Apply ``rule`` (``name`` or ``name(arg, ...)``) to the source values."""
    m = _RULE.match(rule.strip())
    if not m or m.group(1) not in RULES:
        raise KeyError(f"unknown format rule: {rule!r}")
    args = [a.strip() for a in m.group(2).split(",")] if m.group(2) else []
    return RULES[m.group(1)](values, *args)


def normalize(s: str) -> str:
    """Drop typographic differences: hyphen-minus and en dash used as minus -> minus sign; no whitespace."""
    s = re.sub(r"(?<![\d)])[-–](?=[\d.])", MINUS, s)  # a sign, not a range dash between numbers
    return re.sub(r"\s+", "", s)
