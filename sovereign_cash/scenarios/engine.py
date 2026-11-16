"""Evaluate any combination of shocks against a scenario pack.

Pure standard library on purpose: the browser runs a line-for-line JavaScript
port (``web/engine.js``) and the parity tests hold the two to 1e-9 bn. The
models are fitted elsewhere (``pack.py``); this module only does arithmetic.

Sign convention: a positive delta is more borrowing (a higher cash
requirement). Each line in the pack carries the sign it enters the identity
with, e.g. receipts -1, spending +1.

Shock specification (all keys optional; a missing key is no shock)::

    {"ngdp":  {"size": 1.0, "mode": "level"},     # pp; "level" or "growth"
     "cpi":   {"size": 0.5, "mode": "growth"},    # pp
     "rates": {"size": 100.0},                    # bp, parallel, permanent
     "brent": {"size": -20.0},                    # % (Norway only)
     "methods": {"rates": "book"},                # pick among a driver's methods
     "scale": {"fr_recettes_ngdp": 1.5},          # stress one channel's response
     "feedback": true}                            # interest on the extra debt

Channel types: beta_monthly (SARIMAX coefficient, monthly), linear (response
per unit, per period), grid (responses at several sizes, interpolated),
elasticity (baseline x ((1 + gap/100)^eps - 1), averaged over the period's months),
elasticity_monthly (the same per month on a monthly baseline) and
matrix (shock-year x effect-year coefficients, as the OBR ready reckoner).

Level mode: the driver's level sits ``size`` per cent above baseline from the
shock date onward (a one-off shift). Growth mode: the driver grows ``size`` pp
a year faster, so the level gap compounds: 100 * ((1 + size/100)^(t/12) - 1)
after t months. See ``sovereign_cash.equations`` for each channel's formula.
"""
from __future__ import annotations

import math
from typing import Any

SCHEMA_VERSION = 1
MODES = ("level", "growth")


class PackError(ValueError):
    """The pack or the shock specification is malformed."""


def level_gap(size: float, mode: str, tau: int) -> float:
    """Percent gap of the driver level vs baseline, ``tau`` months after onset.

    ``tau`` = 0 means the shock has not reached this month yet (lags included).
    """
    if tau <= 0 or size == 0.0:
        return 0.0
    if mode == "level":
        return size
    if mode == "growth":
        return 100.0 * ((1.0 + size / 100.0) ** (tau / 12.0) - 1.0)
    raise PackError(f"unknown mode {mode!r}; expected one of {MODES}")


def _power(gap: float, eps: float) -> float:
    """(1 + gap/100)^eps - 1: the exact response of a constant-elasticity line to a level gap."""
    if gap == 0.0:
        return 0.0
    if gap <= -100.0:
        raise PackError("a level gap of -100% or less has no constant-elasticity response")
    return (1.0 + gap / 100.0) ** eps - 1.0


def _interp_grid(knots: list[float], values: list[list[float]], x: float) -> list[float]:
    """Piecewise-linear in ``x`` across knots; linear extrapolation off the ends."""
    if len(knots) != len(values) or len(knots) < 2:
        raise PackError("grid needs at least two knots, one value row per knot")
    if any(b <= a for a, b in zip(knots, knots[1:])):
        raise PackError("grid knots must be strictly increasing")
    i = 0
    while i < len(knots) - 2 and x > knots[i + 1]:
        i += 1
    x0, x1 = knots[i], knots[i + 1]
    w = (x - x0) / (x1 - x0)
    return [a + w * (b - a) for a, b in zip(values[i], values[i + 1])]


def _shock(shocks: dict, driver: str) -> tuple[float, str]:
    s = shocks.get(driver) or {}
    size = float(s.get("size", 0.0) or 0.0)
    mode = s.get("mode", "level") or "level"
    if mode not in MODES:
        raise PackError(f"shock {driver!r}: unknown mode {mode!r}")
    if not math.isfinite(size):
        raise PackError(f"shock {driver!r}: size must be finite")
    return size, mode


def channel_delta(ch: dict, shocks: dict, n_periods: int, month_period: list[int] | None) -> list[float]:
    """Change in the channel's line, per period, before the identity sign."""
    kind = ch["type"]
    size, mode = _shock(shocks, ch["driver"])
    out = [0.0] * n_periods
    if size == 0.0:
        return out
    if kind == "beta_monthly":
        if month_period is None:
            raise PackError(f"channel {ch['id']}: beta_monthly needs month_period")
        beta = float(ch["beta"])
        for tau, p in zip(ch["tau"], month_period):
            if p is None or p < 0:
                continue
            out[p] += beta * level_gap(size, mode, int(tau))
        return out
    if kind == "linear":
        per_unit = ch["per_unit"].get(mode)
        if per_unit is None:
            raise PackError(f"channel {ch['id']}: no '{mode}' response")
        return [size * float(v) for v in per_unit]
    if kind == "grid":
        return _interp_grid([float(k) for k in ch["knots"]],
                            [[float(v) for v in row] for row in ch["values"]], size)
    if kind == "elasticity":
        # constant elasticity: baseline x ((1 + gap/100)^eps - 1), averaged over the period's months
        eps = float(ch["elasticity"])
        return [float(b) * sum(_power(level_gap(size, mode, int(t)), eps) for t in taus) / len(taus)
                for b, taus in zip(ch["base"], ch["tau_months"])]
    if kind == "elasticity_monthly":
        # the same, month by month on the model's own monthly baseline (FR/DE/IT level specs)
        if month_period is None:
            raise PackError(f"channel {ch['id']}: elasticity_monthly needs month_period")
        eps = float(ch["elasticity"])
        for b, t, p in zip(ch["base"], ch["tau"], month_period):
            if p is None or p < 0:
                continue
            out[p] += float(b) * _power(level_gap(size, mode, int(t)), eps)
        return out
    if kind == "matrix":
        # coef[j][p]: effect in period p of one unit of the determinant in shock year j.
        # "level" determinants (price or income levels, %) take the level gap in year j;
        # "rate" determinants (inflation, pp) take the growth in year j: every year in
        # growth mode, the first exposed year only in level mode; "persistent" ones
        # (interest-rate levels, pp) stay shifted every year in either mode.
        coef = ch["coef"]
        x = []
        first = True
        for j, tau in enumerate(ch["tau"]):
            w = float(ch.get("exposure", [1.0] * len(ch["tau"]))[j])
            if ch["kind"] == "level":
                v = level_gap(size, mode, int(tau))
            elif ch["kind"] == "persistent":  # a rate level that stays shifted, whatever the mode
                v = size if int(tau) > 0 else 0.0
            elif ch["kind"] == "rate":
                if int(tau) <= 0:
                    v = 0.0
                elif mode == "growth" or first:
                    v = size
                    first = False
                else:
                    v = 0.0
            else:
                raise PackError(f"channel {ch.get('id')}: matrix kind must be level, rate or persistent")
            x.append(w * v)
        return [sum(x[j] * float(coef[j][p]) for j in range(len(x))) for p in range(n_periods)]
    raise PackError(f"channel {ch.get('id')}: unknown type {kind!r}")


def _feedback(fb: dict, pre: list[float], rates_bp: float) -> list[float]:
    """Interest on the extra borrowing the shocks create.

    Extra debt accumulates from the scenario's own deltas; in each period the
    stock carried in from earlier periods pays the full marginal rate and the
    period's own new borrowing pays half (issued through the year). The
    feedback is itself borrowed, so it is solved period by period.
    """
    rates = [float(r) + rates_bp / 100.0 for r in fb["rate_pct"]]
    out, stock = [], 0.0
    for d, r in zip(pre, rates):
        # f = r/100 * (stock + 0.5 * (d + f))  =>  f = r/100*(stock + d/2) / (1 - r/200)
        k = r / 100.0
        f = k * (stock + 0.5 * d) / (1.0 - 0.5 * k)
        out.append(f)
        stock += d + f
    return out


def evaluate(pack: dict, cc: str, shocks: dict[str, Any] | None = None) -> dict:
    """Baseline, deltas by line and by driver, and the shocked path for one country."""
    if pack.get("schema_version") != SCHEMA_VERSION:
        raise PackError(f"pack schema {pack.get('schema_version')} != engine {SCHEMA_VERSION}")
    shocks = shocks or {}
    c = pack["countries"][cc]
    periods = c["periods"]
    n = len(periods)
    signs = {ln["id"]: float(ln["sign"]) for ln in c["lines"]}
    by_line = {k: [0.0] * n for k in signs}
    by_driver: dict[str, list[float]] = {}
    methods = {**c.get("default_methods", {}), **(shocks.get("methods") or {})}
    for ch in c["channels"]:
        group = ch.get("method_group")
        if group is not None:
            chosen = methods.get(ch["driver"])
            if chosen is None:
                raise PackError(f"{cc}: driver {ch['driver']!r} has several methods and no default")
            if group != chosen:
                continue
        d = channel_delta(ch, shocks, n, c.get("month_period"))
        k = float((shocks.get("scale") or {}).get(ch["id"], 1.0))
        if not (0.0 <= k <= 5.0):
            raise PackError(f"scale for {ch['id']} must be within [0, 5]")
        if k != 1.0:
            d = [k * v for v in d]
        line = ch["line"]
        if line not in by_line:
            raise PackError(f"channel {ch['id']} targets unknown line {line!r}")
        sign = signs[line]
        drv = by_driver.setdefault(ch["driver"], [0.0] * n)
        for i, v in enumerate(d):
            by_line[line][i] += v
            drv[i] += sign * v
    pre = [sum(signs[k] * by_line[k][i] for k in by_line) for i in range(n)]
    fb = [0.0] * n
    if shocks.get("feedback") and c.get("feedback"):
        rates_bp, _ = _shock(shocks, "rates")
        if rates_bp == 0.0:  # curve mode: new borrowing prices between the two anchors
            rates_bp = 0.5 * (_shock(shocks, "r3m")[0] + _shock(shocks, "r10y")[0])
        fb = _feedback(c["feedback"], pre, rates_bp)
    total = [a + b for a, b in zip(pre, fb)]
    base = [float(v) for v in c["baseline"]["total"]]
    return {
        "cc": cc,
        "periods": periods,
        "baseline": base,
        "delta_by_line": by_line,
        "delta_by_driver": by_driver,
        "feedback": fb,
        "delta": total,
        "scenario": [b + d for b, d in zip(base, total)],
    }
