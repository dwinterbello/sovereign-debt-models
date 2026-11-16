"""ISIN-level PSPP / PEPP allocator.

ECB publishes country stock S and residual WAM W, not a holdings file.
Identification is S and W. Aging and market-neutral flow are how a CB
gets there; they do not override W.

    min  ||h − h_neutral||²
    s.t. 1′h = S*
         y′h = S* · W_pub
         one-sided aging box first
         two-way / cap-box only if W_pub is outside that box

h_neutral = aged + pro-rata flow (purchases ∝ eligible outstanding,
sales ∝ holdings). That is what stops a 10bn sale landing on one ISIN.

S* = largest face ≤ min(S, Σ cap o_i) whose cap box contains W_pub.
Filling every ISIN to cap pins WAM at the outstanding-weighted universe
WAM. Published holdings can sit below that (DE PSPP 2017–24). Shrink
the coupon book until a short-tilt can hit W_pub. Leftover is slack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from issuance.tenor import bucket_from_years

PSPP_CAP = 0.33
PEPP_CAP = 0.50
PROGRAM_RULES = {
    "PSPP": {"cap": PSPP_CAP, "resid_lo": 0.0, "resid_hi": 50.0, "buy_lo": 1.0, "buy_hi": 31.0},
    "PEPP": {"cap": PEPP_CAP, "resid_lo": 0.0, "resid_hi": 50.0, "buy_lo": 0.25, "buy_hi": 32.0},
}
# ECB reinvestment decisions. APP: partial from Mar-2023, none from Jul-2023 (Dec-2022 and
# Jun-2023 statements). PEPP: partial from Jul-2024, none from Jan-2025 (Dec-2023 statement).
PROGRAM_PHASES = {
    "PSPP": (("2023-03-01", "partial"), ("2023-07-01", "runoff")),
    "PEPP": (("2024-07-01", "partial"), ("2025-01-01", "runoff")),
}


def phase_of(program: str, month) -> str:
    out = "buy"
    for start, name in PROGRAM_PHASES.get(str(program), ()):
        if pd.Timestamp(month) >= pd.Timestamp(start):
            out = name
    return out


NONCENTRAL_ISIN = "{country}_NONCENTRAL"
WAM_TOL = 0.05
# Max share of |net flow| in one ISIN when the spread box is feasible.
FLOW_SHARE = 0.20
# Two-way budgets tried before the full cap box (fraction of outstanding / book).
SWITCH_STEPS = (
    (0.03, 0.03),
    (0.10, 0.08),
    (0.25, 0.20),
)


@dataclass
class AllocResult:
    holdings: np.ndarray
    isins: np.ndarray
    resid: np.ndarray
    outstanding: np.ndarray
    slack: float
    target_stock: float
    target_wam: float
    achieved_stock: float
    achieved_wam: float
    wam_lo: float
    wam_hi: float
    ok: bool
    aged_stock: float = 0.0
    aged_wam: float = float("nan")
    flow_mn: float = 0.0
    turnover_mn: float = 0.0
    extra_turnover_mn: float = 0.0
    n_changed: int = 0
    wam_used: float = float("nan")
    mode: str = "stock"


def _project_affine(h0: np.ndarray, y: np.ndarray, stock: float, duration: float) -> np.ndarray:
    n = int(h0.size)
    if n == 0:
        return h0
    if n == 1:
        return np.array([stock], dtype=float)
    sy = float(y.sum())
    sy2 = float(np.dot(y, y))
    rhs1 = stock - float(h0.sum())
    rhs2 = duration - float(np.dot(y, h0))
    det = n * sy2 - sy * sy
    if abs(det) < 1e-10:
        return h0 + (rhs1 / n)
    a = (rhs1 * sy2 - rhs2 * sy) / det
    b = (n * rhs2 - sy * rhs1) / det
    return h0 + a + b * y


def _extreme_wam(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, stock: float) -> tuple[float, float]:
    if stock <= 0:
        return float("nan"), float("nan")
    lo = np.minimum(lo, hi)
    if stock < float(lo.sum()) - 1e-6 or stock > float(hi.sum()) + 1e-6:
        return float("nan"), float("nan")
    base = lo.copy()
    rem0 = stock - float(base.sum())
    room = hi - lo

    def _fill(order: np.ndarray) -> float:
        rem, num = rem0, float(np.dot(base, y))
        for i in order:
            take = min(float(room[i]), rem)
            num += take * float(y[i])
            rem -= take
            if rem <= 1e-9:
                break
        if rem > 1e-6:
            return float("nan")
        return num / stock

    return _fill(np.argsort(y)), _fill(np.argsort(-y))


def _slsqp(h0, y, stock, duration, lo, hi) -> tuple[np.ndarray, bool]:
    from scipy.optimize import minimize
    cons = [
        {"type": "eq", "fun": lambda x: np.sum(x) - stock},
        {"type": "eq", "fun": lambda x: np.dot(x, y) - duration},
    ]
    res = minimize(
        lambda x: float(np.sum((x - h0) ** 2)),
        h0, method="SLSQP",
        bounds=list(zip(lo.tolist(), hi.tolist())),
        constraints=cons,
        options={"maxiter": 400, "ftol": 1e-12},
    )
    h = np.clip(res.x, lo, hi)
    stock_ok = abs(float(h.sum()) - stock) < 1e-2
    dur_ok = abs(float(np.dot(h, y)) - duration) < 5e-2
    return h, bool(res.success and stock_ok and dur_ok)


def project_box_affine(
    h0: np.ndarray, y: np.ndarray, stock: float, duration: float,
    lo: np.ndarray, hi: np.ndarray, max_iter: int = 80,
) -> tuple[np.ndarray, bool]:
    n = int(h0.size)
    h = np.clip(h0, lo, hi)
    free = np.ones(n, dtype=bool)
    for _ in range(max_iter):
        n_free = int(free.sum())
        if n_free == 0:
            return h, False
        h_fix = np.where(free, 0.0, h)
        s_res = stock - float(h_fix.sum())
        t_res = duration - float(np.dot(h_fix, y))
        h_free = _project_affine(h0[free], y[free], s_res, t_res)
        h = h.copy()
        h[free] = h_free
        viol_lo = free & (h < lo - 1e-8)
        viol_hi = free & (h > hi + 1e-8)
        if not viol_lo.any() and not viol_hi.any():
            stock_ok = abs(float(h.sum()) - stock) < 1e-3
            dur_ok = abs(float(np.dot(h, y)) - duration) < 1e-2
            return h, bool(stock_ok and dur_ok)
        h[viol_lo] = lo[viol_lo]
        h[viol_hi] = hi[viol_hi]
        free[viol_lo] = False
        free[viol_hi] = False
    return np.clip(h, lo, hi), False


def _age_prev(isins: np.ndarray, prev: dict[str, float] | None, hi: np.ndarray) -> np.ndarray:
    if not prev:
        return np.zeros(len(isins), dtype=float)
    aged = np.array([float(prev.get(str(i), 0.0)) for i in isins], dtype=float)
    return np.minimum(np.maximum(aged, 0.0), hi)


def _lock_window(hi: np.ndarray, aged: np.ndarray, y: np.ndarray,
                 buy_lo: float | None, buy_hi: float | None) -> np.ndarray:
    out = hi.copy()
    if buy_lo is None and buy_hi is None:
        return out
    lo_cut = -np.inf if buy_lo is None else float(buy_lo)
    hi_cut = np.inf if buy_hi is None else float(buy_hi)
    lock = (y < lo_cut) | (y > hi_cut)
    out[lock] = np.minimum(out[lock], aged[lock])
    return out


def _purchase_bounds(
    aged: np.ndarray, y: np.ndarray, hi: np.ndarray,
    stock: float, buy_lo: float | None, buy_hi: float | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    aged_stock = float(aged.sum())
    if aged_stock <= 1e-8:
        # Opening book: same purchase window as later months. Unrestricted hi
        # would buy <1y and >31y paper the programme cannot buy.
        hi_b = _lock_window(hi, np.zeros_like(hi), y, buy_lo, buy_hi)
        return np.zeros_like(hi), hi_b, "stock"
    if stock + 1e-6 >= aged_stock:
        lo = aged.copy()
        hi_b = _lock_window(hi, aged, y, buy_lo, buy_hi)
        if float(hi_b.sum()) + 1e-6 < stock:
            return lo, hi.copy(), "purchase_relaxed"
        return lo, hi_b, "purchase"
    return np.zeros_like(hi), aged.copy(), "sale"


def _switch_bounds(
    aged: np.ndarray, hi: np.ndarray, y: np.ndarray, outst: np.ndarray,
    buy_lo: float | None, buy_hi: float | None,
    stock: float, cap_frac: float, book_frac: float,
    lo_b: np.ndarray, hi_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cap_i = cap_frac * np.maximum(outst, 0.0)
    budget = book_frac * max(stock, float(aged.sum()), 1.0)
    if float(cap_i.sum()) > budget > 1e-8:
        cap_i = cap_i * (budget / float(cap_i.sum()))
    lo_s = np.maximum(0.0, aged - cap_i)
    hi_s = np.minimum(hi, aged + cap_i)
    lo_s = np.minimum(lo_s, lo_b)
    hi_s = np.maximum(hi_s, hi_b)
    hi_s = _lock_window(hi_s, aged, y, buy_lo, buy_hi)
    return lo_s, hi_s


def _neutral_start(
    aged: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    stock: float, outst: np.ndarray,
) -> np.ndarray:
    """Aged + pro-rata flow across every name with room (market-neutral)."""
    h0 = np.clip(aged, lo, hi)
    residual = stock - float(h0.sum())
    if residual > 1e-8:
        room = np.maximum(hi - h0, 0.0)
        w = np.where(room > 1e-8, np.maximum(outst, 0.0), 0.0)
        if float(w.sum()) > 1e-8:
            add = np.minimum(room, w / w.sum() * residual)
            h0 = h0 + add
            leftover = residual - float(add.sum())
            if leftover > 1e-8:
                room2 = np.maximum(hi - h0, 0.0)
                if float(room2.sum()) > 1e-8:
                    h0 = h0 + room2 / room2.sum() * leftover
    elif residual < -1e-8:
        room = np.maximum(h0 - lo, 0.0)
        w = np.where(room > 1e-8, np.maximum(h0, 0.0), 0.0)
        if float(w.sum()) > 1e-8:
            cut = np.minimum(room, w / w.sum() * (-residual))
            h0 = h0 - cut
            leftover = -residual - float(cut.sum())
            if leftover > 1e-8:
                room2 = np.maximum(h0 - lo, 0.0)
                if float(room2.sum()) > 1e-8:
                    h0 = h0 - room2 / room2.sum() * leftover
    return np.clip(h0, lo, hi)


def _flow_cap_box(
    aged: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    stock: float, flow_abs: float, w_pub: float, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap per-ISIN |Δ|. Start at FLOW_SHARE; widen only if W_pub is outside that box."""
    if flow_abs < 100.0:
        return lo, hi
    for share in (FLOW_SHARE, 0.33, 0.50, 0.75, 1.0):
        band = share * flow_abs
        lo_c = np.maximum(lo, aged - band)
        hi_c = np.minimum(hi, aged + band)
        if float(lo_c.sum()) - 1e-6 > stock or stock > float(hi_c.sum()) + 1e-6:
            continue
        w_lo, w_hi = _extreme_wam(y, lo_c, hi_c, stock)
        if _box_contains(w_pub, w_lo, w_hi):
            return lo_c, hi_c
    return lo, hi


def _solve(h0, y, stock, duration, lo, hi) -> tuple[np.ndarray, bool]:
    h, ok = project_box_affine(h0, y, stock, duration, lo, hi)
    if not ok:
        h, ok = _slsqp(h0, y, stock, duration, lo, hi)
    return h, ok


def _clamp_w(w_pub: float, w_lo: float, w_hi: float) -> float:
    w = float(w_pub)
    if pd.notna(w_lo) and pd.notna(w_hi):
        w = min(max(w, float(w_lo)), float(w_hi))
    return w


def _box_contains(w_pub: float, w_lo: float, w_hi: float, eps: float = 1e-6) -> bool:
    """W_pub is inside the box. No WAM_TOL slop — that slop is for ok=."""
    if pd.isna(w_lo) or pd.isna(w_hi):
        return False
    return float(w_lo) - eps <= float(w_pub) <= float(w_hi) + eps


def _wam_feasible_stock(y: np.ndarray, hi: np.ndarray, w_pub: float, s_max: float) -> float:
    """Largest face ≤ s_max whose [0, cap] box still contains W_pub.

    Cap-fill is one point (universe WAM). If that point is above W_pub,
    drop longs until a short-concentrated book can hit the print.
    Search uses a tight contain so S* is not the 0.05y-too-long boundary.
    """
    if s_max <= 1e-8 or pd.isna(w_pub):
        return float(s_max)
    lo = np.zeros_like(hi)
    w_lo, w_hi = _extreme_wam(y, lo, hi, float(s_max))
    if _box_contains(w_pub, w_lo, w_hi):
        return float(s_max)
    best = 0.0
    lo_s, hi_s = 0.0, float(s_max)
    for _ in range(48):
        mid = 0.5 * (lo_s + hi_s)
        w_lo, w_hi = _extreme_wam(y, lo, hi, mid)
        if _box_contains(w_pub, w_lo, w_hi):
            best = mid
            lo_s = mid
        else:
            hi_s = mid
    return float(best)


def allocate_isin(
    target_stock: float,
    target_wam: float,
    universe: pd.DataFrame,
    prev: dict[str, float] | None = None,
    cap: float = PSPP_CAP,
    buy_lo: float | None = None,
    buy_hi: float | None = None,
    prev_wams: list[float] | None = None,
    tilt_max: float | None = None,
    qmax: float | None = None,
    prev_target: float | None = None,
    phase: str = "buy",
) -> AllocResult:
    """Hit published stock + WAM. Age and spread the flow; two-way only if needed.

    phase="runoff": the Eurosystem neither buys nor sells; the book ages and the non-central
    slack takes the rest of the published change. phase="partial": the coupon book buys at most
    the published stock's change plus what matured out of the book."""
    del prev_wams, tilt_max, qmax
    empty = AllocResult(
        holdings=np.zeros(0), isins=np.array([], dtype=object),
        resid=np.zeros(0), outstanding=np.zeros(0),
        slack=max(0.0, float(target_stock)),
        target_stock=float(target_stock), target_wam=float(target_wam),
        achieved_stock=0.0, achieved_wam=float("nan"),
        wam_lo=float("nan"), wam_hi=float("nan"), ok=False,
    )
    if universe is None or universe.empty or target_stock <= 0:
        return empty
    u = universe.dropna(subset=["isin", "outstanding", "resid"]).copy()
    u = u[(u["outstanding"] > 0) & (u["resid"] > 0)]
    if u.empty:
        return empty
    isins = u["isin"].astype(str).to_numpy()
    outst = pd.to_numeric(u["outstanding"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(u["resid"], errors="coerce").to_numpy(dtype=float)
    hi = np.minimum(outst, cap * outst)
    aged = _age_prev(isins, prev, hi)
    # S* on the box we can actually hold (purchase window + already-aged <1y).
    hi_use = _lock_window(hi, aged, y, buy_lo, buy_hi)
    stock_cap = min(float(target_stock), float(hi_use.sum()))
    stock = _wam_feasible_stock(y, hi_use, float(target_wam), stock_cap)
    aged_stock = float(aged.sum())
    aged_wam = float(np.dot(aged, y) / aged_stock) if aged_stock > 1e-8 else float("nan")
    if prev and phase in ("partial", "runoff") and aged_stock > 1e-8:
        live = set(isins.tolist())
        matured = float(sum(v for k, v in prev.items() if k not in live))
        if phase == "runoff" and aged_stock <= float(target_stock) + 1e-6:
            # No trades. Re-spread the aged face so the book still prints the published WAM;
            # the caller books that as a revision, not as flow.
            stock = aged_stock
            box_lo, box_hi = np.zeros_like(hi), hi.copy()
            w_lo, w_hi = _extreme_wam(y, box_lo, box_hi, stock)
            w = _clamp_w(float(target_wam), w_lo, w_hi)
            h, ok = _solve(aged.copy(), y, stock, stock * w, box_lo, box_hi)
            if not ok:
                h = aged.copy()
            h = np.clip(h, box_lo, box_hi)
            ach = float(h.sum())
            ach_wam = float(np.dot(h, y) / ach) if ach > 0 else float("nan")
            return AllocResult(
                holdings=h, isins=isins, resid=y, outstanding=outst,
                slack=max(0.0, float(target_stock) - ach),
                target_stock=float(target_stock), target_wam=float(target_wam),
                achieved_stock=ach, achieved_wam=ach_wam,
                wam_lo=float(w_lo) if pd.notna(w_lo) else float("nan"),
                wam_hi=float(w_hi) if pd.notna(w_hi) else float("nan"),
                ok=bool(pd.notna(ach_wam) and abs(ach_wam - float(target_wam)) <= WAM_TOL),
                aged_stock=aged_stock, aged_wam=aged_wam, flow_mn=0.0,
                turnover_mn=float(np.abs(h - aged).sum()), wam_used=w, mode="runoff",
            )
        if prev_target is not None:
            budget = max(0.0, float(target_stock) - float(prev_target) + matured)
            stock = min(stock, aged_stock + budget)
    slack = max(0.0, float(target_stock) - stock)
    if stock <= 1e-8:
        empty.slack = float(target_stock)
        return empty
    lo_b, hi_b, mode = _purchase_bounds(aged, y, hi, stock, buy_lo, buy_hi)
    w_pub = float(target_wam)

    candidates: list[tuple[str, np.ndarray, np.ndarray]] = [(mode, lo_b, hi_b)]
    if mode != "stock":
        for cap_frac, book_frac in SWITCH_STEPS:
            lo_s, hi_s = _switch_bounds(
                aged, hi, y, outst, buy_lo, buy_hi, stock, cap_frac, book_frac, lo_b, hi_b,
            )
            candidates.append((mode + "_switch", lo_s, hi_s))
        lo_r = np.zeros_like(hi)
        hi_r = hi_use
        candidates.append((mode + "_reshuffle", lo_r, hi_r))

    chosen = None
    fallback = None
    for name, lo_x, hi_x in candidates:
        w_lo, w_hi = _extreme_wam(y, lo_x, hi_x, stock)
        fallback = (name, lo_x, hi_x, w_lo, w_hi)
        if _box_contains(w_pub, w_lo, w_hi):
            chosen = fallback
            break
    if chosen is None:
        chosen = fallback
    mode, box_lo, box_hi, w_lo, w_hi = chosen
    w = _clamp_w(w_pub, w_lo, w_hi)

    flow_abs = abs(stock - aged_stock)
    box_lo, box_hi = _flow_cap_box(aged, box_lo, box_hi, stock, flow_abs, w_pub, y)
    w_lo, w_hi = _extreme_wam(y, box_lo, box_hi, stock)
    w = _clamp_w(w_pub, w_lo, w_hi)
    h0 = _neutral_start(aged, box_lo, box_hi, stock, outst)

    h, ok = _solve(h0, y, stock, stock * w, box_lo, box_hi)
    if not ok:
        h, ok = _solve(h0, y, stock, stock * w, box_lo if mode.endswith("reshuffle") else lo_b,
                       box_hi if mode.endswith("reshuffle") else hi_b)
        if ok and not mode.endswith("reshuffle"):
            box_lo, box_hi = lo_b, hi_b
    if not ok:
        # Last resort: fill toward published WAM inside the widest tried box.
        h = h0
        h, ok = _solve(h0, y, stock, stock * w, box_lo, box_hi)
    h = np.clip(h, box_lo, box_hi)
    achieved = float(h.sum())
    achieved_wam = float(np.dot(h, y) / achieved) if achieved > 0 else float("nan")
    turnover = float(np.abs(h - aged).sum())
    extra = max(0.0, turnover - abs(achieved - aged_stock))
    n_changed = int(np.sum(np.abs(h - aged) > 1e-4)) if aged_stock > 1e-8 else int(np.sum(h > 1e-4))
    stock_hit = abs(achieved - stock) < 1e-2
    wam_hit = pd.notna(achieved_wam) and abs(achieved_wam - float(target_wam)) <= WAM_TOL
    return AllocResult(
        holdings=h, isins=isins, resid=y, outstanding=outst,
        slack=slack + max(0.0, float(target_stock) - slack - achieved),
        target_stock=float(target_stock), target_wam=float(target_wam),
        achieved_stock=achieved, achieved_wam=achieved_wam,
        wam_lo=float(w_lo) if pd.notna(w_lo) else float("nan"),
        wam_hi=float(w_hi) if pd.notna(w_hi) else float("nan"),
        ok=bool(stock_hit and wam_hit),
        aged_stock=aged_stock,
        aged_wam=aged_wam,
        flow_mn=achieved - aged_stock,
        turnover_mn=turnover,
        extra_turnover_mn=extra,
        n_changed=n_changed,
        wam_used=float(w),
        mode=mode,
    )


def result_frame(res: AllocResult, month, country: str, program: str) -> pd.DataFrame:
    rows = []
    for isin, h, y, o in zip(res.isins, res.holdings, res.resid, res.outstanding):
        if h <= 1e-8:
            continue
        rows.append({
            "month": month,
            "country": country,
            "program": program,
            "isin": str(isin),
            "tenor_bucket": bucket_from_years(float(y)),
            "resid_yrs": float(y),
            "outstanding_mn": float(o),
            "official_stock_mn": float(h),
            "wam_target": res.target_wam,
            "wam_achieved": res.achieved_wam,
        })
    if res.slack > 1e-6:
        rows.append({
            "month": month,
            "country": country,
            "program": program,
            "isin": NONCENTRAL_ISIN.format(country=country),
            "tenor_bucket": "non_central",
            "resid_yrs": float("nan"),
            "outstanding_mn": float("nan"),
            "official_stock_mn": float(res.slack),
            "wam_target": res.target_wam,
            "wam_achieved": res.achieved_wam,
        })
    return pd.DataFrame(rows)


def fit_row(res: AllocResult, month, country: str, program: str, n_live: int) -> dict:
    return {
        "month": month,
        "country": country,
        "program": program,
        "target_stock_mn": res.target_stock,
        "achieved_stock_mn": res.achieved_stock,
        "slack_mn": res.slack,
        "target_wam": res.target_wam,
        "achieved_wam": res.achieved_wam,
        "wam_used": res.wam_used,
        "wam_error": (res.achieved_wam - res.target_wam
                      if pd.notna(res.achieved_wam) else float("nan")),
        "wam_lo": res.wam_lo,
        "wam_hi": res.wam_hi,
        "n_isin": int((res.holdings > 1e-8).sum()) if res.holdings.size else 0,
        "n_changed": int(res.n_changed),
        "n_live": int(n_live),
        "ok": bool(res.ok),
        "aged_stock_mn": res.aged_stock,
        "aged_wam": res.aged_wam,
        "flow_mn": res.flow_mn,
        "turnover_mn": res.turnover_mn,
        "extra_turnover_mn": res.extra_turnover_mn,
        "mode": res.mode,
    }
