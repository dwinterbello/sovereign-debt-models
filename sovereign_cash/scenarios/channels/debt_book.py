"""Interest on the ISIN book at a grid of parallel shifts to the government zero curve.

``debt.build.project`` keeps locked coupons on the outstanding book and prices new
issuance (bonds at the issuance-mix forward par, bills at the 3-month rate) off the
shifted curve. Responses are calendar-year interest differences vs the 0bp run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sovereign_cash.scenarios.channels.common import ts

RATES_GRID_BP = (-200.0, -100.0, 0.0, 100.0, 200.0, 300.0)


class DebtBook:
    def __init__(self, grid_bp: tuple[float, ...] = RATES_GRID_BP):
        from sovereign_cash.debt import build as B

        self.grid = tuple(float(b) for b in grid_bp)
        self.inputs = B.prepare_inputs()
        self.runs = {b: self.inputs.project(b) for b in self.grid}
        snap = self.inputs.snap.set_index("cc")
        self.marginal_rate = snap["marginal_rate"].astype(float).to_dict() if "marginal_rate" in snap else {}
        self.y10 = snap["y10"].astype(float).to_dict() if "y10" in snap else {}
        self.asof = ts(self.inputs.asof)
        self.y0 = int(self.inputs.y0)

    def countries(self) -> set[str]:
        return set(self.runs[0.0]["cc"].astype(str))

    def interest(self, cc: str, bp) -> pd.Series:
        d = self.runs[bp]
        return d[d["cc"] == cc].set_index("year")["interest_bn"].astype(float)

    def annual(self, cc: str, col: str, bp: float = 0.0) -> pd.Series:
        d = self.runs[bp]
        return d[d["cc"] == cc].set_index("year")[col].astype(float)

    def _row(self, cc: str, bp, years: list[int], weights: list[dict[int, float]] | None) -> list[float]:
        base, shk = self.interest(cc, 0.0), self.interest(cc, bp)
        diff = (shk - base).reindex(sorted(set(base.index) | set(shk.index))).fillna(0.0)
        if weights is None:
            return [float(diff.get(y, 0.0)) for y in years]
        return [float(sum(w * diff.get(y, 0.0) for y, w in wt.items())) for wt in weights]

    def channel(self, cc: str, years: list[int], *, line: str = "interest",
                weights: list[dict[int, float]] | None = None, note: str = "") -> dict | None:
        """``weights`` maps each period onto calendar years (fiscal years); None = calendar."""
        if cc not in self.countries():
            return None
        rows = [self._row(cc, b, years, weights) for b in self.grid]
        return {
            "id": f"{cc}_rates_book", "type": "grid", "driver": "rates", "line": line,
            "method_group": "book", "knots": list(self.grid), "values": rows,
            "method": "ISIN book: locked coupons + new issuance at the shifted forward par" + note,
            "badge": "ISIN book (grid)",
            "source": "sovereign_cash.debt.build.project; shift added to the government zero curve",
        }

    def annual_table(self, cc: str) -> pd.DataFrame:
        """``debt.build.combine``'s country-year table (as ``debt_country_annual.parquet``)."""
        from sovereign_cash.debt import build as B

        if not hasattr(self, "_cya"):
            i = self.inputs
            self._cya = B.combine(i.ca_hist, self.runs[0.0], i.ann, i.y0)
        d = self._cya
        return d[d["cc"] == cc].set_index("year")

    def curve_channels(self, cc: str, years: list[int], *, weights: list[dict[int, float]] | None = None,
                       cache: dict | None = None) -> list[dict]:
        """Grids in each anchor (the other at zero), the rest of the curve from curve_pca.

        The book is run on the anchored shape at each grid point, as the parallel grid is;
        the two anchors add. ``pack._check_curve`` measures what adding leaves out.
        """
        from sovereign_cash.scenarios import curve_pca

        if cc not in self.countries():
            return []
        fac, mkt = curve_pca.factors_for(cc, cache)
        if "error" in fac:
            return []
        out = []
        for driver in ("r3m", "r10y"):
            rows = []
            for b in self.grid:
                if b == 0.0:
                    rows.append(self._row(cc, 0.0, years, weights))
                    continue
                key = (driver, mkt, b)
                if key not in self.runs:
                    x3, x10 = (b, 0.0) if driver == "r3m" else (0.0, b)
                    self.runs[key] = self.inputs.project(curve_pca.shape(fac, x3, x10))
                rows.append(self._row(cc, key, years, weights))
            out.append({
                "id": f"{cc}_{driver}_book", "type": "grid", "driver": driver, "line": "interest",
                "knots": list(self.grid), "values": rows,
                "method": (f"ISIN book repriced on a curve shift anchored at the {'3-month' if driver == 'r3m' else '10-year'} "
                           f"point; other maturities move by their expected change given both anchors "
                           f"({mkt.upper()} curve history, {fac['sample'][0][:4]}-{fac['sample'][1][:4]}); "
                           f"run at {', '.join(f'{int(k):+d}' for k in self.grid)}bp, straight lines between"),
                "badge": "ISIN book (curve)",
                "source": "sovereign_cash.scenarios.curve_pca + sovereign_cash.debt.build.project",
                "factor_market": mkt,
            })
        return out

    def feedback_rate(self, cc: str) -> float | None:
        for src in (self.marginal_rate, self.y10):
            v = src.get(cc)
            if v is not None and np.isfinite(v):
                return float(v)
        return None
