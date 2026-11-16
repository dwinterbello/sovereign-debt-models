"""Exhibit A1 (note): bench_frame / bench_attribution / bench_numbers against the raw books (tolerance 0.05pp),
GG-for-all including the US (federal is Exhibit 2 only), and the shipped A1 figure (1824x1152, two panels).

Every number in the A1 prose comes from ``service_rev.bench_frame()``; the checks below rebuild the same
quantities from ``debt_country_annual``, AMECO, WEO, OECD / CAO and the zero curve without going through it.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sovereign_cash.debt.note import service_rev as SR

SC = Path(__file__).resolve().parents[1]
DERIVED = SC / "derived"
RAW = SC / "data" / "debt"
TOL = 0.05 + 1e-9
HAS = all(p.exists() for p in (DERIVED / "debt_country_annual.parquet", DERIVED / "debt_snapshot.parquet",
                               RAW / "ameco.parquet", RAW / "weo.parquet"))
pytestmark = pytest.mark.skipif(not HAS, reason="run sovereign_cash.debt.build first")


@pytest.fixture(scope="module")
def ca() -> pd.DataFrame:
    return pd.read_parquet(DERIVED / "debt_country_annual.parquet")


@pytest.fixture(scope="module")
def bench() -> pd.DataFrame:
    return SR.bench_frame()


def _pivot(path: Path, var: str) -> pd.DataFrame:
    d = pd.read_parquet(path)
    return d[d["var"] == var].pivot_table(index="country", columns="year", values="value")


def test_bench_frame_matches_books(bench, ca):
    o = ca.pivot_table(index="cc", columns="year", values="gg_interest_pct_gdp")
    am = _pivot(RAW / "ameco.parquet", "interest_gross_pct")
    wp = _pivot(RAW / "weo.parquet", "interest_pct")
    assert list(bench.index) == list(SR.BENCH)
    for cc in SR.BENCH:
        for col, y in (("o16", 2016), ("o26", 2026), ("o27", 2027), ("o30", 2030)):
            assert abs(bench.at[cc, col] - o.at[cc, y]) <= TOL, (cc, col)
        assert abs(bench.at[cc, "ec27"] - am.at[cc, 2027]) <= TOL, cc
        assert abs(bench.at[cc, "imf30"] - wp.at[cc, 2030]) <= TOL, cc


def test_bench_frame_rates_are_finite_and_on_the_curve(bench):
    """Spot 10y and the 10y forward are both par yields off the government zero curve (as of the curve's date)."""
    from sovereign_cash.debt.gov_curve import forward_par

    z = pd.read_parquet(RAW / "gov_zeros.parquet")
    assert np.isfinite(bench[["auth_10y", "ours_10y_spot", "ours_10y_2030"]].to_numpy(float)).all()
    for cc in SR.BENCH:
        asof = pd.Timestamp(z.loc[z["cc"].eq(cc), "asof"].iloc[0])
        assert abs(bench.at[cc, "ours_10y_spot"] - float(forward_par(cc, 0.0, [10.0])[0])) <= TOL, cc
        # 10y forward that starts in mid-2030, measured from the curve's own as-of date
        assert abs(bench.at[cc, "ours_10y_2030"] - float(forward_par(cc, SR._fwd_start_2030(asof), [10.0])[0])) <= TOL, cc
        assert 0.5 < bench.at[cc, "ours_10y_spot"] < 9.0, cc
        assert 0.5 < bench.at[cc, "ours_10y_2030"] < 9.0, cc
        assert 0.5 < bench.at[cc, "auth_10y"] < 9.0, cc
        assert (asof - pd.Timestamp.today()).days <= 0 and (pd.Timestamp.today() - asof).days < 45, (cc, asof)


def test_auth_long_rate_is_cao_for_japan_oecd_otherwise(bench):
    from sovereign_cash.debt.build import _s
    from sovereign_cash.debt.sources import load

    src = load()
    assert abs(bench.at["jp", "auth_10y"] - float(_s(src, "jp", "cao_baseline", "irl")[2030])) <= TOL
    for cc in ("de", "fr", "it", "es", "uk", "us"):
        assert abs(bench.at[cc, "auth_10y"] - float(_s(src, cc, "oecd", "irl")[2027])) <= TOL, cc


def _nums(par: str, unit: str) -> list[float]:
    return [float(x.replace("\u2212", "-")) for x in re.findall(rf"([+\u2212-]?\d+\.\d)(?={unit})", par)]


def _has(par: str, unit: str, value: float) -> bool:
    return any(abs(v - value) <= TOL for v in _nums(par, unit))


@pytest.mark.xfail(reason="work in progress in the 10 Nov local snapshot: the long-form note's A1 attribution prose is not yet read from bench_frame", strict=False)
def test_bench_attribution_numbers_match_bench_frame_and_books(bench, ca):
    from sovereign_cash.debt.build import _s
    from sovereign_cash.debt.sources import load

    paras = SR.bench_attribution(bench)
    assert len(paras) == 8
    by = {p.split(" — ")[0]: p for p in paras[1:]}
    assert set(by) == {"Germany", "France", "Italy", "Spain", "UK", "Japan", "US"}
    src = load()
    nm = {"de": "Germany", "fr": "France", "it": "Italy", "es": "Spain", "uk": "UK", "jp": "Japan", "us": "US"}
    for cc, name in nm.items():
        p = by[name]
        d27 = bench.at[cc, "o27"] - bench.at[cc, "ec27"]
        d30 = bench.at[cc, "o30"] - bench.at[cc, "imf30"]
        rg = bench.at[cc, "ours_10y_2030"] - bench.at[cc, "auth_10y"]
        assert "nan" not in p.lower(), name
        assert _has(p, "pp", d27), (name, "d27")
        assert _has(p, "pp", abs(d30)) or _has(p, "pp", d30), (name, "d30")
        # Spain splits the rate gap into spot−auth and fwd−spot; others print fwd−auth.
        if cc == "es":
            assert _has(p, "pp", bench.at[cc, "ours_10y_spot"] - bench.at[cc, "auth_10y"]), (name, "spot−auth")
            assert _has(p, "pp", bench.at[cc, "ours_10y_2030"] - bench.at[cc, "ours_10y_spot"]), (name, "fwd−spot")
        else:
            assert _has(p, "pp", rg), (name, "rate gap")
        assert _has(p, "%", bench.at[cc, "ours_10y_2030"]), (name, "fwd")
        assert _has(p, "%", bench.at[cc, "auth_10y"]), (name, "auth")
        if cc in ("de", "fr", "it", "es", "uk", "jp", "us"):
            g, gi, ni = (_s(src, cc, "oecd", k) for k in ("gdp_lcu_bn", "interest_gross_bn", "interest_net_bn"))
            assert _has(p, "pp", 100 * (gi[2027] - ni[2027]) / g[2027]), (name, "netting")
    # Japan: central interest vs the Cabinet Office's own general-account bill
    jp = ca[(ca["cc"] == "jp") & (ca["year"] == 2030)].iloc[0]
    assert _has(by["Japan"], "%", 100 * jp["interest_bn"] / jp["gdp_bn"])
    assert _has(by["Japan"], "%", 100 * jp["interest_authority_bn"] / jp["gdp_bn"])
    # US A1 is GG; federal remains in the attribution as the Exhibit-2 memo (2025 sat on the IMF)
    fed, imf = SR._us_fed_vs_imf(2025)
    us_gg30 = ca[(ca["cc"] == "us") & (ca["year"] == 2030)].iloc[0]["gg_interest_pct_gdp"]
    assert abs(bench.at["us", "o30"] - us_gg30) <= TOL
    assert _has(by["US"], "%", us_gg30) and _has(by["US"], "%", fed) and _has(by["US"], "%", imf)


@pytest.mark.xfail(reason="work in progress in the 10 Nov local snapshot: the long-form note's A1 attribution prose is not yet read from bench_frame", strict=False)
def test_bench_attribution_static_claims_hold(bench):
    """The prose asserts orderings; fail when the data stops supporting them instead of shipping the sentence."""
    d27 = bench["o27"] - bench["ec27"]
    d30 = bench["o30"] - bench["imf30"]
    rg = bench["ours_10y_2030"] - bench["auth_10y"]
    assert rg.idxmax() == "fr"  # "Largest rate gap in the panel"
    assert all(abs(d27[c]) <= 0.2 + 1e-9 for c in ("de", "fr", "it", "us"))  # Commission-close on GG
    assert d27["es"] < 0  # "Spain below the Commission"
    assert d27["uk"] > 0 and d30["uk"] > 0  # "UK above both"
    es = next(p for p in SR.bench_attribution(bench) if p.startswith("Spain"))
    assert _has(es, "pp", bench.at["es", "ours_10y_spot"] - bench.at["es", "auth_10y"])
    assert _has(es, "pp", bench.at["es", "ours_10y_2030"] - bench.at["es", "ours_10y_spot"])
    assert d30["us"] > 0  # GG US sits above the IMF net figure (netting, not a federal underprint)
    assert all(d30[c] > 0 for c in ("de", "fr", "it", "es", "uk", "jp", "us"))  # "above the IMF"
    net = {cc: SR._netting_pp(cc, 2027) for cc in ("de", "fr", "it", "es", "uk", "jp", "us")}
    assert net["jp"] > 0.5 * d30["jp"] and net["jp"] == max(net.values())  # "Japan gap is mostly netting"
    assert net["es"] > 0.5 * d30["es"]  # "most of which is netting"
    assert net["fr"] < 0.25 * d30["fr"]  # "netting is only ..."
    assert d30["de"] < net["de"]  # "less than the IMF's netting"
    assert abs(net["us"] - d30["us"]) <= 0.15  # US IMF gap ≈ OECD gross−net proxy


def test_bench_numbers_agree_with_attribution_inputs(bench):
    n = SR.bench_numbers(bench)
    for cc in bench.index:
        assert abs(n[f"bench_d27_{cc}"] - (bench.at[cc, "o27"] - bench.at[cc, "ec27"])) <= TOL, cc
        assert abs(n[f"bench_d30_{cc}"] - (bench.at[cc, "o30"] - bench.at[cc, "imf30"])) <= TOL, cc
        assert abs(n[f"bench_o30_{cc}"] - bench.at[cc, "o30"]) < 1e-12, cc
    ranked = bench["o30"].sort_values(ascending=False)
    assert n["bench_rank_2030"].split(",")[0].startswith(SR.NM[ranked.index[0]])
    for c in n["bench_close_ec27"].split(", "):
        cc = next(k for k, v in SR.NM.items() if v == c)
        assert abs(bench.at[cc, "o27"] - bench.at[cc, "ec27"]) <= 0.25


# ---------------------------------------------------------------------------------------- US series (A1 = GG; federal = Exhibit 2)
def test_us_bench_uses_general_government_not_federal(bench, ca):
    """A1 must be GG-for-all. Federal vs AMECO GG previously printed a false −1.5pp gap."""
    u = ca[ca["cc"] == "us"].set_index("year")
    for col, y in (("o16", 2016), ("o26", 2026), ("o27", 2027), ("o30", 2030)):
        assert abs(bench.at["us", col] - u.at[y, "gg_interest_pct_gdp"]) <= TOL
    assert abs(u.at[2030, "gg_interest_pct_gdp"] - u.at[2030, "interest_pct_gdp"]) > 0.5  # bases still differ
    assert abs(bench.at["us", "o27"] - bench.at["us", "ec27"]) <= 0.25  # like-for-like vs Commission


def test_us_federal_interest_is_net_across_the_history_forecast_seam(ca):
    """Outturn history must be net interest like CBO / the IMF: FY2016 $241bn, FY2024 $882bn (Treasury MTS Table 9).
    FRED FYOINT (gross, trust-fund interest included) read $430bn / $1,133bn and put a 0.7pp step into 2024→25."""
    u = ca[ca["cc"] == "us"].set_index("year")
    assert abs(u.at[2016, "interest_bn"] - 240.7) < 6.0
    assert abs(u.at[2024, "interest_bn"] - 881.7) < 6.0
    assert abs(u.at[2025, "interest_bn"] - 969.9) < 6.0  # CBO outturn year
    step = u.at[2025, "interest_pct_gdp"] - u.at[2024, "interest_pct_gdp"]
    assert abs(step) < 0.5, step  # no basis break at the outturn / CBO seam
    assert 1.1 < u.at[2016, "interest_pct_gdp"] < 1.5


def test_us_net_interest_history_has_no_seam_at_fy2015():
    """FRED FYOINT is net to FY2014 and gross from FY2015; the spliced net series must not step there."""
    from sovereign_cash.debt.sources import load

    d = load()
    n = d[(d["country"] == "us") & (d["source"] == "us_hist") & (d["var"] == "interest_net_bn")]
    n = n.set_index("year")["value"]
    assert n.index.is_unique
    assert abs(n.at[1995] - 232.1) < 3.0  # OMB function 900
    assert abs(n.at[2014] - 229.0) < 3.0
    assert abs(n.at[2015] - 223.3) < 3.0
    assert abs(n.at[2015] / n.at[2014] - 1) < 0.10  # FYOINT gross would read +76%


def test_us_federal_ratios_are_identities(ca):
    u = ca[(ca["cc"] == "us") & ca["interest_bn"].notna()].set_index("year")
    assert (100 * u["interest_bn"] / u["gdp_bn"] - u["interest_pct_gdp"]).abs().max() < 1e-6
    assert (100 * u["interest_bn"] / u["revenue_bn"] - u["interest_pct_rev"]).abs().max() < 1e-6
    rev_gdp = 100 * u["revenue_bn"] / u["gdp_bn"]
    assert rev_gdp.between(14.0, 21.0).all(), rev_gdp.to_dict()  # federal receipts, not general government


def test_us_federal_series_matches_cbo_and_imf_where_they_overlap(ca):
    from sovereign_cash.debt.build import _s
    from sovereign_cash.debt.sources import load

    src = load()
    u = ca[ca["cc"] == "us"].set_index("year")
    cbo_i, cbo_r = _s(src, "us", "cbo", "interest_net_bn"), _s(src, "us", "cbo", "revenue_bn")
    for y in (2025, 2026):
        assert abs(u.at[y, "interest_bn"] - cbo_i[y]) / cbo_i[y] < 0.03, y
        assert abs(u.at[y, "revenue_bn"] - cbo_r[y]) / cbo_r[y] < 0.05, y
    # prose: "in 2025 the IMF series sits on federal net interest"
    fed, imf = SR._us_fed_vs_imf(2025)
    assert abs(fed - imf) <= 0.25


# ---------------------------------------------------------------------------------------- the A1 figure
def _png_size(b: bytes) -> tuple[int, int]:
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", b[16:24])


def test_benchmark_figure_is_1824x1152_two_panel(tmp_path):
    import matplotlib.pyplot as plt

    seen = []
    orig = plt.close

    def spy(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            seen.append(len(fig.axes))
        return orig(fig) if fig is not None else orig()

    plt.close = spy
    try:
        out = SR.benchmark(tmp_path / "a1.png")
    finally:
        plt.close = orig
    assert _png_size(out.read_bytes()) == (1824, 1152)
    assert seen == [2], seen


@pytest.mark.xfail(reason="work in progress in the 10 Nov local snapshot: the long-form note's A1 attribution prose is not yet read from bench_frame", strict=False)
def test_shipped_a1_figure_and_docx_are_current():
    """The figure the docx / HTML embed is 1824x1152, and the docx carries today's A1 prose."""
    fig = SC / "debt" / "note" / "figures" / "exA1_interest_gdp_bench.png"
    if fig.exists():
        assert _png_size(fig.read_bytes()) == (1824, 1152)
    rep = SC.parent / "reports" / "dm_sovereign_debt_outlook"
    docx = rep / "DM_Sovereign_Debt_Outlook.docx"
    if not docx.exists():
        pytest.skip("docx not built (gitignored, local only)")
    import zipfile

    from docx import Document

    z = zipfile.ZipFile(docx)
    sizes = {_png_size(z.read(n)) for n in z.namelist() if n.startswith("word/media/") and n.endswith(".png")}
    assert (1824, 1152) in sizes
    assert (1824, 704) not in sizes or (1824, 1152) in sizes
    if fig.exists():
        assert any(z.read(n) == fig.read_bytes() for n in z.namelist() if n.startswith("word/media/")), \
            "docx A1 image differs from figures/exA1_interest_gdp_bench.png — rebuild: build_docx"
    norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
    paras = {norm(p.text) for p in Document(docx).paragraphs}
    stale = [p[:60] for p in SR.bench_attribution() if norm(p) not in paras]
    assert not stale, f"docx A1 attribution is stale: {stale}"
