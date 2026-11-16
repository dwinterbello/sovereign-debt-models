"""PSPP cells quote thousands commas. PEPP does not. Both must survive _num + melt."""

import io

import pandas as pd

from issuance.fetch_ecb_qe import _align_published_stock, _is_cum_col, _melt_safe, _num, _section_frames


def test_num_strips_thousands_comma():
    assert _num("-9,418") == -9418.0
    assert _num("435,323") == 435323.0
    assert _num("1,868,273") == 1868273.0
    assert _num(-1275) == -1275.0
    assert _num("0") == 0.0
    assert _num("299921") == 299921.0
    assert pd.isna(_num("-"))
    assert pd.isna(_num(""))


def test_cum_col_names():
    assert _is_cum_col("cumulative net purchases")
    assert _is_cum_col("Cumulative")
    assert not _is_cum_col("Aug-26")
    assert not _is_cum_col("31/08/2026")


def test_melt_quoted_pspp_cells():
    """Regression: pd.to_numeric('-9,418') is NaN and used to drop August."""
    csv = (
        "Jurisdiction,Jul-26,Aug-26,Cumulative net purchases\n"
        'Germany,-1275,"-9,418","435,323"\n'
        'France,0,0,"360,233"\n'
        'Italy,-3626,"-2,957","282,861"\n'
    )
    melted = _melt_safe(pd.read_csv(io.StringIO(csv)), "PSPP", "net")
    de_aug = melted[(melted.country == "DE") & (melted.date == pd.Timestamp("2026-08-01"))]
    assert len(de_aug) == 1
    assert float(de_aug.iloc[0]["value"]) == -9418.0
    it_aug = melted[(melted.country == "IT") & (melted.date == pd.Timestamp("2026-08-01"))]
    assert float(it_aug.iloc[0]["value"]) == -2957.0
    pub = melted.attrs["published"].set_index("country")["published_stock_mn"]
    assert float(pub["DE"]) == 435323.0
    assert float(pub["FR"]) == 360233.0
    assert float(pub["IT"]) == 282861.0


def test_melt_pepp_unquoted_integers():
    csv = (
        "Jurisdiction,Jul-26,Aug-26,Cumulative\n"
        "Germany,-525,-5698,299921\n"
        "France,-338,0,239525\n"
        "Italy,-2935,-3250,221009\n"
    )
    melted = _melt_safe(pd.read_csv(io.StringIO(csv)), "PEPP", "net")
    de_aug = melted[(melted.country == "DE") & (melted.date == pd.Timestamp("2026-08-01"))]
    assert float(de_aug.iloc[0]["value"]) == -5698.0
    pub = melted.attrs["published"].set_index("country")["published_stock_mn"]
    assert float(pub["DE"]) == 299921.0
    assert float(pub["FR"]) == 239525.0
    assert float(pub["IT"]) == 221009.0


def test_section_frames_keeps_quoted_cells():
    # Live ECB files have 80–140 date columns. The splitter skips lines with
    # fewer than 5 commas so notes / titles do not become headers.
    months = ",".join(f"31/0{m}/2026" for m in range(1, 7))
    zeros = ",".join(["0"] * 6)
    text = (
        "Monthly net purchases\n"
        f"Jurisdiction,{months},31/07/2026,31/08/2026,cumulative net purchases\n"
        f'Germany,{zeros},-1275,"-9,418","435,323"\n'
        f'France,{zeros},0,0,"360,233"\n'
        "\n"
        "WAM of PSPP portfolio holdings\n"
        f"Jurisdiction,{months},31/07/2026,31/08/2026,\n"
        f"Germany,{zeros},6.47,6.55,\n"
        f"France,{zeros},6.11,6.03,\n"
    )
    sections = _section_frames(text)
    kinds = [t for t, _ in sections]
    assert any("Monthly net" in t for t in kinds)
    assert any("WAM" in t for t in kinds)
    net = next(df for t, df in sections if "Monthly net" in t)
    melted = _melt_safe(net, "PSPP", "net")
    de_aug = melted[(melted.country == "DE") & (melted.date >= pd.Timestamp("2026-08-01"))]
    assert float(de_aug.iloc[0]["value"]) == -9418.0


def test_live_ecb_files_split_net_and_wam():
    from issuance.fetch_ecb_qe import _classify
    from issuance.paths import RAW
    files = {
        RAW / "ecb" / "pspp_breakdown.csv": ("Monthly net purchases", "WAM of PSPP"),
        RAW / "ecb" / "pepp_public.csv": ("Net purchases", "WAM of Public Sector"),
    }
    for path, expect in files.items():
        if not path.exists():
            continue
        sections = _section_frames(path.read_text(encoding="utf-8", errors="replace"))
        titles = [t for t, _ in sections]
        kinds = {_classify(t) for t in titles}
        assert "net" in kinds and "wam" in kinds
        assert any(expect[0] in t for t in titles)
        assert any(expect[1] in t for t in titles)


def test_live_holdings_match_published_column():
    """If the cache exists, last cumsum must equal the CSV published column."""
    from issuance.paths import CACHE
    path = CACHE / "ecb_qe_holdings.parquet"
    if not path.exists():
        return
    ecb = pd.read_parquet(path)
    last = ecb.sort_values("date").groupby(["program", "country"]).tail(1)
    core = last[last["country"].isin(["DE", "FR", "IT"])]
    assert not core.empty
    if "published_stock_mn" in core.columns:
        gap = (core["holdings_mn"] - core["published_stock_mn"]).abs()
        assert float(gap.max()) <= 2.0
    de_pspp = last[(last.program == "PSPP") & (last.country == "DE")]
    assert not de_pspp.empty
    assert pd.Timestamp(de_pspp.iloc[0]["date"]) >= pd.Timestamp("2026-08-01")
    if "published_stock_mn" in de_pspp.columns:
        assert float(de_pspp.iloc[0]["holdings_mn"]) == float(de_pspp.iloc[0]["published_stock_mn"])
    assert pd.notna(de_pspp.iloc[0].get("wam_yrs"))


def test_align_published_stock_moves_dust_only():
    df = pd.DataFrame({
        "program": ["PSPP", "PSPP", "PSPP"],
        "country": ["LV", "LV", "DE"],
        "date": pd.to_datetime(["2026-07-31", "2026-08-31", "2026-08-31"]),
        "value": [10.0, -4.0, 100.0],
        "holdings_mn": [3068.0, 3064.0, 435000.0],
        "published_stock_mn": [3070.0, 3070.0, 900000.0],
    })
    out = _align_published_stock(df)
    lv = out[(out.country == "LV")].sort_values("date").iloc[-1]
    de = out[out.country == "DE"].iloc[0]
    assert abs(float(lv.holdings_mn) - 3070.0) < 1e-6
    assert abs(float(lv.value) - 2.0) < 1e-6
    assert float(de.holdings_mn) == 435000.0
