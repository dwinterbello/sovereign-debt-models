"""ONS public finances: the cash the gilt programme exists to raise.

Gilt supply is not decided in the gilt market. It falls out of the central government net cash
requirement, which the ONS publishes monthly, and the DMO then converts into a remit. Carrying the
cash requirement alongside issuance is what makes it possible to say whether a heavy auction month
is a funding need or a timing artefact.

Two groups of series are pulled. The first is fiscal: CGNCR is the one that matters and the rest are
context. The second is the holder base, which is the part almost nobody uses. The ONS publishes
overseas holdings of British government securities quarterly back to 1966, household holdings back to
1986, and the total central government long-term debt stock to divide them by, which together give a
published decomposition of who owns the gilt market without a single private data source.

Each series comes from the ONS time-series endpoint, which serves monthly, quarterly and annual
vintages of the same identifier in one JSON document, and all three are kept.

A warning on signs. The ONS reports net borrowing with the national-accounts convention, which is
not the one a trader would guess, so the framework keys off the cash requirement instead, where a
positive number unambiguously means cash the government must raise.

    python ons.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

import progress

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "ons"
OUTPUT = HERE / "output"

SERIES_PATH = "https://www.ons.gov.uk/{topic}/timeseries/{cdid}/{dataset}/data"
STALE_HOURS = 20
TIMEOUT = 90

# The public finances live under one topic and the holder breakdown under another, because the
# holdings are a national-accounts matter rather than a fiscal one.
FINANCE_TOPIC = "economy/governmentpublicsectorandtaxes/publicsectorfinance"
ACCOUNTS_TOPIC = "economy/grossdomesticproductgdp"
PAYMENTS_TOPIC = "economy/nationalaccounts/balanceofpayments"

# A series answers on one topic path and 404s on the others, and the ONS moves them between topics
# occasionally, so a miss falls through the rest rather than failing the series.
FALLBACK_TOPICS = (FINANCE_TOPIC, ACCOUNTS_TOPIC, PAYMENTS_TOPIC)


@dataclass(frozen=True)
class Item:
    cdid: str
    key: str
    label: str
    role: str
    topic: str = FINANCE_TOPIC
    dataset: str = "pusf"


SERIES = (
    Item("ruuw", "cgncr", "Central government net cash requirement",
         "the cash the gilt and bill programme has to raise; positive means raising"),
    Item("rurq", "psncr", "Public sector net cash requirement",
         "the wider public sector version, useful when CGNCR jumps on timing"),
    Item("j5ii", "psnb_ex", "Public sector net borrowing, excluding banks",
         "the accruals deficit, published with the national-accounts sign convention"),
    Item("jw2t", "current_deficit_ex", "Public sector current budget deficit, excluding banks",
         "the deficit net of investment, which is what the fiscal rules target"),
    Item("hf6w", "psnd_ex_bn", "Public sector net debt, excluding banks",
         "the stock the market has to hold, in billions"),
    Item("hf6x", "psnd_ex_pct_gdp", "Public sector net debt, excluding banks, share of GDP",
         "the same stock scaled, for comparison across decades"),
    Item("mf6u", "cg_expenditure", "Central government total expenditure",
         "context for whether the cash requirement is spending or receipts driven"),
    Item("mf6t", "cg_capital_expenditure", "Central government total capital expenditure",
         "the investment share, which the fiscal rules treat differently"),
    Item("bkpw", "cg_gross_debt", "Central government gross debt",
         "the whole stock at face value, monthly back to 1975, for scale"),
    Item("mex2", "apf_gilt_holdings_nominal", "APF gilt holdings at nominal value",
         "an independent monthly reading of official holdings, which the rebuilt Bank "
         "portfolio is checked against"),
    Item("mw7l", "il_capital_uplift", "Gross index-linked gilt capital uplift",
         "debt the government accrues without issuing anything, which supply figures miss"),
    Item("lsiw", "gilt_premia", "Net premia and discounts on gilt issuance",
         "the gap between cash raised and nominal issued, which is large when coupons "
         "are far from market yields"),
    # ---- who holds the stock ---------------------------------------------------------------
    # The gilt market's holder base is published, quarterly, and almost nobody uses it. Each of
    # these is a level in pounds; the shares are formed in framework.py against the total.
    Item("j8mv", "gilt_stock_total", "Long-term debt securities issued by central government",
         "the whole gilt stock at face value, the denominator for every holder share",
         dataset="pusf"),
    Item("j8mn", "bill_stock_total", "Short-term debt securities issued by central government",
         "the bill stock, which the front end of net supply swings on"),
    Item("hewd", "overseas_gilt_holdings", "Overseas holdings of British government securities",
         "the marginal buyer for thirty years, quarterly back to 1966",
         topic=PAYMENTS_TOPIC, dataset="ukea"),
    Item("aarb", "overseas_bill_holdings", "Overseas holdings of Treasury bills",
         "the short-dated part of the same flow", topic=PAYMENTS_TOPIC, dataset="ukea"),
    Item("nisu", "household_gilt_holdings", "Household holdings of central government securities",
         "retail, which is small but rises sharply when yields do",
         topic=ACCOUNTS_TOPIC, dataset="ukea"),
    Item("fzeu", "boe_gilt_holdings_face", "Bank of England holdings of central government "
         "long-term debt at face value",
         "a second official-holdings reading, monthly, on the national-accounts definition"),
    # ---- the holder base by sector, which is the part everyone says is not published ---------
    # It is published, quarterly back to 1987, under titles nobody would search for: the ONS's
    # financial accounts carry holdings of British government securities by ESA sector, in series
    # named things like "ESA:IP:AS:LEVEL". IP is insurance and pension funds, MF is monetary
    # financial institutions, OI is the other financial institutions where funds and leveraged
    # money sit, RW is the rest of the world. That distinction is the one that matters and the one
    # a four-way split cannot make: a liability-matching pension fund and a hedge fund financing
    # gilts in repo respond to the same price in opposite directions.
    #
    # Two things to know. MF includes the Bank of England, so commercial banks are MF less the
    # Bank, which is why fzeu above is fetched. And every one of these has a matching FLOW series,
    # which is the ONS's own estimate of transactions with the revaluation already removed: a
    # published version of what behaviour.py otherwise has to approximate, and therefore a check
    # on it rather than a duplicate of it.
    Item("nizc", "sector_level_insurance_pension", "Insurance corporations and pension funds: "
         "holdings of British government securities",
         "the liability matchers, who buy long duration because they are short it",
         topic=ACCOUNTS_TOPIC, dataset="ukea"),
    Item("nntw", "sector_level_banks_and_boe", "Monetary financial institutions: holdings of "
         "British government securities",
         "banks and the Bank of England together; net the Bank off for commercial banks",
         topic=ACCOUNTS_TOPIC, dataset="ukea"),
    Item("njsa", "sector_level_other_financial", "Other financial institutions: holdings of "
         "British government securities",
         "investment funds, money market funds and the leveraged accounts; the most price "
         "sensitive money in the market",
         topic=ACCOUNTS_TOPIC, dataset="ukea"),
    Item("nldu", "sector_level_overseas", "Rest of the world: holdings of British government "
         "securities",
         "the national-accounts version of the overseas holding, which should match hewd",
         topic=ACCOUNTS_TOPIC, dataset="ukea"),
    Item("njfk", "sector_level_local_authorities", "Local authorities: holdings of British "
         "government securities", "small, included so the sectors reconcile to the total",
         topic=ACCOUNTS_TOPIC, dataset="ukea"),
    Item("nkeq", "sector_level_public_corporations", "Public corporations: holdings of British "
         "government securities", "small, included so the sectors reconcile to the total",
         topic=ACCOUNTS_TOPIC, dataset="ukea"),
    Item("nkky", "sector_level_private_corporations", "Private non-financial corporations: "
         "holdings of British government securities",
         "small, included so the sectors reconcile to the total",
         topic=ACCOUNTS_TOPIC, dataset="ukea"),
    Item("nyxr", "sector_level_all_uk", "All UK holders: holdings of British government "
         "securities", "the UK total, for checking the sectors add up",
         topic=ACCOUNTS_TOPIC, dataset="ukea"),
    # The transactions. Positive is net acquisition, and the revaluation is already out.
    Item("nbtf", "sector_flow_insurance_pension", "Insurance corporations and pension funds: net "
         "transactions in British government securities",
         "the ONS's own purchase figure, against which our valuation adjustment is checked",
         topic=ACCOUNTS_TOPIC, dataset="ukea"),
    Item("ngcz", "sector_flow_banks_and_boe", "Monetary financial institutions: net transactions "
         "in British government securities", "includes the Bank's own operations",
         topic=ACCOUNTS_TOPIC, dataset="ukea"),
    Item("ncmd", "sector_flow_other_financial", "Other financial institutions: net transactions "
         "in British government securities", "the funds, and the sharpest quarterly swings",
         topic=ACCOUNTS_TOPIC, dataset="ukea"),
    Item("nexl", "sector_flow_overseas", "Rest of the world: net transactions in British "
         "government securities", "overseas buying, published rather than inferred",
         topic=ACCOUNTS_TOPIC, dataset="ukea"),
    Item("nbmx", "sector_flow_households", "Households: net transactions in British government "
         "securities", "retail, small", topic=ACCOUNTS_TOPIC, dataset="ukea"),
)


OFFLINE = False     # set by build(fetch=False); every read then comes from the cache alone


def _age_hours(path: Path) -> float:
    if not path.exists():
        return float("inf")
    return (time.time() - path.stat().st_mtime) / 3600


def _fetch(item: Item, session: requests.Session, refresh: bool) -> dict:
    path = CACHE / f"{item.cdid}.json"
    if not refresh and _age_hours(path) < STALE_HOURS:
        return json.loads(path.read_text(encoding="utf-8"))
    if OFFLINE:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        raise RuntimeError(f"{item.cdid}: offline and not cached")

    topics = (item.topic,) + tuple(topic for topic in FALLBACK_TOPICS if topic != item.topic)
    response = None
    for topic in topics:
        url = SERIES_PATH.format(topic=topic, cdid=item.cdid, dataset=item.dataset)
        response = session.get(url, timeout=TIMEOUT)
        if response.status_code == 200:
            break
    if response is None or response.status_code != 200:
        code = response.status_code if response is not None else "no reply"
        if path.exists():
            progress.write(f"    {item.cdid}: {code}, keeping cached copy")
            return json.loads(path.read_text(encoding="utf-8"))
        raise RuntimeError(f"{item.cdid}: ONS returned {code} on every topic path")

    payload = response.json()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _observations(payload: dict, item: Item) -> pd.DataFrame:
    """Flatten the three vintages the endpoint returns into one long frame."""
    description = payload.get("description", {})
    units = f"{description.get('preUnit', '')}{description.get('unit', '')}".strip()
    if not units:
        # Some series leave both unit fields empty and state the unit in the title instead.
        title = str(description.get("title", ""))
        for pattern, name in ((r"£\s?bn|£\s?billion", "£bn"), (r"£\s?m|£\s?million", "£m"),
                              (r"%|per\s?cent", "%")):
            if re.search(pattern, title, re.I):
                units = name
                break
    units = units or "unknown"
    rows = []
    for block, frequency in (("months", "monthly"), ("quarters", "quarterly"),
                             ("years", "annual")):
        for record in payload.get(block, []):
            value = record.get("value")
            if value in (None, "", ".."):
                continue
            if frequency == "monthly":
                stamp = pd.to_datetime(record["date"], format="%Y %b", errors="coerce")
            elif frequency == "quarterly":
                # end_time lands on 23:59:59.999999999, which parquet cannot store.
                stamp = pd.Period(record["date"].replace(" ", ""), freq="Q").end_time.normalize()
            else:
                stamp = pd.to_datetime(f"{record['date']}-12-31", errors="coerce")
            if pd.isna(stamp):
                continue
            rows.append({"date": stamp, "frequency": frequency, "series": item.key,
                         "label": item.label, "cdid": item.cdid.upper(),
                         "units": units, "value": float(value),
                         "released": record.get("updateDate")})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["released"] = pd.to_datetime(frame["released"], errors="coerce", utc=True) \
            .dt.tz_localize(None)
    return frame


def build(refresh: bool = False, quiet: bool = False, fetch: bool = True) -> pd.DataFrame:
    global OFFLINE
    OFFLINE = not fetch
    session = requests.Session()
    session.headers.update({"User-Agent": "uk_flows/1.0 (public data, contact via repo)"})

    frames = []
    with progress.bar(total=len(SERIES), label="ONS series", unit="series",
                      enabled=not quiet) as step:
        for item in SERIES:
            try:
                payload = _fetch(item, session, refresh=refresh)
                piece = _observations(payload, item)
            except Exception as error:                       # noqa: BLE001 - one series, not the run
                progress.write(f"    {item.cdid}: {error}")
                step(item.key)
                continue
            if piece.empty:
                progress.write(f"    {item.cdid}: no observations")
            else:
                frames.append(piece)
            step(item.key)

    if not frames:
        raise RuntimeError("no ONS series parsed")

    frame = (pd.concat(frames, ignore_index=True)
             .sort_values(["series", "frequency", "date"])
             .drop_duplicates(["series", "frequency", "date"], keep="last")
             .reset_index(drop=True))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT / "public_finances.parquet", index=False)
    frame.to_csv(OUTPUT / "public_finances.csv.gz", index=False, compression="gzip")

    if not quiet:
        monthly = frame[frame["frequency"] == "monthly"]
        print(f"  public_finances {len(frame):>7,} rows, {frame['series'].nunique()} series, "
              f"monthly to {monthly['date'].max():%Y-%m}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-fetch", action="store_true",
                        help="use the cache only, touch no network")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    build(refresh=arguments.refresh, quiet=arguments.quiet, fetch=not arguments.no_fetch)


if __name__ == "__main__":
    main()
