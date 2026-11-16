"""US coupon auctions by bidder class (Treasury fiscal data auctions_query).

Competitive accepted amounts split into primary dealers, indirect bidders (foreign and domestic
real money bidding through dealers, incl. FIMA-like accounts) and direct bidders. The dealer
take-up share is the standard read of end demand: dealers are obliged to bid, so a high dealer
share means end investors left paper on the street. SOMA add-ons are non-competitive and excluded.
"""
from __future__ import annotations

import pandas as pd

from issuance.http import session
from issuance.paths import CACHE

URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
FIELDS = ("auction_date,security_type,security_term,original_security_term,cusip,high_yield,"
          "bid_to_cover_ratio,comp_accepted,primary_dealer_accepted,indirect_bidder_accepted,"
          "direct_bidder_accepted,total_accepted,inflation_index_security,floating_rate")


def _pages(start: str) -> list[dict]:
    s, rows, page = session(), [], 1
    while True:
        r = s.get(URL, params={"fields": FIELDS, "filter": f"auction_date:gte:{start}",
                               "page[size]": 10000, "page[number]": page, "sort": "auction_date"},
                  timeout=120)
        r.raise_for_status()
        js = r.json()
        rows += js["data"]
        if page >= int(js["meta"]["total-pages"]):
            return rows
        page += 1


def fetch_us_bidders(start: str = "2009-01-01") -> pd.DataFrame:
    df = pd.DataFrame(_pages(start))
    df = df[df["security_type"].isin(["Note", "Bond", "TIPS", "FRN"])]
    num = ["high_yield", "bid_to_cover_ratio", "comp_accepted", "primary_dealer_accepted",
           "indirect_bidder_accepted", "direct_bidder_accepted", "total_accepted"]
    for c in num:
        df[c] = pd.to_numeric(df[c].replace("null", None), errors="coerce")
    df["auction_date"] = pd.to_datetime(df["auction_date"])
    df = df[df["comp_accepted"] > 0].copy()
    for k in ("primary_dealer", "indirect_bidder", "direct_bidder"):
        df[k.split("_")[0] + "_share"] = df[f"{k}_accepted"] / df["comp_accepted"]
    linker = df["inflation_index_security"].astype(str).eq("Yes") | df["security_type"].eq("TIPS")
    frn = df["floating_rate"].astype(str).eq("Yes") | df["security_type"].eq("FRN")
    df["term"] = (df["original_security_term"].str.replace("-Year", "y").str.replace(" ", "")
                  .where(~linker, "TIPS " + df["original_security_term"].str.replace("-Year", "y")))
    df.loc[frn, "term"] = "FRN"
    keep = ["auction_date", "term", "cusip", "high_yield", "bid_to_cover_ratio", "comp_accepted",
            "total_accepted", "primary_share", "indirect_share", "direct_share"]
    out = df[keep].rename(columns={"bid_to_cover_ratio": "btc", "primary_share": "dealer_share"})
    out = out.sort_values("auction_date").reset_index(drop=True)
    out.to_parquet(CACHE / "us_bidders.parquet", index=False)
    print(f"  us bidders: {len(out)} auctions {out.auction_date.min().date()}..{out.auction_date.max().date()}")
    return out


if __name__ == "__main__":
    fetch_us_bidders()
