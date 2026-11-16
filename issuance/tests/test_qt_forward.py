import pandas as pd

from issuance.qt_forward import (
    EA_HALT_SOURCE,
    EA_SOURCE,
    ES_HALT_SOURCE,
    ES_SOURCE,
    UK_SOURCE,
    US_SOURCE,
    qt_halt_delta,
    qt_path,
)


def test_guidance_does_not_book_private_sales():
    months = pd.DataFrame({
        "country": ["UK", "US", "DE", "FR", "IT", "ES"],
        "month": [pd.Timestamp("2026-10-01")] * 6,
    })
    # Widen so a redemption month is in the frame if the book has one.
    more = []
    for country in ("UK", "US", "DE", "FR", "IT", "ES"):
        for month in pd.date_range("2026-10-01", "2027-09-01", freq="MS"):
            more.append({"country": country, "month": month})
    path = qt_path(pd.DataFrame(more))
    uk = path[path["country"].eq("UK")]
    us = path[path["country"].eq("US")]
    ea = path[path["country"].isin(["DE", "FR", "IT"])]
    es = path[path["country"].eq("ES")]
    assert (uk["qt_source"] == UK_SOURCE).all()
    assert (us["qt_source"] == US_SOURCE).all()
    assert (ea["qt_source"] == EA_SOURCE).all()
    assert (es["qt_source"] == ES_SOURCE).all()
    assert (uk["official_dv01_usd"].abs() < 1e-9).all()
    # Rollover is absorption, not a runoff. Face stock stays put.
    assert (us["official_dv01_usd"] >= -1e-9).all()
    assert (us["official_stock_delta_mn"].abs() < 1e-9).all()
    # SOMA add-ons sit on top of announced sizes the forward already books: no official DV01 forward.
    assert (us["official_dv01_usd"].abs() < 1e-9).all()
    # APF book has gilts maturing inside the year. They leave the stock only.
    assert uk["official_stock_delta_mn"].sum() < 0
    # PSPP/PEPP are not reinvested, so the allocated book redeems.
    assert ea["official_stock_delta_mn"].sum() < 0
    assert (ea["official_face_mn"] <= 1e-6).all()
    # ES: country stock pace (no ISIN book); 1m residual DV01, not WAM×face.
    assert es["official_stock_delta_mn"].sum() < 0
    assert es["official_dv01_usd"].sum() < 0
    assert abs(es["official_dv01_usd"].sum()) < 5.0


def test_ea_reinvest_halt_zeroes_private_supply_from_runoff():
    more = [
        {"country": c, "month": m}
        for c in ("DE", "FR", "IT", "ES", "UK", "US")
        for m in pd.date_range("2026-10-01", "2027-09-01", freq="MS")
    ]
    base = qt_path(pd.DataFrame(more), ea_reinvest=False)
    halt = qt_path(pd.DataFrame(more), ea_reinvest=True)
    ea = ["DE", "FR", "IT", "ES"]
    assert base[base.country.isin(ea)]["official_face_mn"].sum() < 0
    assert abs(halt[halt.country.isin(["DE", "FR", "IT"])]["official_face_mn"].sum()) < 1e-9
    assert abs(halt[halt.country.eq("ES")]["official_face_mn"].sum()) < 1e-9
    assert (halt[halt.country.isin(["DE", "FR", "IT"])]["qt_source"] == EA_HALT_SOURCE).all()
    assert (halt[halt.country.eq("ES")]["qt_source"] == ES_HALT_SOURCE).all()
    # UK / US unchanged by the EA switch
    assert (halt[halt.country.eq("UK")]["qt_source"] == UK_SOURCE).all()
    assert (halt[halt.country.eq("US")]["qt_source"] == US_SOURCE).all()
    d = qt_halt_delta()
    assert set(d["country"]) >= {"DE", "FR", "IT", "ES"}
    assert (d["runoff_eur_bn"] > 0).all()
    assert abs(float(d["halt_face_mn"].sum())) < 1e-6
