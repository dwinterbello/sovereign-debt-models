import pandas as pd

from issuance.habitat import build, price_response


def test_jp_bill_foreigners_are_the_habitat_buyer():
    h = build()
    jp = h[(h.country == "JP") & (h.sleeve == "bills") & (h.leg == "foreign")]
    assert len(jp) > 24
    assert jp.habitat.all()
    assert jp.z.abs().max() <= 4


def test_us_bill_auction_foreigners_are_not_the_money_fund():
    h = build()
    us = h[(h.country == "US") & (h.sleeve == "bills") & (h.leg == "foreign")]
    assert len(us) > 24
    assert not us.habitat.any()


def test_official_is_a_flow():
    h = build()
    off = h[(h.country == "US") & (h.sleeve == "int") & (h.leg == "official")]
    assert (off.flow < 0).any()
    assert (off.flow > 0).any()


def test_insurers_do_not_buy_the_ultra_cheapening():
    r = price_response()
    row = r[(r.country == "JP") & (r.sleeve == "ultra") & (r.buyer == "insurers")].iloc[0]
    assert row.stance == "insensitive"


def test_regional_banks_buy_the_intermediate_cheapening():
    r = price_response()
    row = r[(r.country == "JP") & (r.sleeve == "int") & (r.buyer == "regional_banks")].iloc[0]
    assert row.stance == "sensitive"


def test_foreigners_buy_bills_after_yields_fall():
    r = price_response()
    row = r[(r.country == "JP") & (r.sleeve == "bills") & (r.buyer == "foreigners")].iloc[0]
    assert row.stance == "rally"


def test_dealer_bill_book_is_elastic():
    from issuance.experiments.buyer_tape import build as tape

    row = tape()
    hit = row[(row.country == "US") & (row.buyer == "us_dealers") & (row.sleeve == "bills")].iloc[0]
    assert hit.role == "elastic"
    assert hit["n_2010+"] >= 36


def test_uk_bank_tenor_tape_is_elastic():
    from issuance.experiments.buyer_tape import build as tape

    row = tape()
    hit = row[(row.country == "UK") & (row.buyer == "uk_banks") & (row.sleeve == "ultra")].iloc[0]
    assert hit.role == "elastic"
    assert hit["n_2010+"] >= 36


def test_z1_life_insurers_are_a_liability_matcher():
    from issuance.experiments.buyer_tape import build as tape

    row = tape()
    hit = row[(row.country == "US") & (row.buyer == "life_insurance") & (row.sleeve == "coupon")].iloc[0]
    assert hit.role == "liability matcher"
    assert hit["n_2010+"] >= 16


def test_gilt_role_on_japanese_insurers_is_not_a_forward_link():
    from issuance.experiments.buyer_tape import build as tape

    row = tape()
    hit = row[(row.sleeve == "ultra") & (row.buyer == "insurers")].iloc[0]
    assert hit.role == "liability matcher"
    assert hit.verdict == "no link"


def test_us_pensions_do_not_chase_the_long_bond_auction():
    r = price_response()
    row = r[(r.country == "US") & (r.sleeve == "ultra") & (r.buyer == "pension")].iloc[0]
    assert row.stance == "insensitive"
