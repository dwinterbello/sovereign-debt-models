"""One module per source of shock responses; each returns pack entries for its countries.

``sarimax``    FR / DE / IT component SARIMAX (fitted betas, monthly)
``obr``        UK, OBR ready reckoner (shock-year x effect-year coefficients)
``cbo``        US, CBO rules of thumb (sustained shocks, fiscal years)
``annual``     ES / SE / NO / JP, annual central-government lines x elasticities
``debt_book``  every country, ISIN-book interest at a grid of parallel shifts
"""
