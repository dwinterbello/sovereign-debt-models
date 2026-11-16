"""Every calculation behind the shock dashboard, written out.

``registry.EQUATIONS`` holds each formula (LaTeX), what it computes and the code
that implements it; ``page.render(pack)`` fills in the fitted numbers from the
scenario pack, so the page and the dashboard always show the same model.
"""
