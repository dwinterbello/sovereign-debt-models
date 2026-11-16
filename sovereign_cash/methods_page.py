"""Build the Sovereign financing Methodology page from the current scenario pack.

Writes ``methodology.html`` and the redirect alias ``bottom_up_methods.html``. ``run_all`` calls this before the
scenario stage so a page exists even if that stage fails; the scenario stage rebuilds it from the fresh pack
(``sovereign_cash.web.pages.build``) and gates its embedded vintage against the shock dashboard.
"""
from __future__ import annotations

from pathlib import Path

from sovereign_cash.methodology.page import ALIAS, OUT, build as _build  # noqa: F401

HERE = Path(__file__).resolve().parent


def build() -> Path:
    return _build()


if __name__ == "__main__":
    print(build())
