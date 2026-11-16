"""Refresh every country cash framework, then debt / note / methodology / hub.

One command: country cash → primary_path → debt board → note (HTML/PDF/docx) →
FYTD → methodology → scenario pack / shock dashboard → financing hub.

A country that fails keeps its previous outputs; the failure is printed and the
exit code is non-zero so the site pipeline flags it.

  PYTHONPATH=. .venv/bin/python -m sovereign_cash.run_all [--refresh] [--only uk se]
"""
from __future__ import annotations

import argparse
import importlib
import sys
import traceback

from sovereign_cash.build_financing import build as build_financing
from sovereign_cash.build_hub import COUNTRIES, build

CC = [cc for cc, _, _ in COUNTRIES if cc != "us"]
SIGNAL_MAX_AGE_DAYS = 7


def primary_path() -> None:
    """Regenerate ``derived/primary_path.json`` before the debt stamp / note read it."""
    from sovereign_cash.scenarios import primary as P

    P.write(P.build())


def debt(refresh: bool = False) -> None:
    """Debt & supply tab: panel and scenarios every run; the panel test (~1 min) when a week old."""
    import time

    from sovereign_cash.debt import build as debt_build
    from sovereign_cash.debt import page, signal

    primary_path()
    debt_build.run(refresh)
    sig = debt_build.DERIVED / "debt_signal.json"
    if not sig.exists() or time.time() - sig.stat().st_mtime > SIGNAL_MAX_AGE_DAYS * 86400:
        signal.run()
    page.build()


def note() -> None:
    """DM sovereign debt outlook: HTML/PDF figures + Word pack from the stamped debt board."""
    import shutil
    from pathlib import Path

    from sovereign_cash.debt.note import build_docx
    from sovereign_cash.debt.note.build import build as build_note

    build_note(pdf=True)
    docx = build_docx.build()
    # catalog DOWNLOADS path is the undated name
    stable = Path(docx).with_name("DM_Sovereign_Debt_Outlook.docx")
    if Path(docx).resolve() != stable.resolve():
        shutil.copy2(docx, stable)


def scenarios() -> None:
    """Scenario pack (fits, authority tables, debt book), then the shock dashboard and Methodology pages."""
    from sovereign_cash.scenarios import pack
    from sovereign_cash.web import pages

    import time

    from sovereign_cash.scenarios import curve_pca
    from sovereign_cash.scenarios.channels.debt_book import DebtBook

    # curve factors move slowly; re-estimate weekly (four curve histories, ~1 min)
    if not curve_pca.CACHE.exists() or (__import__("os").environ.get("SOVDEBT_PULL") == "1" and time.time() - curve_pca.CACHE.stat().st_mtime > 7 * 86400):
        curve_pca.build_cache()
    book = DebtBook()
    p = pack.build(book=book)
    p["checks"] = pack.check(p, book)
    pack.write(p)
    pages.build(p)
    if p["errors"]:
        raise RuntimeError(f"scenario pack: {p['errors']}")


def _assert_methodology_hub() -> None:
    """Refuse to leave the old Equations / How forecasts work split on disk."""
    from pathlib import Path

    here = Path(__file__).resolve().parent
    meth = here / "methodology.html"
    hub = here / "financing_dashboard.html"
    eq = here / "equations.html"
    if not meth.is_file() or meth.stat().st_size < 20_000:
        raise RuntimeError(
            f"methodology.html missing or thin ({meth.stat().st_size if meth.is_file() else 0}); "
            "run sovereign_cash.web.pages / methods_page"
        )
    hub_txt = hub.read_text(encoding="utf-8") if hub.is_file() else ""
    if "methodology.html" not in hub_txt or "How forecasts work" in hub_txt or 'data-g="equations"' in hub_txt:
        raise RuntimeError(
            "financing_dashboard.html is not the Methodology hub; "
            "rebuild via sovereign_cash.build_financing"
        )
    if eq.is_file():
        eq_txt = eq.read_text(encoding="utf-8")
        if "methodology.html" not in eq_txt and eq.stat().st_size > 5_000:
            raise RuntimeError(
                "equations.html must redirect to methodology.html (not a full page)"
            )


_OPTIONAL_DASHBOARD_STAGES = frozenset({"fytd", "scenarios"})


def _dashboard_only() -> int:
    """Rebuild hub / financing / methods / fytd / scenarios from warm derived."""
    failed: list[str] = []
    try:
        from sovereign_cash.fytd_page import build as build_fytd

        build_fytd()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failed.append("fytd")
    try:
        from sovereign_cash.methods_page import build as build_methods

        build_methods()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failed.append("methods")
    try:
        scenarios()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failed.append("scenarios")
    try:
        build(refresh=False)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failed.append("hub")
    try:
        build_financing()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failed.append("financing")
    try:
        _assert_methodology_hub()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failed.append("methodology_hub")
    if failed:
        print("failed:", ", ".join(failed))
    core_fail = [f for f in failed if f not in _OPTIONAL_DASHBOARD_STAGES]
    if core_fail:
        return 1
    opt_fail = [f for f in failed if f in _OPTIONAL_DASHBOARD_STAGES]
    if opt_fail:
        print("degraded (optional stages):", ", ".join(opt_fail))
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-fetch sources")
    ap.add_argument(
        "--dashboard-only",
        action="store_true",
        help="site Refresh: rebuild HTML from warm derived; skip country run() / debt fetch",
    )
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.dashboard_only:
        return _dashboard_only()
    failed = []
    try:
        # own bottom-up CPI paths: the price driver of every country's macro block (macro_authority); a country
        # that fails keeps its previous path, and macro_authority falls back to WEO CPI where none exists
        from sovereign_cash import own_inflation

        errs = own_inflation.refresh().get("_meta", {}).get("errors") or {}
        if errs:
            print("own_inflation errors:", errs)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failed.append("own_inflation")
    for cc in a.only or CC:
        try:
            mod = importlib.import_module(f"sovereign_cash.{cc}.run")
        except ModuleNotFoundError:
            continue
        try:
            mod.run(refresh=a.refresh)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            failed.append(cc)
    if not a.only:
        try:
            debt(a.refresh)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            failed.append("debt")
    try:
        from sovereign_cash.fytd_page import build as build_fytd
        build_fytd()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failed.append("fytd")
    try:
        from sovereign_cash.methods_page import build as build_methods
        build_methods()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failed.append("methods")
    try:
        scenarios()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failed.append("scenarios")
    build(refresh=a.refresh)
    build_financing()
    try:
        _assert_methodology_hub()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failed.append("methodology_hub")
    if failed:
        print("failed:", ", ".join(failed))
        core_fail = [f for f in failed if f not in _OPTIONAL_DASHBOARD_STAGES]
        if core_fail:
            return 1
        opt_fail = [f for f in failed if f in _OPTIONAL_DASHBOARD_STAGES]
        if opt_fail:
            print("degraded (optional stages):", ", ".join(opt_fail))
            return 2
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
