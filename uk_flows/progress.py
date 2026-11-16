"""Progress reporting for the bulk download loops, with or without tqdm installed.

Bulk download loops walk thousands of files, so they need to
show that they are alive. tqdm is used when it is available and a plain periodic line otherwise, which
keeps tqdm an optional dependency: the pipeline must run on a locked-down machine with nothing but
pandas and requests.

    with bar(total=len(jobs), label="day-files") as step:
        for job in jobs:
            ...
            step(f"{missing} empty")
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

PLAIN_UPDATES = 20          # lines printed by the fallback over a whole run


@contextmanager
def bar(total: int, label: str = "", unit: str = "file", enabled: bool = True):
    """Yield a `step(note)` callable that advances the display by one item."""
    if not enabled or total <= 0:
        yield lambda note="": None
        return

    if tqdm is not None:
        with tqdm(total=total, desc=label, unit=unit, leave=True,
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                             "[{elapsed}<{remaining}] {postfix}") as meter:
            def step(note: str = "") -> None:
                if note:
                    meter.set_postfix_str(note, refresh=False)
                meter.update(1)

            yield step
        return

    started = time.time()
    every = max(1, total // PLAIN_UPDATES)
    state = {"done": 0}

    def step(note: str = "") -> None:
        state["done"] += 1
        done = state["done"]
        if done % every and done != total:
            return
        elapsed = time.time() - started
        rate = done / elapsed if elapsed else 0
        remaining = (total - done) / rate if rate else 0
        print(f"  {label}: {done}/{total} ({done/total:.0%}), {elapsed/60:.1f}m elapsed, "
              f"{remaining/60:.1f}m left{', ' + note if note else ''}", flush=True)

    yield step
    print(f"  {label}: {total} done in {(time.time() - started)/60:.1f}m", flush=True)


def write(message: str) -> None:
    """Print without tearing the bar, for warnings raised mid-loop."""
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message, file=sys.stderr, flush=True)
