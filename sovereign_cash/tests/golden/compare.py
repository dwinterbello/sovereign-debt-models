"""Compare a fresh capture with the frozen goldens.

    PYTHONPATH=. .venv/bin/python sovereign_cash/tests/golden/compare.py [--tol 1e-9]

Captures into a temporary folder, then reports every number that moved by more
than ``tol`` (absolute, in the block's own units). Exit 0 when nothing moved.
If input hashes differ, the data changed and a mismatch is not a code fault.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capture  # noqa: E402


def _walk(a, b, path, tol, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'golden' if k in a else 'new'}")
                continue
            _walk(a[k], b[k], f"{path}/{k}", tol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} -> {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", tol, out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(float(a) - float(b)) > tol:
            out.append(f"{path}: {a!r} -> {b!r}")
    elif a != b:
        out.append(f"{path}: {str(a)[:80]!r} -> {str(b)[:80]!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--golden", type=Path, default=HERE / "data")
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        capture.main(["--out", tmp])
        new = Path(tmp)
        old_in = json.loads((a.golden / "inputs.json").read_text())
        new_in = json.loads((new / "inputs.json").read_text())
        changed = sorted(k for k in set(old_in) | set(new_in) if old_in.get(k) != new_in.get(k))
        if changed:
            print(f"NOTE: {len(changed)} input files differ from the golden run (data, not code):")
            for k in changed[:20]:
                print("   ", k)
        diffs = []
        for f in sorted(a.golden.glob("*.json")):
            if f.name in ("inputs.json", "status.json"):
                continue
            g = new / f.name
            if not g.exists():
                diffs.append(f"{f.name}: not produced")
                continue
            _walk(json.loads(f.read_text()), json.loads(g.read_text()), f.stem, a.tol, diffs)
    by: dict[str, int] = {}
    for d in diffs:
        k = "/".join(d.split(":")[0].split("/")[:4])
        by[k] = by.get(k, 0) + 1
    for k, n in sorted(by.items()):
        print(f"MOVED {n:5d}  {k}")
    for d in diffs[:200]:
        print("DIFF", d)
    print(f"{len(diffs)} differences above {a.tol}")
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
