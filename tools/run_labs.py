"""Run every lab (or a subset) and report what passed.

    python tools/run_labs.py              # everything
    python tools/run_labs.py --part 3     # only Part III
    python tools/run_labs.py --module 20  # only module 20
    python tools/run_labs.py --gpu        # also run labs marked as GPU-only

Each lab is a standalone Python script. It passes if it exits with status 0, which means every assertion inside
held. Labs that need a GPU start with the line `# requires: gpu` and are skipped unless you pass --gpu.
Labs that need heavy optional dependencies declare them with `# requires: <package>` and are skipped, loudly,
when the package is missing.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def requirements(lab: Path) -> list[str]:
    reqs = []
    for line in lab.read_text(encoding="utf-8").splitlines()[:15]:
        if line.startswith("# requires:"):
            reqs += [r.strip() for r in line.split(":", 1)[1].split(",") if r.strip()]
    return reqs


def missing(reqs: list[str], gpu: bool) -> list[str]:
    out = []
    for r in reqs:
        if r == "gpu":
            if not gpu:
                out.append("gpu")
        elif importlib.util.find_spec(r) is None:
            out.append(r)
    return out


def main() -> int:
    # One line per lab, printed as soon as the lab finishes. Without line buffering, stdout to a pipe (as in CI) holds
    # everything until the process exits, and a job that runs out of time shows nothing at all.
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, nargs="+", help="one or more part numbers")
    ap.add_argument("--module")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    labs = sorted(ROOT.glob("part-*/*/labs/lab_*.py"))
    if args.part is not None:
        labs = [l for l in labs if any(l.parts[len(ROOT.parts)].startswith(f"part-{p}-") for p in args.part)]
    if args.module is not None:
        labs = [l for l in labs if l.parent.parent.name.startswith(f"{int(args.module):02d}-")]

    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")
    failed, skipped = [], []
    for lab in labs:
        name = lab.relative_to(ROOT).as_posix()
        miss = missing(requirements(lab), args.gpu)
        if miss:
            skipped.append(name)
            print(f"SKIP  {name}  (needs {', '.join(miss)})")
            continue
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(lab)], cwd=lab.parent, env=env,
                capture_output=True, text=True, timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            failed.append(name)
            print(f"FAIL  {name}  (timed out after {args.timeout}s)")
            continue
        dt = time.time() - t0
        if proc.returncode == 0:
            print(f"PASS  {name}  ({dt:.1f}s)")
        else:
            failed.append(name)
            print(f"FAIL  {name}  ({dt:.1f}s)")
            print(proc.stdout[-3000:])
            print(proc.stderr[-3000:])

    print(f"\n{len(labs) - len(failed) - len(skipped)} passed, {len(failed)} failed, {len(skipped)} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
