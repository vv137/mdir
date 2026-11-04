#!/usr/bin/env python3
"""Runs the K-Cl osmotic-pressure fit of #140 with `mdir run`.

    run.py equilibrate <system> [--replicas 8] [--nvt-ps 100] [--npt-ps 1000]
    run.py sample      <system> --name NAME --sigma S [--ns 2.5] [--start npt]
    run.py analyze     <system> --name NAME --sigma S [--drop 500]
    run.py fit         <system> [--updates 3] [--ns 2.5] [--drop 500]

<system> is a directory made by build.py (system.prmtop, system.inpcrd).
Each replica runs in <system>/rep<i>, from the stage templates beside this
script: 1-min.toml (once, copied to every replica), 2-nvt.toml and
3-npt.toml with the replica's seeds, then 4-md.toml at sigma'. The walls,
the slab, and the area of a replica are those of the cell its stage 3 ends
with (cell.txt). `fit` samples at sigma0, then moves sigma' to the root of
f = <Pi> - Pi_exp reweighted from the rows of the last runs, at most to
the edge of the region where N_eff/N >= 0.5, samples again there, each
replica from its last state, and stops when |f| of a fresh set of runs is
below its standard error, or after --updates updates.

Options common to all: --mdir (the binary, `mdir` on PATH by default),
--cpu (the CPU target, with --threads), --parallel (runs at once, the
number of replicas by default).
"""
import argparse
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import osmotic  # noqa: E402

PS_PER_STEP = 0.002


def render(template, out, values):
    text = (HERE / template).read_text()
    for key, value in values.items():
        text = text.replace(f"@{key}@", str(value))
    left = sorted(set(re.findall(r"@[A-Z0-9_]+@", text)))
    if left:
        raise SystemExit(f"{template}: unfilled {left}")
    pathlib.Path(out).write_text(text)


def common(args, stage, cell=None, sigma=osmotic.SIG0):
    """The values of the placeholders shared by the stages."""
    lz = cell[2] if cell else 96.0
    x12, x6 = osmotic.x_of(sigma)
    dyn = stage != "min"
    return dict(
        X12=repr(x12), X6=repr(x6), SIGMA=f"{sigma:.5f}",
        SCALING="NONE" if stage == "md" else "CELL",
        ZLO=f"{lz / 4:.6f}", ZHI=f"{3 * lz / 4:.6f}", ZC=f"{lz / 2:.6f}",
        OBSERVE_NBFIX='observe    = ["x12", "x6"]' if dyn else "",
        OBSERVE_WALL='observe    = ["z0"]' if dyn else "",
        OBSERVE_SLAB='observe    = ["s"]' if dyn else "",
        TARGET="CPU" if args.cpu else "GPU",
        THREADS=f"threads   = {args.threads}" if args.cpu else "")


def mdir(args, directory, control):
    log = pathlib.Path(directory) / (pathlib.Path(control).stem + ".out")
    with open(log, "w") as out:
        status = subprocess.run([args.mdir, "run", control], cwd=directory, stdout=out, stderr=subprocess.STDOUT).returncode
    if status != 0:
        raise SystemExit(f"{directory}/{control} failed with {status}; see {log}")


def in_parallel(args, jobs):
    with ThreadPoolExecutor(max_workers=args.parallel or len(jobs)) as pool:
        for f in [pool.submit(*job) for job in jobs]:
            f.result()


def replicas(args):
    n = args.replicas
    if n is None:      # 8 for a new equilibration; afterwards, the replicas that have a cell
        n = 8 if args.command == "equilibrate" else len(list(pathlib.Path(args.system).glob("rep*/cell.txt")))
    return [pathlib.Path(args.system) / f"rep{i}" for i in range(1, n + 1)]


def cell_of(rep):
    return [float(v) for v in (rep / "cell.txt").read_text().split()]


def equilibrate(args):
    reps = replicas(args)
    first = reps[0]
    first.mkdir(parents=True, exist_ok=True)
    render("1-min.toml", first / "1-min.toml", common(args, "min"))
    print("== 1-min")
    mdir(args, first, "1-min.toml")
    nvt, npt = round(args.nvt_ps / PS_PER_STEP), round(args.npt_ps / PS_PER_STEP)
    for i, rep in enumerate(reps, 1):
        rep.mkdir(parents=True, exist_ok=True)
        if rep != first:
            shutil.copy(first / "min.h5", rep / "min.h5")
        render("2-nvt.toml", rep / "2-nvt.toml", dict(common(args, "nvt"), SEED=1000 * i + 11,
                                                      NVT_STEPS=nvt, NVT_ROWS=math.gcd(nvt, 5000)))
        render("3-npt.toml", rep / "3-npt.toml", dict(common(args, "npt"), SEED=1000 * i + 12,
                                                      NPT_STEPS=npt, NPT_ROWS=math.gcd(npt, 5000)))
    print(f"== 2-nvt and 3-npt, {len(reps)} replicas")
    in_parallel(args, [(lambda r: (mdir(args, r, "2-nvt.toml"), mdir(args, r, "3-npt.toml")), r) for r in reps])
    pressures = []
    for rep in reps:
        rows = [l.split() for l in (rep / "npt.energy").read_text().splitlines() if not l.startswith("#")]
        area, volume = float(rows[-1][9]), float(rows[-1][8])
        edge = math.sqrt(area)
        (rep / "cell.txt").write_text(f"{edge:.6f} {edge:.6f} {volume / area:.6f}\n")
        tail = rows[len(rows) // 5:]
        pressures.append(sum(float(r[7]) for r in tail) / len(tail))
    mean = sum(pressures) / len(pressures)
    sd = math.sqrt(sum((p - mean) ** 2 for p in pressures) / max(len(pressures) - 1, 1))
    print(f"   cells: {[cell_of(r) for r in reps]}")
    print(f"   reported pressure of stage 3 after its first fifth: {mean:.1f} +- {sd / math.sqrt(len(pressures)):.1f} atm")


def sample(args, name=None, sigma=None, start=None):
    name, sigma, start = name or args.name, sigma if sigma is not None else args.sigma, start or args.start
    steps = round(args.ns * 1000 / PS_PER_STEP)
    reps = replicas(args)
    for i, rep in enumerate(reps, 1):
        seed = 100 * (zlib.crc32(name.encode()) % 10000) + i
        render("4-md.toml", rep / f"{name}.toml", dict(common(args, "md", cell_of(rep), sigma), START=f"{start}.h5",
                                                       NAME=name, STEPS=steps, SEED=seed))
    print(f"== {name}: sigma' = {sigma:.5f} A, {len(reps)} replicas x {args.ns} ns from {start}.h5")
    in_parallel(args, [(mdir, args, r, f"{name}.toml") for r in reps])


def analyze(args, name=None, sigma=None):
    name, sigma = name or args.name, sigma if sigma is not None else args.sigma
    return osmotic.analyze([str(r / name) for r in replicas(args)], sigma, args.drop)


def fit(args):
    record = pathlib.Path(args.system) / "fit.json"
    history, sigma, name, start = [], osmotic.SIG0, "s0", "npt"
    for update in range(args.updates + 1):
        sample(args, name, sigma, start)
        result = analyze(args, name, sigma)
        history.append(dict(name=name, sigma=sigma, **result))
        record.write_text(json.dumps(history, indent=1))
        if abs(result["f"]) < result["f_se"]:
            print(f"== |f| below its SE at sigma' = {sigma:.5f} A: stop")
            break
        if update == args.updates:
            print("== no more updates")
            break
        sigma, start, name = result["next"], name, f"u{update + 1}"
    print(f"   history in {record}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["equilibrate", "sample", "analyze", "fit"])
    ap.add_argument("system")
    ap.add_argument("--replicas", type=int, default=None,
                    help="8 for equilibrate; afterwards, the replicas that the system has")
    ap.add_argument("--nvt-ps", type=float, default=100.0)
    ap.add_argument("--npt-ps", type=float, default=1000.0)
    ap.add_argument("--name", default="s0")
    ap.add_argument("--sigma", type=float, default=osmotic.SIG0)
    ap.add_argument("--start", default="npt", help="the checkpoint the production starts from, without .h5")
    ap.add_argument("--ns", type=float, default=2.5, help="ns of each production run")
    ap.add_argument("--drop", type=float, default=500.0, help="ps dropped at the start of each run")
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--mdir", default=os.environ.get("MDIR", "mdir"))
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=0)
    args = ap.parse_args()
    dict(equilibrate=equilibrate, sample=sample, analyze=analyze, fit=fit)[args.command](args)


if __name__ == "__main__":
    main()
