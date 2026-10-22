#!/usr/bin/env python3
"""Measures a learned potential on liquid water of growing size (ML4's
baselines, docs/ml4-baselines.md): peak device memory against atoms and
pairs, and time per pair and per step, either through MDIR's model term
(`mdir run`, the plugin's allocator and `MDIR_MLIP_TIMING`) or through the
package's own ASE calculator in Python (`--package`, torch's allocator).

For each count of waters (29.9 A^3 each, water-box.py's `--liquid`), it
writes the system to OUT/w<count>/ and

- with an artifact: runs `mdir run` for `--steps` steps of NVE (0.5 fs, no
  classical term) and reads from the log the peak of the plugin's allocator,
  the pairs of the last call, and the time per step of the second half of
  the run (TorchScript's profiling executor specializes the model in the
  first evaluations, which the second half leaves out);
  `MDIR_MLIP_TIMING=1` adds the model's own time per evaluation, apart from
  its pair list, from the plugin's standard error, as a mean over every
  evaluation after the first;
- with `--stages MODEL.mlir` (a model in stages, `painn-stages.py export`,
  D294): runs `mdir run` with `backend = "STAGES"`, and reads
  the peak of MDIR's own allocator (`MDIR_DEVICE_MEMORY`), which holds the
  model's fields, the rows of its tiles, and its bodies' buffers; the pairs
  are counted within the cutoff (ASE's neighbor list). A run through the
  plugin reports MDIR's peak beside the plugin's;
- with `--package KIND:PATH` (`mace:MODEL`, `painn:CHECKPOINT`): evaluates
  the energy, the forces, and the stress with the package's calculator
  (`water-box.py --kind`), and reports `torch.cuda.max_memory_allocated` above
  the model's own, the pairs within the cutoff, and the shortest of three
  evaluations after a warm-up.

A size that fails (out of memory, or any error) is reported as such and ends
the sweep, since the larger ones would fail too. One JSON object per size is
printed, and the table at the end.

usage: model-sweep.py OUT --waters 8 16 32 ...
           (--artifact A.pt | --stages MODEL.mlir | --package KIND:PATH)
           [--precision mixed|double] [--steps 60] [--mdir mdir]
           [--stage-field-precision DOUBLE]
           [--seed 1]
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def water_box():
    spec = importlib.util.spec_from_file_location("water_box", os.path.join(HERE, "water-box.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = """[input]
topology    = "system.top"
coordinates = "system.gro"

[output]
energy_interval = {interval}

[energy]
cutoff = 0.2
pairlist_distance = 0.25
electrostatics = "CUTOFF"

[[energy.model]]
name = "model"
backend = "{backend}"
path = "{artifact}"
{stage_field_precision}

[dynamics]
integrator = "VELOCITY_VERLET"
time_step  = 0.0005
steps      = {steps}

[ensemble]
ensemble    = "NVE"
temperature = 300.0

[boundary]
type = "PERIODIC"

[execution]
target = "GPU"
precision = "{precision}"
"""


def run_mdir(directory, args):
    with open(os.path.join(directory, "run.toml"), "w") as f:
        f.write(RUN.format(steps=args.steps, interval=max(1, args.steps // 10),
                           artifact=os.path.abspath(args.artifact or args.stages),
                           backend="STAGES" if args.stages else "METATOMIC",
                           stage_field_precision=(
                               'stage_field_precision = "DOUBLE"'
                               if args.stage_field_precision else ''),
                           precision=args.precision))
    env = dict(os.environ, MDIR_MLIP_TIMING="1", MDIR_DEVICE_MEMORY="1")
    begin = time.perf_counter()
    p = subprocess.run([args.mdir, "run", "run.toml"], cwd=directory, env=env,
                       capture_output=True, text=True)
    wall = time.perf_counter() - begin
    with open(os.path.join(directory, "run.log"), "w") as f:
        f.write(p.stdout)
    with open(os.path.join(directory, "run.err"), "w") as f:
        f.write(p.stderr)
    out = {"wall_s": wall}
    text = p.stdout + p.stderr
    if p.returncode != 0:
        tail = [l for l in text.splitlines() if l.strip()][-3:]
        out["failed"] = " | ".join(tail)[:400]
        return out
    m = re.search(r"(\d+) pairs and (\d+) calls in the last; the plugin's allocator held at "
                  r"most ([\d.]+) MiB allocated, ([\d.]+) MiB reserved", text)
    if m:
        out.update(pairs=int(m.group(1)), calls=int(m.group(2)),
                   peak_allocated_MiB=float(m.group(3)), peak_reserved_MiB=float(m.group(4)))
    m = re.search(r"device memory \(MDIR_DEVICE_MEMORY\): peak ([\d.]+) MiB allocated, "
                  r"([\d.]+) MiB with the pool", text)
    if m:
        out.update(mdir_peak_MiB=float(m.group(1)), mdir_reserved_MiB=float(m.group(2)))
        if args.stages:
            out.update(peak_allocated_MiB=float(m.group(1)),
                       peak_reserved_MiB=float(m.group(2)))
    m = re.search(r"from step \d+ to step \d+, ([\d.]+) ms per step", text)
    if m:
        out["ms_per_step"] = float(m.group(1))
    # MDIR_MLIP_TIMING: the plugin's means per evaluation, after the first.
    for line in text.splitlines():
        m = re.search(r"\(MDIR_MLIP_TIMING\): \d+ evaluations after the first, ([\d.]+) us "
                      r"of the pair list \(\w+\) and ([\d.]+) us of the model", line)
        if m:
            out["pairlist_ms"] = float(m.group(1)) / 1e3
            out["model_ms"] = float(m.group(2)) / 1e3
    return out


def run_package(positions, cell, args):
    import torch
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    wb = water_box()
    wb.KIND = args.package.split(":", 1)[0]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    calc = wb.calculator(args.package.split(":", 1)[1], "cuda")
    held = torch.cuda.memory_allocated()
    atoms = Atoms(["O", "H", "H"] * (len(positions) // 3), positions=positions, cell=cell,
                  pbc=True)
    cutoff = 5.0 if wb.KIND == "painn" else float(calc.models[0].r_max)
    out = {"pairs": int(len(neighbor_list("i", atoms, cutoff)))}
    times = []
    try:
        for k in range(4):
            atoms.calc = calc
            calc.reset()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            atoms.get_potential_energy()
            atoms.get_forces()
            atoms.get_stress()
            torch.cuda.synchronize()
            if k:
                times.append(time.perf_counter() - begin)
    except RuntimeError as e:  # torch.OutOfMemoryError, or one raised through TorchScript
        if "out of memory" not in str(e):
            raise
        out["failed"] = "out of memory: " + [l for l in str(e).splitlines()
                                             if "out of memory" in l][0].strip()[:200]
        return out
    out.update(peak_allocated_MiB=(torch.cuda.max_memory_allocated() - held) / 1048576.0,
               peak_reserved_MiB=torch.cuda.max_memory_reserved() / 1048576.0,
               model_ms=min(times) * 1e3)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out")
    p.add_argument("--waters", type=int, nargs="+", required=True)
    p.add_argument("--artifact")
    p.add_argument("--stages")
    p.add_argument("--package")
    p.add_argument("--precision", default="mixed")
    p.add_argument("--stage-field-precision", choices=["DOUBLE"])
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--mdir", default="mdir")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if (bool(args.artifact) + bool(args.package) + bool(args.stages)) != 1:
        sys.exit("give one of --artifact, --stages, and --package")
    if args.stage_field_precision and not args.stages:
        sys.exit("--stage-field-precision is for --stages only")
    wb = water_box()
    rows = []
    for count in args.waters:
        directory = os.path.join(args.out, f"w{count}")
        os.makedirs(directory, exist_ok=True)
        positions, cell = wb.liquid(count, args.seed)
        positions = np.round(wb.wrap(positions, cell) / 10.0, 7) * 10.0
        wb.write(directory, positions, cell)
        row = {"waters": count, "atoms": 3 * count}
        row.update(run_mdir(directory, args) if args.artifact or args.stages
                   else run_package(positions, cell, args))
        if args.stages and "pairs" not in row:
            from ase import Atoms
            from ase.neighborlist import neighbor_list
            atoms = Atoms(["O", "H", "H"] * count, positions=positions, cell=cell, pbc=True)
            row["pairs"] = int(len(neighbor_list("i", atoms, 5.0)))
        if "pairs" in row and "peak_allocated_MiB" in row:
            row["kB_per_pair"] = row["peak_allocated_MiB"] * 1048.576 / row["pairs"]
        # Through mdir run, the time per step of the second half (the model's
        # mean includes the first evaluations, which TorchScript specializes);
        # through the package, its shortest evaluation.
        time_ms = row.get("ms_per_step") or (row.get("model_ms") if args.package else None)
        if time_ms and row.get("pairs"):
            row["us_per_pair"] = time_ms * 1e3 / row["pairs"]
        print(json.dumps(row), flush=True)
        rows.append(row)
        if "failed" in row:
            break
    print("| Atoms | Pairs | Peak allocated (MiB) | Reserved (MiB) | kB per pair | Model (ms) "
          "| µs per pair | Step (ms) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "failed" in r:
            print(f"| {r['atoms']} | {r.get('pairs', '')} | failed: {r['failed'][:80]} | | | | | |")
            continue
        print("| {atoms} | {pairs} | {a:.0f} | {b:.0f} | {c:.1f} | {d} | {e} | {f} |".format(
            atoms=r["atoms"], pairs=r.get("pairs", ""), a=r.get("peak_allocated_MiB", 0),
            b=r.get("peak_reserved_MiB", 0), c=r.get("kB_per_pair", 0),
            d=f"{r['model_ms']:.1f}" if "model_ms" in r else "",
            e=f"{r['us_per_pair']:.2f}" if "us_per_pair" in r else "",
            f=f"{r['ms_per_step']:.1f}" if "ms_per_step" in r else ""))


if __name__ == "__main__":
    main()
