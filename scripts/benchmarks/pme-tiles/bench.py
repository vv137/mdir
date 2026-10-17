#!/usr/bin/env python3
"""The step of the Amber suite with the spreading of PME of two builds and
of each layout (D[pme-tiles], docs/pme-m1.md, Section 3.3), alternated in
rounds within one job (the first round dropped), then one run of each
under Nsight Systems for the kernels of PME.

    bench.py MAIN BRANCH OUT SYSTEMS VARIANTS ROUNDS [SCALE] [PRECISION]

MAIN and BRANCH are `mdir` programs (copies that no build replaces during
the job); OUT a directory for the control files, logs, and results.json;
SYSTEMS the names of the suite (`jac_nve`, `cellulose_nve`, ...), which
`MDIR_BENCH_DIR` holds as `scripts/benchmarks/amber/bench.py prepare`
made it; VARIANTS a comma-separated list of `main` (MAIN as it is) or a
value of MDIR_PME_SPREAD for BRANCH (`global`, `tiles8`, ...), each
optionally with `@ORDER` (the order of the B-splines, the grid of the
suite kept) or `@prolate` (the prolate kernel at its default, D266, with
its own grid); SCALE multiplies the steps of the suite; PRECISION is
MIXED by default. The rate is taken from the ns/day of the log, which has
more digits than its ms per step, and printed as ms per step. `nsys` is
taken from the path; without it the kernels are not profiled. Run on an
idle device; a comparison of rates is one job on one node."""
import csv
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

main_mdir, branch_mdir, out_dir, systems, variants = sys.argv[1:6]
rounds = int(sys.argv[6])
scale = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
precision = sys.argv[8] if len(sys.argv) > 8 else "MIXED"
suite = os.environ["MDIR_BENCH_DIR"]
nsys = shutil.which("nsys")
os.makedirs(out_dir, exist_ok=True)
subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,power.limit,temperature.gpu",
                "--format=csv"])


def control(system, variant):
    text = open(os.path.join(suite, system, "mdir.toml")).read()
    text = text.replace('"../Amber24_Benchmark_Suite', '"' + suite + "/Amber24_Benchmark_Suite")
    steps = int(re.search(r"^steps\s*=\s*(\d+)", text, re.M).group(1))
    steps = max(100, int(steps * scale) // 100 * 100)
    text = re.sub(r"^steps\s*=\s*\d+", f"steps      = {steps}", text, flags=re.M)
    text = re.sub(r"^energy_interval\s*=\s*\d+", f"energy_interval = {steps}", text, flags=re.M)
    text = re.sub(r'^precision\s*=\s*"\w+"', f'precision = "{precision}"', text, flags=re.M)
    layout, _, order = variant.partition("@")
    if order == "prolate":
        for key in ("beta", "grid", "order", "influence"):
            text = re.sub(rf"^{key}\s*=.*\n", "", text, flags=re.M)
        text = text.replace("[pme]\n", '[pme]\nkernel = "PROLATE"\n')
    elif order:
        text = re.sub(r"^order\s*=\s*\d+", f"order     = {order}", text, flags=re.M)
    directory = os.path.join(out_dir, system, variant)
    os.makedirs(directory, exist_ok=True)
    open(os.path.join(directory, "mdir.toml"), "w").write(text)
    step = float(re.search(r"^time_step\s*=\s*([0-9.]+)", text, re.M).group(1))
    return directory, steps, step


def run(system, variant, prefix=()):
    directory, steps, step = control(system, variant)
    layout = variant.partition("@")[0]
    env = dict(os.environ)
    if layout != "main":
        env["MDIR_PME_SPREAD"] = layout
    program = main_mdir if layout == "main" else branch_mdir
    p = subprocess.run(list(prefix) + [program, "run", "mdir.toml"], cwd=directory, env=env,
                       capture_output=True, text=True)
    text = p.stdout + p.stderr
    open(os.path.join(directory, "run.log"), "a").write(text)
    rates = re.findall(r"([0-9.]+) ns per day", text)
    ms = step * 86400.0 / float(rates[-1]) if rates and p.returncode == 0 else None
    return ms, steps, directory


results = {}
for system in systems.split(","):
    names = variants.split(",")
    times = {v: [] for v in names}
    for k in range(rounds):
        for v in names:
            ms, steps, _ = run(system, v)
            print("round", k, system, v, ms if ms else "FAILED", flush=True)
            if ms and k > 0:
                times[v].append(ms)
    for v in names:
        t = times[v]
        entry = dict(system=system, variant=v, ms=t,
                     mean=statistics.mean(t) if t else None,
                     sd=statistics.stdev(t) if len(t) > 1 else None)
        if nsys:
            directory, steps, _ = control(system, v)
            report = os.path.join(directory, "profile")
            run(system, v, prefix=[nsys, "profile", "--force-overwrite=true", "-t", "cuda", "-o", report])
            stats = subprocess.run([nsys, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                                    "--force-export=true", report + ".nsys-rep"],
                                   capture_output=True, text=True).stdout
            body = stats[stats.find("Time (%)"):] if "Time (%)" in stats else ""
            pme = {}
            for row in csv.DictReader(io.StringIO(body)):
                name = row.get("Name", "")
                if "pme" in name:
                    pme[name] = float(row.get("Total Time (ns)", 0) or 0) / 1000.0 / steps
            entry["pme_us_per_step"] = pme
        print(json.dumps(dict(system=system, variant=v, mean_ms=entry["mean"], sd=entry["sd"],
                              pme={n: round(x, 2) for n, x in sorted(entry.get("pme_us_per_step", {}).items())})),
              flush=True)
        results.setdefault(system, []).append(entry)
    json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
