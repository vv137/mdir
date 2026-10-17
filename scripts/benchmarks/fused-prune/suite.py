#!/usr/bin/env python3
"""The Amber suite, main against the branch, alternated within one job (#362).

    suite.py WORK [SYSTEM ...]

MAIN_BIN and NEW_BIN name the two binaries of mdir (frozen copies), and
MDIR_BENCH_DIR the prepared suite (scripts/benchmarks/amber). Each system
runs with the settings of the suite (bench.py: groups with a dual list of
3 / 0.6 Å beyond the cutoff at 2 fs, in proportion to the step), twice the
suite's steps; ROUNDS rounds (default 4), the order of the binaries
reversed each round, the first round dropped. The rate is that of the
second half of each run ("from step ... ms per step"). Prints µs a step,
the mean and standard deviation of the rounds kept, for each binary, and
the ratio of the branch to main.
"""
import os
import re
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "amber"))
import bench  # noqa: E402

work = os.path.abspath(sys.argv[1])
names = sys.argv[2:] or list(bench.SYSTEMS)
rounds = int(os.environ.get("ROUNDS", 4))
binaries = {"main": os.environ["MAIN_BIN"], "new": os.environ["NEW_BIN"]}
suite = os.path.join(os.environ["MDIR_BENCH_DIR"], "Amber24_Benchmark_Suite")
os.makedirs(work, exist_ok=True)
link = os.path.join(work, "Amber24_Benchmark_Suite")
if not os.path.exists(link):
    os.symlink(suite, link)


def run(binary, path, w):
    shutil.rmtree(w, ignore_errors=True)
    os.makedirs(w)
    with open(os.path.join(w, "out.log"), "w") as out:
        subprocess.run([binary, "run", path], cwd=w, stdout=out, stderr=subprocess.STDOUT,
                       env=dict(os.environ, MDIR_COMPILE_CACHE="off"))
    log = open(os.path.join(w, "out.log")).read()
    m = re.search(r"from step [0-9]+ to step [0-9]+, ([0-9.]+) ms per step", log)
    pruned = re.search(r"pruned .*", log)
    return (float(m.group(1)) * 1000 if m else None), (pruned.group(0) if pruned else "")


subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm,power.draw", "--format=csv"])
summary = []
for name in names:
    system = bench.SYSTEMS[name]
    target = os.path.join(work, name)
    os.makedirs(target, exist_ok=True)
    path = bench.write_mdir(name, system, target, steps=2 * system["steps"], skin=3.0,
                            prune_skin=0.6, neighbor_structure="GROUPS")
    times = {v: [] for v in binaries}
    for r in range(rounds):
        for v in (list(binaries) if r % 2 == 0 else list(binaries)[::-1]):
            t, pruned = run(binaries[v], path, os.path.join(target, f"{v}-r{r}"))
            print(f"{name} round {r} {v}: {t} µs/step; {pruned}", flush=True)
            if r and t:
                times[v].append(t)
    m, n = (statistics.mean(times[v]) for v in binaries)
    sd = [statistics.stdev(times[v]) if len(times[v]) > 1 else 0.0 for v in binaries]
    summary.append(f"| {name} | {m:.1f} ± {sd[0]:.1f} | {n:.1f} ± {sd[1]:.1f} | {n / m:.3f} |")
print("| System | main (µs/step) | branch | branch / main |\n|---|---|---|---|")
print("\n".join(summary))
