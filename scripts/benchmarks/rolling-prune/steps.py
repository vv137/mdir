"""Whole steps and the pruning per step, variants against main (#357).

    steps.py SYSTEM VARIANTS [ROUNDS]

Real dynamics of the system's settings (groups, dual list), the variants
alternated round by round, round 0 dropped; µs per step from the log
("from step ... ms per step"). Then each variant once under nsys: the
kernels of the refreshes (prunings; builds apart) per step.
"""
import csv
import io
import os
import re
import shutil
import statistics
import subprocess
import sys

# The directory of the inputs (in/SYSTEM/m96.toml or bench.toml) and the
# frozen binaries; NSYS the profiler.
S = os.environ.get("ROLLING_PRUNE_DIR", os.getcwd())
NSYS = os.environ.get("NSYS", "nsys")
MAIN = os.environ.get("MAIN_BIN", S + "/frozen-main/bin/mdir")
EXP = os.environ.get("EXP_BIN", S + "/frozen-exp6/bin/mdir")
TABLE = {
    "main": (MAIN, {}),
    "skip": (EXP, {"MDIR_EXP_LOOP_SKIP": "1"}),
    "rolling": (EXP, {"MDIR_EXP_ROLLING": "1"}),
    "rollskip": (EXP, {"MDIR_EXP_ROLLING": "1", "MDIR_EXP_LOOP_SKIP": "1"}),
    "lanes2": (EXP, {"MDIR_EXP_PRUNE_LANES": "2"}),
    "lanes4": (EXP, {"MDIR_EXP_PRUNE_LANES": "4"}),
}
STEPS = {"tip3p": 2000, "opc": 2000, "jac": 10000, "ubq": 5000, "stmv": 1000}
name = sys.argv[1]
variants = sys.argv[2].split(",")
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
steps = STEPS[name]
src = f"{S}/in/{name}"
out = f"{S}/steps" + os.environ.get("OUT_SUFFIX", "") + f"/{name}"
os.makedirs(out, exist_ok=True)
text = open(f"{src}/{'bench' if name == 'ubq' else 'm96'}.toml").read()
if name != "stmv":
    for k in ("topology", "coordinates", "checkpoint"):
        text = re.sub(rf'{k}\s*=\s*"(?:[^"]*/)?([^"/]+)"', f'{k} = "{src}/\\1"', text)
text = re.sub(r"(?m)^steps\s*=\s*\d+", f"steps      = {steps}", text)
text = re.sub(r"energy_interval\s*=\s*\d+", f"energy_interval = {steps // 2}", text)
open(f"{out}/run.toml", "w").write(text)


def run(v, tag, nsys=False):
    binary, extra = TABLE[v]
    w = f"{out}/{v}-{tag}"
    shutil.rmtree(w, ignore_errors=True)
    os.makedirs(w)
    cmd = [binary, "run", f"{out}/run.toml"]
    if nsys:
        cmd = [NSYS, "profile", "-t", "cuda", "-o", f"{w}/prof", "--force-overwrite", "true"] + cmd
    with open(f"{w}/out.log", "w") as o:
        subprocess.run(cmd, cwd=w, env=dict(os.environ, MDIR_COMPILE_CACHE="off", **extra),
                       stdout=o, stderr=subprocess.STDOUT)
    log = open(f"{w}/out.log").read()
    m = re.search(r"from step [0-9]+ to step [0-9]+, ([0-9.]+) ms per step", log)
    return w, (float(m.group(1)) * 1000 if m else None)


subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm,power.draw", "--format=csv"])
times = {v: [] for v in variants}
for r in range(rounds):
    for v in (variants if r % 2 == 0 else variants[::-1]):
        _, t = run(v, f"r{r}")
        print(f"{name} round {r} {v}: {t} µs/step", flush=True)
        if r and t:
            times[v].append(t)
for v in variants:
    w, _ = run(v, "nsys", nsys=True)
    r = subprocess.run([NSYS, "stats", "-r", "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum", "-f", "csv", "-q",
                        f"{w}/prof.nsys-rep"], capture_output=True, text=True).stdout
    parts = r.split("Time (%)")
    kern = list(csv.DictReader(io.StringIO("Time (%)" + parts[1])))
    prune = sum(float(k["Total Time (ns)"]) for k in kern if "refresh_neighbors" in k["Name"])
    build = sum(float(k["Total Time (ns)"]) for k in kern if "build_neighbors" in k["Name"])
    loop = sum(float(k["Total Time (ns)"]) for k in kern if "pair_for" in k["Name"])
    total = sum(float(k["Total Time (ns)"]) for k in kern)
    d2d = 0.0
    if len(parts) > 2:
        mem = list(csv.DictReader(io.StringIO("Time (%)" + parts[2])))
        d2d = sum(float(m["Total Time (ns)"]) for m in mem if "Device-to-Device" in m["Operation"])
    log = open(f"{w}/out.log").read()
    print(f"{name} {v} nsys per step (µs): prunings and their kernels {prune / steps / 1e3:.1f}, "
          f"D2D copies {d2d / steps / 1e3:.1f}, builds {build / steps / 1e3:.1f}, loops over pairs "
          f"{loop / steps / 1e3:.1f}, all kernels {total / steps / 1e3:.1f}; "
          + "; ".join(l for l in log.splitlines() if re.search("pruned|built", l)), flush=True)
for v in variants:
    t = times[v]
    print(f"{name} {v}: {statistics.mean(t):.1f} ± {statistics.stdev(t) if len(t) > 1 else 0:.1f} µs/step "
          f"(n={len(t)})", flush=True)
subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm,power.draw", "--format=csv"])
