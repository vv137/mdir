"""Whole steps and the kernels of the loops and prunings, variants alternated (#362).

    steps.py SYSTEM VARIANTS [ROUNDS] [NSYS=0|1]

VARIANTS among main (MAIN_BIN), off (NEW_BIN with MDIR_FUSED_PRUNE=off), on
(NEW_BIN with MDIR_FUSED_PRUNE=always), default (NEW_BIN as it is). The
inputs are in FUSED_PRUNE_DIR/in/SYSTEM (m96.toml, or bench.toml for ubq,
with the files it names), as in scripts/benchmarks/rolling-prune. Round 0
dropped; µs per step from the log ("from step ... ms per step"). Then each
variant once under nsys (NSYS the profiler): per kernel of the loops over
pairs and of the refreshes, the calls and the mean, and the sums per step.
"""
import csv
import io
import os
import re
import shutil
import statistics
import subprocess
import sys

S = os.environ.get("FUSED_PRUNE_DIR", os.getcwd())
NSYS = os.environ.get("NSYS", "nsys")
MAIN = os.environ.get("MAIN_BIN", S + "/frozen-main/bin/mdir")
NEW = os.environ.get("NEW_BIN", S + "/frozen-new/bin/mdir")
TABLE = {
    "main": (MAIN, {}),
    "off": (NEW, {"MDIR_FUSED_PRUNE": "off"}),
    "on": (NEW, {"MDIR_FUSED_PRUNE": "always"}),
    "default": (NEW, {}),
}
STEPS = {"tip3p": 2000, "opc": 2000, "jac": 10000, "ubq": 5000, "stmv": 1000, "cel": 3000,
         "w8k": 6000, "w16k": 4000, "w32k": 3000, "w64k": 2000}
name = sys.argv[1]
variants = sys.argv[2].split(",")
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
profile = (sys.argv[4] != "0") if len(sys.argv) > 4 else True
steps = int(os.environ.get("STEPS", STEPS[name]))
src = f"{S}/in/{name}"
out = f"{S}/prof" + os.environ.get("OUT_SUFFIX", "") + f"/{name}"
os.makedirs(out, exist_ok=True)
text = open(f"{src}/{'bench' if name == 'ubq' else 'm96'}.toml").read()
if name not in ("stmv", "cel"):
    for k in ("topology", "coordinates", "checkpoint"):
        text = re.sub(rf'{k}\s*=\s*"(?:[^"]*/)?([^"/]+)"', f'{k} = "{src}/\\1"', text)
text = re.sub(r"(?m)^steps\s*=\s*\d+", f"steps      = {steps}", text)
text = re.sub(r"energy_interval\s*=\s*\d+", f"energy_interval = {steps // 2}", text)
for key in ("pairlist_distance", "pruned_distance"):
    if os.environ.get(key.upper()):
        text = re.sub(rf"(?m)^{key}\s*=.*", f"{key} = {os.environ[key.upper()]}", text)
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
for v in variants if profile else []:
    w, _ = run(v, "nsys", nsys=True)
    r = subprocess.run([NSYS, "stats", "-r", "cuda_gpu_kern_sum", "-f", "csv", "-q",
                        f"{w}/prof.nsys-rep"], capture_output=True, text=True).stdout
    kern = list(csv.DictReader(io.StringIO(r[r.index("Time (%)"):])))
    log = open(f"{w}/out.log").read()
    pick = [k for k in kern if re.search("pair_for|refresh_neighbors", k["Name"])]
    for k in sorted(pick, key=lambda k: -float(k["Total Time (ns)"])):
        print(f"  {v} {k['Name'][:60]:60s} calls {k['Instances']:>7s} mean {float(k['Avg (ns)']) / 1e3:8.1f} µs "
              f"per step {float(k['Total Time (ns)']) / steps / 1e3:8.1f} µs", flush=True)
    loop = sum(float(k["Total Time (ns)"]) for k in kern if "pair_for" in k["Name"])
    prune = sum(float(k["Total Time (ns)"]) for k in kern if "refresh_neighbors" in k["Name"])
    build = sum(float(k["Total Time (ns)"]) for k in kern if "build_neighbors" in k["Name"])
    total = sum(float(k["Total Time (ns)"]) for k in kern)
    print(f"{name} {v} nsys per step (µs): loops over pairs {loop / steps / 1e3:.1f}, refreshes "
          f"{prune / steps / 1e3:.1f}, both {(loop + prune) / steps / 1e3:.1f}, builds {build / steps / 1e3:.1f}, "
          f"all kernels {total / steps / 1e3:.1f}; "
          + "; ".join(l for l in log.splitlines() if re.search("pruned|built", l)), flush=True)
for v in variants:
    t = times[v]
    if t:
        print(f"{name} {v}: {statistics.mean(t):.1f} ± {statistics.stdev(t) if len(t) > 1 else 0:.1f} µs/step "
              f"(n={len(t)})", flush=True)
subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm,power.draw", "--format=csv"])
