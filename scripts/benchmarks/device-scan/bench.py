#!/usr/bin/env python3
"""Compare two group-scan lowerings on prepared Amber systems.

    bench.py MAIN BRANCH OUTPUT SYSTEM[,SYSTEM...] ROUNDS [STEPS]

Set MDIR_BENCH_DIR to the prepared Amber suite. The two binaries run in
alternating order on the same idle GPU. The first round warms each binary;
the remaining rounds give the median rate. Each run has its own control file
and output directory. No build may replace either binary during the job.
STEPS may shorten a profiling run without changing other controls.
"""

import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

main, branch, out, systems, rounds = sys.argv[1:6]
suite = Path(os.environ["MDIR_BENCH_DIR"]).resolve()
out = Path(out).resolve()
rounds = int(rounds)
profile_steps = sys.argv[6] if len(sys.argv) > 6 else None


def prepare(system, variant):
    source = suite / system / "mdir.toml"
    text = source.read_text()
    text = text.replace('"../Amber24_Benchmark_Suite',
                        '"' + str(suite / "Amber24_Benchmark_Suite"))
    if profile_steps is not None:
        steps = profile_steps
        text = re.sub(r"^steps\s*=\s*\d+", "steps = " + steps,
                      text, count=1, flags=re.M)
        text = re.sub(r"^energy_interval\s*=\s*\d+", "energy_interval = " + steps,
                      text, count=1, flags=re.M)
    target = out / system / variant
    target.mkdir(parents=True, exist_ok=True)
    (target / "mdir.toml").write_text(text)
    step = float(re.search(r"^time_step\s*=\s*([0-9.]+)", text, re.M).group(1))
    return target, step


results = {}
for system in systems.split(","):
    results[system] = {"main": [], "branch": []}
    for round_number in range(rounds):
        for variant, binary in (("main", main), ("branch", branch)):
            target, step = prepare(system, variant)
            run = subprocess.run([binary, "run", "mdir.toml"], cwd=target,
                                 capture_output=True, text=True)
            log = run.stdout + run.stderr
            (target / f"round-{round_number}.log").write_text(log)
            rates = re.findall(r"([0-9.]+) ns per day", log)
            if run.returncode or not rates:
                print(system, variant, round_number, "FAILED", flush=True)
                sys.exit(run.returncode or 1)
            rate = float(rates[-1])
            ms = step * 86400.0 / rate
            print(system, variant, round_number, f"{ms:.6f} ms/step",
                  flush=True)
            if round_number or rounds == 1:
                results[system][variant].append(ms)

for system, rows in results.items():
    a = statistics.median(rows["main"])
    b = statistics.median(rows["branch"])
    print(system, "median ms/step", f"main={a:.6f}",
          f"branch={b:.6f}", f"ratio={b/a:.4f}")
