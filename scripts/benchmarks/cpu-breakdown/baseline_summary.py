"""Summarize the NVE CPU baseline of issue #552.

    CPU_BENCH_DIR=... python baseline_summary.py TAG SYSTEM[@TAG]...

Round zero is warm-up. A missing or failed variant is shown, never silently
replaced by another round or system. Ratios divide MDIR ms/step by the
reference ms/step; a ratio above one means MDIR took longer.
"""

import json
import os
import statistics
import sys
from collections import defaultdict


def read(system, tag):
    path = os.path.join(os.environ["CPU_BENCH_DIR"], system, "runs", tag,
                        "results.jsonl")
    with open(path) as file:
        rows = [json.loads(line) for line in file]
    hosts = {row["host"] for row in rows}
    cpus = {row["cpu"] for row in rows}
    if len(hosts) != 1 or len(cpus) != 1:
        raise ValueError(f"{system}: rounds used different hosts or CPUs")
    by_variant = defaultdict(list)
    for row in rows:
        if row["round"] > 0:
            by_variant[row["variant"]].append(row)
    result = {}
    for variant, repeats in by_variant.items():
        if any(not row["ok"] or row.get("rate_ms") is None for row in repeats):
            raise ValueError(f"{system}: failed or untimed {variant} in a kept round")
        rates = [row["rate_ms"] for row in repeats]
        result[variant] = (statistics.mean(rates),
                           statistics.stdev(rates) if len(rates) > 1 else 0,
                           len(rates))
    return next(iter(hosts)), next(iter(cpus)), result


def rate(stats, variant):
    if variant not in stats:
        return "—"
    mean, spread, n = stats[variant]
    return f"{mean:.3f} ± {spread:.3f} ({n})"


def ratio(stats, variant, reference):
    if variant not in stats or reference not in stats:
        return "—"
    return f"{stats[variant][0] / stats[reference][0]:.3f}"


def main():
    tag, *systems = sys.argv[1:]
    if not systems:
        raise SystemExit("usage: baseline_summary.py TAG SYSTEM[@TAG]...")
    print("| System | Host | Variant | ms/step, mean ± sample SD (kept rounds) |")
    print("|---|---|---|---:|")
    data = {}
    for named in systems:
        system, run_tag = (named.split("@", 1) if "@" in named else
                           (named, tag))
        host, cpu, stats = read(system, run_tag)
        data[system] = stats
        for variant in sorted(stats):
            print(f"| {system} | {host} ({cpu}) | {variant} | {rate(stats, variant)} |")
    print("\n| System | MDIR mixed / GROMACS default, 1t | "
          "MDIR mixed / GROMACS exact, 1t | "
          "MDIR mixed / GROMACS default, 16t | "
          "MDIR mixed / GROMACS exact, 16t | "
          "MDIR double / serial pmemd, 1t |")
    print("|---|---:|---:|---:|---:|---:|")
    for system, stats in data.items():
        cells = [ratio(stats, "m1", "g1"), ratio(stats, "m1", "x1"),
                 ratio(stats, "m16", "g16"), ratio(stats, "m16", "x16"),
                 ratio(stats, "d1", "p1")]
        print("| " + system + " | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
