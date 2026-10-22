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
    affinities = {tuple(row["affinity"]) for row in rows}
    grids = {tuple(row["grid"]) for row in rows}
    cutoffs = {row["cutoff"] for row in rows}
    if any(len(group) != 1 for group in
           (hosts, cpus, affinities, grids, cutoffs)):
        raise ValueError(f"{system}: node, affinity, grid, or cutoff changed")
    by_variant = defaultdict(list)
    expected = {"m1", "m16", "d1", "d16", "g1", "g16", "x1", "x16", "p1"}
    for row in rows:
        if row["round"] > 0:
            by_variant[row["variant"]].append(row)
    if set(by_variant) != expected:
        raise ValueError(f"{system}: missing or unexpected variants: "
                         f"{sorted(expected ^ set(by_variant))}")
    kept_rounds = {row["round"] for row in rows if row["round"] > 0}
    if len(kept_rounds) != 3:
        raise ValueError(f"{system}: expected three kept rounds, got {sorted(kept_rounds)}")
    warmups = [row for row in rows if row["round"] == 0]
    if len(warmups) != len(expected) or {row["variant"] for row in warmups} != expected:
        raise ValueError(f"{system}: incomplete warm-up round")
    result = {}
    for variant, repeats in by_variant.items():
        if any(not row["ok"] or row.get("rate_ms") is None for row in repeats):
            raise ValueError(f"{system}: failed or untimed {variant} in a kept round")
        if len({row["round"] for row in repeats}) != len(repeats):
            raise ValueError(f"{system}: duplicate kept round of {variant}")
        if {row["round"] for row in repeats} != kept_rounds:
            raise ValueError(f"{system}: missing kept round of {variant}")
        rates = [row["rate_ms"] for row in repeats]
        result[variant] = (statistics.mean(rates),
                           statistics.stdev(rates) if len(rates) > 1 else 0,
                           len(rates),
                           {row["round"]: row["rate_ms"] for row in repeats})
    return next(iter(hosts)), next(iter(cpus)), result


def rate(stats, variant):
    if variant not in stats:
        return "—"
    mean, spread, n, _ = stats[variant]
    return f"{mean:.3f} ± {spread:.3f} ({n})"


def ratio(stats, variant, reference):
    if variant not in stats or reference not in stats:
        return "—"
    rates = stats[variant][3]
    refs = stats[reference][3]
    if set(rates) != set(refs):
        raise ValueError(f"unpaired rounds for {variant} and {reference}")
    paired = [rates[r] / refs[r] for r in sorted(rates)]
    return (f"{statistics.mean(paired):.3f} ± "
            f"{statistics.stdev(paired) if len(paired) > 1 else 0:.3f}")


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
