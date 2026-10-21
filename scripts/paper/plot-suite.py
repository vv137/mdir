#!/usr/bin/env python3
"""Plots the rates of MDIR and of pmemd.cuda over the Amber suite (Figure
10.1) from the runs of run-suite-repeats.sh: the mean over the repeats,
with the sample standard deviation as an error bar, and the mean of the
ratios of the repeats above each group.

    plot-suite.py LOGS OUTPUT.png [DEVICE]

With LOGS/runs.jsonl (the protocol of mdin.GPU, D[suite-symmetric]) the
bars are pmemd.cuda, MDIR with the searched reaches, and MDIR with the
default reaches, all with the output of mdin.GPU, and the ratio is that of
the searched reaches. Without it LOGS is that of the protocol before
(mdir-N.log and pmemd-N.log), with two bars.

DEVICE names the device in the title ("one RTX 4090 (450 W)" by default).

Needs matplotlib.
"""
import glob
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYSTEMS = ["jac_nve", "jac_nve_4fs", "jac_npt", "jac_npt_4fs", "factorix_nve",
           "factorix_npt", "cellulose_nve", "cellulose_npt", "stmv_npt_4fs"]
ATOMS = {"jac": 23558, "factorix": 90906, "cellulose": 408609, "stmv": 1067095}


def rates(path):
    found = {}
    for line in open(path):
        fields = line.split()
        if len(fields) == 3 and fields[1] in SYSTEMS:
            found[fields[1]] = float(fields[2])
    return found


def short_series(logs):
    """The rates of the protocol before D[suite-symmetric], by repeat."""
    series = {"mdir": [], "pmemd": []}
    for engine in series:
        for path in sorted(glob.glob(os.path.join(logs, f"{engine}-*.log"))):
            found = rates(path)
            if len(found) == len(SYSTEMS):
                series[engine].append(found)
    ratios = [[100 * m[s] / p[s] for m, p in zip(series["mdir"], series["pmemd"])]
              for s in SYSTEMS]
    bars = [("mdir", f"MDIR ({len(series['mdir'])} runs)", "#c0392b"),
            ("pmemd", f"pmemd.cuda 26, SPFP ({len(series['pmemd'])} runs)",
             "#2c3e50")]
    return series, bars, ratios


def symmetric_series(logs):
    """The rates of the protocol of mdin.GPU, by variant and repeat."""
    by = {}
    for line in open(os.path.join(logs, "runs.jsonl")):
        if not line.strip():
            continue
        run = json.loads(line)
        if run.get("ns_per_day") and not run.get("shared"):
            by.setdefault(run["variant"], {}).setdefault(
                run["repeat"], {})[run["system"]] = run["ns_per_day"]
    series = {variant: [found for _, found in sorted(by.get(variant, {}).items())
                        if len(found) == len(SYSTEMS)]
              for variant in ("output", "searched", "default")}
    repeats = sorted(set(by.get("searched", {})) & set(by.get("output", {})))
    ratios = [[100 * by["searched"][r][s] / by["output"][r][s] for r in repeats
               if s in by["searched"][r] and s in by["output"][r]]
              for s in SYSTEMS]
    bars = [("output", f"pmemd.cuda 26, SPFP ({len(series['output'])} runs)",
             "#2c3e50"),
            ("searched", f"MDIR, searched reaches ({len(series['searched'])} "
             "runs)", "#c0392b"),
            ("default", f"MDIR, default reaches ({len(series['default'])} "
             "runs)", "#e59866")]
    return series, bars, ratios


def main():
    logs, output = sys.argv[1], sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "one RTX 4090 (450 W)"
    symmetric = os.path.exists(os.path.join(logs, "runs.jsonl"))
    series, bars, ratios = (symmetric_series if symmetric else short_series)(logs)
    stats = {e: ([statistics.mean(r[s] for r in runs) for s in SYSTEMS],
                 [statistics.stdev(r[s] for r in runs) if len(runs) > 1 else 0
                  for s in SYSTEMS])
             for e, runs in series.items()}

    x = range(len(SYSTEMS))
    width = 0.8 / len(bars)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for k, (engine, label, color) in enumerate(bars):
        mean, std = stats[engine]
        ax.bar([i + (k - (len(bars) - 1) / 2) * width for i in x], mean, width,
               yerr=std, capsize=2, label=label, color=color)
    for i, s in enumerate(SYSTEMS):
        top = max(stats[engine][0][i] for engine, _, _ in bars)
        ax.text(i, top * 1.25, f"{statistics.mean(ratios[i]):.0f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(20, 4000)
    ax.set_ylabel("ns/day (log scale)")
    labels = []
    for s in SYSTEMS:
        name, rest = s.split("_", 1)
        labels.append(f"{name}\n{rest}\n{ATOMS[name]:,} atoms")
    ax.set_xticks(list(x), labels, fontsize=8.5)
    title = (f"The Amber suite on {device}, mixed precision: MDIR against "
             "pmemd.cuda")
    if symmetric:
        title += ", the steps and output of mdin.GPU"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=150)


if __name__ == "__main__":
    main()
