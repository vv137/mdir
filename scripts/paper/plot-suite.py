#!/usr/bin/env python3
"""Plots the rates of MDIR and of pmemd.cuda over the Amber suite (Figure
10.1) from the runs of run-suite-repeats.sh: the mean over the repeats,
with the sample standard deviation as an error bar, and the ratio of the
means above each pair.

    plot-suite.py LOGS OUTPUT.png

Needs matplotlib.
"""
import glob
import os
import re
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


def main():
    logs, output = sys.argv[1], sys.argv[2]
    series = {"mdir": [], "pmemd": []}
    for engine in series:
        for path in sorted(glob.glob(os.path.join(logs, f"{engine}-*.log"))):
            found = rates(path)
            if len(found) == len(SYSTEMS):
                series[engine].append(found)
    stats = {e: ([statistics.mean(r[s] for r in runs) for s in SYSTEMS],
                 [statistics.stdev(r[s] for r in runs) if len(runs) > 1 else 0
                  for s in SYSTEMS])
             for e, runs in series.items()}

    x = range(len(SYSTEMS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for k, (engine, label, color) in enumerate(
            [("mdir", f"MDIR ({len(series['mdir'])} runs)", "#c0392b"),
             ("pmemd", f"pmemd.cuda 26, SPFP ({len(series['pmemd'])} runs)",
              "#2c3e50")]):
        mean, std = stats[engine]
        ax.bar([i + (k - 0.5) * width for i in x], mean, width, yerr=std,
               capsize=2, label=label, color=color)
    for i, s in enumerate(SYSTEMS):
        a, b = stats["mdir"][0][i], stats["pmemd"][0][i]
        ax.text(i, max(a, b) * 1.25, f"{100 * a / b:.0f}%", ha="center",
                fontsize=9, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(20, 4000)
    ax.set_ylabel("ns/day (log scale)")
    labels = []
    for s in SYSTEMS:
        name, rest = s.split("_", 1)
        labels.append(f"{name}\n{rest}\n{ATOMS[name]:,} atoms")
    ax.set_xticks(list(x), labels, fontsize=8.5)
    ax.set_title("The Amber suite on one RTX 3090 (300 W), mixed precision: "
                 "MDIR against pmemd.cuda", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=150)


if __name__ == "__main__":
    main()
