"""Plots the results of run.sh and of the scans in results/. See README.md.

    python plot.py results/2026-09-30-NVIDIA-GeForce-RTX-3090.csv

Writes pairs.png, search.png, and model.png next to the CSV file."""

import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
measured = sys.argv[1]
out = os.path.dirname(os.path.abspath(measured))


def rows(path):
    with open(path) as file:
        return list(csv.DictReader(line for line in file
                                   if not line.startswith("#")))


data = rows(measured)
reference = rows(os.path.join(here, "results", "reference.csv"))
scan_path = os.path.join(here, "results", "scan.csv")
scan = rows(scan_path) if os.path.exists(scan_path) else []


def series(experiment, variant, parameter=None):
    points = [(float(r["reach"]), float(r["microseconds"])) for r in data
              if r["experiment"] == experiment and r["variant"] == variant
              and (parameter is None or r["parameter"] == parameter)
              and r["microseconds"] not in ("", "nan")]
    return sorted(points)


# 1. The loop over pairs against the reach of the list.
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for variant, style in [("matrix 16 lanes", "o-"),
                       ("tiles 8x8 full", "s--"),
                       ("tiles 8x4 full over 4 warps", "^--"),
                       ("tiles 8x4 half f32 atomics", "v:"),
                       ("tiles 8x4 half fixed point", "d:")]:
    points = series("pairs", variant)
    ax.plot(*zip(*points), style, label=variant)
for r in reference:
    if r["quantity"] == "nonbonded kernel per step":
        ax.plot(float(r["reach"]), float(r["value"]), "k*", markersize=12,
                label="GROMACS 2026.3 (half list, cluster pairs)")
    if r["quantity"] == "pair kernel per step (alone)":
        ax.plot(float(r["reach"]), float(r["value"]), "kx", markersize=10,
                label="MDIR, the kernel of the app")
ax.set_xlabel("Reach of the list (Å)")
ax.set_ylabel("Loop over the pairs (µs)")
ax.set_title("JAC, LJ + direct PME in f32, RTX 3090")
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "pairs.png"), dpi=150)

# 2. The search of a build: the width of the cells, and the reach.
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
variants = ["thread per particle", "warp per particle",
            "warp with exclusions", "block per cell"]
for variant in variants:
    points = sorted((float(r["parameter"]), float(r["microseconds"]))
                    for r in data
                    if r["experiment"] == "search" and r["variant"] == variant
                    and float(r["reach"]) == 10.0
                    and r["microseconds"] not in ("", "nan"))
    left.plot([1 / p for p, _ in points], [t for _, t in points], "o-",
              label=variant)
    right.plot(*zip(*series("search", variant, "2")), "o-", label=variant)
left.set_xlabel("Width of the cells (fraction of the reach)")
left.set_ylabel("Search of one build (µs)")
left.set_title("Reach 10 Å")
right.set_xlabel("Reach (Å)")
right.set_title("Cells of half the reach")
for axis in (left, right):
    axis.set_ylim(bottom=0)
    axis.legend(fontsize=8)
fig.suptitle("JAC, search of the neighbor matrix, RTX 3090")
fig.tight_layout()
fig.savefig(os.path.join(out, "search.png"), dpi=150)

# 3. The time per step of the loop and the builds, K(R) + B / I(R), from
# the loop of the prototype and the intervals that the app measured.
intervals = sorted((float(r["reach"]), float(r["steps_between_builds"]))
                   for r in scan if r["label"].startswith("warp search in i32")
                   and r["steps_between_builds"]) or sorted(
    (float(r["reach"]), float(r["value"])) for r in reference
    if r["quantity"] == "steps between builds")


def interval(reach):
    # Linear in the logarithms between the measured points.
    for (r0, i0), (r1, i1) in zip(intervals, intervals[1:]):
        if r0 <= reach <= r1:
            t = (math.log(reach - 8.0) - math.log(r0 - 8.0)) / (
                math.log(r1 - 8.0) - math.log(r0 - 8.0))
            return math.exp(math.log(i0) + t * (math.log(i1) - math.log(i0)))
    return None


loop = series("pairs", "matrix 16 lanes")
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for build, label in [(780, "build 780 µs (search + marks, before)"),
                     (650, "build 650 µs (warp search, index)"),
                     (470, "build 470 µs (warp search, i32)"),
                     (150, "build 150 µs")]:
    points = [(r, k + build / interval(r)) for r, k in loop
              if interval(r) is not None]
    ax.plot(*zip(*points), "o-", label=label)
ax.set_xlabel("Reach of the list (Å)")
ax.set_ylabel("Loop + builds per step (µs)")
ax.set_title("JAC: K(R) + B / I(R), exact test of validity")
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "model.png"), dpi=150)

# 4. The rate of the app against the reach, before and after.
if scan:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for label in sorted({r["label"] for r in scan}):
        points = sorted((float(r["reach"]), float(r["ns_per_day"]))
                        for r in scan if r["label"] == label)
        ax.plot(*zip(*points), "o-", label=label)
    ax.set_xlabel("pairlist_distance (Å)")
    ax.set_ylabel("ns/day")
    ax.set_title("JAC NVE, MDIR on an RTX 3090")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scan.png"), dpi=150)
print("wrote", out)
