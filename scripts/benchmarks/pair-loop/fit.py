#!/usr/bin/env python3
"""Fits the parts of a step against the size of the system (#345, #350).

    fit.py DIR...

Each DIR holds the `results.json` of `sweep.py` (one system: its sites, PME
grid, and for each variant the times of its rounds and the kernels of one
run under nsys) and the `eq.gro` the runs started from. The kernels are
summed by part, per step:

  loop    the loop over pairs (MDIR `pair_for`; GROMACS `nbnxn_kernel`)
  build   builds of the lists (MDIR's `mdrt_gpu_*` kernels other than PME)
  prune   prunings (MDIR `refresh_neighbors`; GROMACS `nbnxn_kernel_prune`)
  pme     PME (spread, gather, solve, FFTs)
  layout  GROMACS's `x_to_nbat` (MDIR has none)
  rest    everything else

and fitted by least squares, with standard errors from the residuals:

  loop    T = a + b N_pairs, N_pairs the pairs within the inner reach,
          N^2 (4/3) pi r_in^3 / (2 V) (a uniform density: within 0.3% of a
          count on the water boxes and JAC)
  build   per call, T = a + b N; with the interval in steps
  prune   per call, T = a + b N; with the interval in steps
  pme     T = a + b N + c M log2 M, M the points of the grid
  rest    T = a + b N

and the two laws of the model of #350: the loop against the inner reach
(T proportional to r_in^k, from the variants m93, m96, m100 of each
system), and the interval of the prunings against the inner skin
s = r_in - r_c (proportional to s^q: q = 1 ballistic, q = 2 diffusive).

Prints Markdown tables. Needs NumPy.
"""
import json
import math
import os
import sys

import numpy as np


def part(name, gmx):
    n = name.lower()
    if "pme" in n or "fft" in n:
        return "pme"
    if gmx:
        if "prune" in n:
            return "prune"
        if "nbnxn_kernel" in n or "nbnxm_kernel" in n:
            return "loop"
        if "nbat" in n:
            return "layout"
        return "rest"
    if "pair_for" in n:
        return "loop"
    if "refresh_neighbors" in n:
        return "prune"
    if n.startswith("mdrt_gpu_"):
        return "build"
    return "rest"


def box_of(path):
    lines = open(path).read().splitlines()
    return [float(v) * 10.0 for v in lines[-1].split()[:3]]


def fit(columns, y):
    """Least squares y = X p; returns p and their standard errors."""
    X = np.column_stack(columns)
    y = np.asarray(y, dtype=float)
    p, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(y) - X.shape[1]
    if dof <= 0:
        return p, np.full_like(p, np.nan)
    sigma2 = float(((y - X @ p) ** 2).sum()) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return p, np.sqrt(np.diag(cov))


def fmt(p, e, scale=1.0, digits=3):
    return f"{p * scale:.{digits}g} ± {e * scale:.2g}"


systems = []
for d in sys.argv[1:]:
    r = json.load(open(os.path.join(d, "results.json")))
    L = box_of(os.path.join(d, "eq.gro"))
    r["volume"] = L[0] * L[1] * L[2]
    r["M"] = r["grid"][0] * r["grid"][1] * r["grid"][2]
    for v, info in r["variants"].items():
        gmx = v == "gmx"
        steps = r["steps"]
        parts = {}
        for k in info["kernels"]:
            p = part(k["name"], gmx)
            parts[p] = parts.get(p, 0.0) + k["ns"] / 1e3 / steps
        info["parts"] = parts
        info["device"] = sum(parts.values())
        rin = info.get("r_in")
        n = r["sites"]
        info["pairs"] = (n * n * 4.0 / 3.0 * math.pi * rin ** 3 / (2.0 * r["volume"])) if rin else None
    systems.append(r)

# The table of the parts.
print("| System | Sites | Variant | Step (µs, mean ± s.e.) | Loop | Build | Prune | PME | Layout | Rest | "
      "Pairs within r_in (M) | Loop per pair (ps) | Build every | Prune every |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in systems:
    for v, info in r["variants"].items():
        t = info["times"]
        step = (f"{np.mean(t):.1f} ± {np.std(t, ddof=1) / math.sqrt(len(t)):.1f}" if len(t) > 1 else "-")
        p = info["parts"]
        per = (f"{p.get('loop', 0) / info['pairs'] * 1e6:.2f}" if info["pairs"] else "-")
        print(f"| {r['name']} | {r['sites']} | {v} | {step} | {p.get('loop', 0):.1f} | {p.get('build', 0):.1f} | "
              f"{p.get('prune', 0):.1f} | {p.get('pme', 0):.1f} | {p.get('layout', 0):.1f} | {p.get('rest', 0):.1f} | "
              f"{info['pairs'] / 1e6 if info['pairs'] else 0:.1f} | {per} | "
              f"{info.get('build_interval', '-')} | {info.get('prune_interval', '-')} |")


def collect(variant, only_water=True):
    rows = []
    for r in systems:
        if only_water and r["name"] == "jac":
            continue
        if variant in r["variants"]:
            rows.append((r, r["variants"][variant]))
    return rows


print("\n**Fits over the water boxes** (µs per step, or per call; N in sites, N_pairs in pairs, M in grid points)\n")
print("| Variant | Part | Law | a | b | c |")
print("|---|---|---|---|---|---|")
for variant in ("m96", "p96", "gmx"):
    rows = collect(variant)
    if len(rows) < 3:
        continue
    n = np.array([r["sites"] for r, _ in rows], dtype=float)
    one = np.ones_like(n)
    pairs = np.array([i["pairs"] or np.nan for _, i in rows])
    loop = np.array([i["parts"].get("loop", 0) for _, i in rows])
    if not np.isnan(pairs).any():
        p, e = fit([one, pairs], loop)
        print(f"| {variant} | loop | a + b N_pairs | {fmt(p[0], e[0])} | {fmt(p[1], e[1], 1e6)} ps/pair | |")
    if variant != "gmx":
        for kind, count_key in (("build", "builds"), ("prune", "prunes")):
            per_call, ok = [], True
            for r, i in rows:
                calls = i.get(count_key)
                if not calls:
                    ok = False
                    break
                per_call.append(i["parts"].get(kind, 0) * r["steps"] / calls)
            if ok:
                p, e = fit([one, n], per_call)
                interval = ", ".join(f"{i.get(count_key[:-1] + '_interval')}" for _, i in rows)
                print(f"| {variant} | {kind}, per call | a + b N | {fmt(p[0], e[0])} | "
                      f"{fmt(p[1], e[1], 1e3)} ns/site | interval (steps): {interval} |")
    else:
        per_call = [i["parts"].get("prune", 0) * i.get("prune_interval", 1) for _, i in rows]
        p, e = fit([one, n], per_call)
        print(f"| gmx | prune, per call (from the interval) | a + b N | {fmt(p[0], e[0])} | "
              f"{fmt(p[1], e[1], 1e3)} ns/site | interval (steps): "
              + ", ".join(str(i.get("prune_interval")) for _, i in rows) + " |")
    m = np.array([r["M"] for r, _ in rows], dtype=float)
    pme = np.array([i["parts"].get("pme", 0) for _, i in rows])
    p, e = fit([one, n, m * np.log2(m)], pme)
    print(f"| {variant} | PME | a + b N + c M log2 M | {fmt(p[0], e[0])} | {fmt(p[1], e[1], 1e3)} ns/site | "
          f"{fmt(p[2], e[2], 1e3)} ns/(point·log2) |")
    rest = np.array([i["parts"].get("rest", 0) + i["parts"].get("layout", 0) for _, i in rows])
    p, e = fit([one, n], rest)
    print(f"| {variant} | rest (with layout) | a + b N | {fmt(p[0], e[0])} | {fmt(p[1], e[1], 1e3)} ns/site | |")
    dev = np.array([i["device"] for _, i in rows])
    p, e = fit([one, n], dev)
    print(f"| {variant} | all kernels | a + b N | {fmt(p[0], e[0])} | {fmt(p[1], e[1], 1e3)} ns/site | |")
    t = np.array([np.mean(i["times"]) for _, i in rows])
    p, e = fit([one, n], t)
    print(f"| {variant} | step (wall) | a + b N | {fmt(p[0], e[0])} | {fmt(p[1], e[1], 1e3)} ns/site | |")

print("\n**The loop against the inner reach** (T_loop proportional to r_in^k, variants m93, m96, m100), "
      "and **the interval of the prunings against the inner skin** (proportional to s^q)\n")
print("| System | k (loop) | k (whole step) | q (prune interval) | Prune intervals (s = 0.3, 0.6, 1.0 Å) |")
print("|---|---|---|---|---|")
pooled_k, pooled_q = [], []
for r in systems:
    vs = [r["variants"].get(v) for v in ("m93", "m96", "m100")]
    if any(v is None for v in vs):
        continue
    rin = np.array([v["r_in"] for v in vs])
    loop = np.array([v["parts"].get("loop", 0) for v in vs])
    step = np.array([np.mean(v["times"]) for v in vs])
    (c, k), (_, ek) = fit([np.ones(3), np.log(rin)], np.log(loop))
    (_, ks), (_, eks) = fit([np.ones(3), np.log(rin)], np.log(step))
    s = rin - r["cutoff"]
    iv = np.array([v.get("prune_interval") or np.nan for v in vs])
    if not np.isnan(iv).any():
        (_, q), (_, eq) = fit([np.ones(3), np.log(s)], np.log(iv))
        pooled_q.append((np.log(s), np.log(iv)))
        qs = fmt(q, eq)
    else:
        qs = "-"
    pooled_k.append((np.log(rin), np.log(loop)))
    print(f"| {r['name']} | {fmt(k, ek)} | {fmt(ks, eks)} | {qs} | {', '.join(f'{x:.2f}' for x in iv)} |")


def pooled(groups):
    """One exponent, an intercept for each system."""
    cols, ys = [], []
    for g, (x, y) in enumerate(groups):
        for xi, yi in zip(x, y):
            row = [0.0] * len(groups) + [xi]
            row[g] = 1.0
            cols.append(row)
            ys.append(yi)
    X = np.array(cols)
    p, e = fit([X[:, j] for j in range(X.shape[1])], ys)
    return p[-1], e[-1]


if len(pooled_k) > 1:
    k, ek = pooled(pooled_k)
    print(f"| pooled | {fmt(k, ek)} | | " + (fmt(*pooled(pooled_q)) if len(pooled_q) > 1 else "-") + " | |")
