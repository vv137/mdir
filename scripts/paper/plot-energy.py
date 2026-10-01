#!/usr/bin/env python3
"""Plots the total energy of JAC NVE (2 fs) over time (Figure 7.1): MDIR
before and after the changes of D112, and pmemd.cuda, over 2 ns; and runs
of 0.4 ns that isolate the cause of the drift, each part of the mixed mode
in f64 in turn.

    plot-energy.py LOGS OUTPUT.png

LOGS holds longjac/{long,fixed,mshake}.log, longjac/pmemd/long.out, and
drift/{mdir,fixed,f64_shake,f64_settles,mshake,double}.log, the logs of
those runs. Needs matplotlib.
"""
import os
import sys
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def mdir(path):
    t, e = [], []
    for line in open(os.path.join(sys.argv[1], path)):
        m = re.search(r"INFO:\s+(\d+)\s+([0-9.]+)\s+(-?[0-9.]+)", line)
        if m:
            t.append(float(m.group(2)) / 1000.0); e.append(float(m.group(3)))
    return t, e

def pmemd(path):
    t, e = [], []
    text = open(os.path.join(sys.argv[1], path)).read().split("A V E R A G E S")[0]
    for m in re.finditer(r"NSTEP =\s+(\d+)\s+TIME\(PS\) =\s+([0-9.]+).*?Etot\s+=\s+(-?[0-9.]+)", text, re.S):
        t.append((float(m.group(2)) - 6.0) / 1000.0); e.append(float(m.group(3)))
    return t, e

fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
for path, label, color in [("longjac/long.log", "MDIR mixed, before (D111)", "#c0392b"),
                           ("longjac/fixed.log", "MDIR mixed, divisions of constraints exact", "#e67e22"),
                           ("longjac/mshake.log", "MDIR mixed, + water by M-SHAKE (D112)", "#27ae60"),
                           ]:
    t, e = mdir(path); a.plot(t, [v - e[0] for v in e], label=label, color=color)
t, e = pmemd("longjac/pmemd/long.out")
a.plot([0.0] + t, [0.0] + [v - (-58141.7007) + (e[0] - (-58141.7007)) * 0 for v in e], label="pmemd.cuda 26 (SPFP)", color="#2c3e50")
a.set_xlabel("time (ns)"); a.set_ylabel("E_total − E_total(0) (kcal/mol)")
a.set_title("JAC NVE, 2 fs, 23,558 atoms, 2 ns")
a.legend(fontsize=8); a.grid(alpha=0.3)
for path, label, color in [("drift/mdir.log", "mixed (D111)", "#c0392b"),
                           ("drift/fixed.log", "mixed, divisions of constraints exact", "#e67e22"),
                           ("drift/f64_shake.log", "  … and SHAKE in f64", "#8e44ad"),
                           ("drift/f64_settles.log", "  … and SETTLE in f64", "#16a085"),
                           ("drift/mshake.log", "  … and water by M-SHAKE in f32 (D112)", "#27ae60"),
                           ("drift/double.log", "double", "#2c3e50")]:
    t, e = mdir(path); b.plot(t, [v - e[0] for v in e], label=label, color=color, marker="o", ms=3)
b.set_xlabel("time (ns)"); b.set_ylabel("E_total − E_total(0) (kcal/mol)")
b.set_title("JAC NVE, 0.4 ns: what the drift comes from")
b.legend(fontsize=8); b.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=150)
