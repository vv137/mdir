"""Tables and fits of the CPU breakdown.

    summarize.py [--tag sweep] SYSTEM...

The environment: CPU_BENCH_DIR, as for bench.py, which also holds
pairs.json (pairs.py). The sampled run of each system (ms16) is broken
down first: prof.py DIR/prof --program ... --json DIR/parts.json.

For each system (runs/<tag>/results.jsonl): the mean and sample standard
deviation of each variant over the rounds after the first; the parts of
MDIR's step from the sampled run (the wall of the main thread, as shares,
times the mean rate of m16); the parts of GROMACS's step from the cycle
accounting of md.log (mean over the kept rounds). Then fits of each part
against the size: a + b N (N sites), the loop over pairs also against the
entries of the matrix, PME's FFT against M log2 M (M grid points).
"""
import json
import math
import os
import re
import statistics
import sys

import numpy as np

if not os.environ.get("CPU_BENCH_DIR"):
    sys.exit("summarize.py: CPU_BENCH_DIR, the directory of the systems, is not set")
S = os.environ["CPU_BENCH_DIR"]
tag = "sweep"
a = sys.argv[1:]
if a and a[0] == "--tag":
    tag = a[1]; a = a[2:]
systems = a
pairs = json.load(open(S + "/pairs.json"))

GMX_ROWS = ["Domain decomp.", "Neighbor search", "Force", "PME mesh", "NB X/F buffer ops.", "Update",
            "Constraints", "Kinetic energy", "Rest", "PME spread", "PME gather", "PME 3D-FFT", "PME solve Elec"]


def gmx_parts(d):
    full = open(d + "/md.log").read()
    log = full[full.index("R E A L   C Y C L E"):]
    # The atom pairs that the kernels computed (each pair once, beyond the
    # cutoff and the padding of the clusters included), M per step.
    flops = full[full.index("M E G A - F L O P S"):]
    units = sum(float(m.group(1)) for m in re.finditer(r"^ NxN [^\n]*?\s([0-9.]+)\s+[0-9.]+\s+[0-9.]+$", flops, re.M))
    steps = None
    out = {}
    for row in GMX_ROWS:
        m = re.search(r"^ " + re.escape(row) + r"\s+(?:\d+\s+\d+\s+(\d+)\s+)?([0-9.]+)\s+[0-9.]+\s+[0-9.]+$",
                      log, re.M)
        if not m:
            continue
        if row == "Force":
            steps = int(m.group(1))
        out[row] = float(m.group(2))
    out = {k: v / steps * 1e3 for k, v in out.items()}  # ms per step
    out["pair units (M)"] = units / steps
    return out


MDIR_PARTS = ["pairs", "list builds", "PME spread", "PME FFT", "PME convolution", "PME gather",
              "excluded pairs", "bonded", "bonded or constraints", "other"]
rows = {}
for s in systems:
    path = f"{S}/{s}/runs/{tag}/results.jsonl"
    if not os.path.exists(path):
        continue
    res = [json.loads(l) for l in open(path)]
    by = {}
    for r in res:
        if r.get("ok") and r.get("rate_ms") and r["round"] > 0 and not r["variant"].startswith(("ms", "os")):
            by.setdefault(r["variant"], []).append(r)
    stats = {v: (statistics.mean(x["rate_ms"] for x in rs), statistics.stdev(x["rate_ms"] for x in rs)
                 if len(rs) > 1 else 0.0, len(rs)) for v, rs in by.items()}
    cpu = {r.get("cpu", "?") for r in res}
    builds = [r["build_interval"] for r in by.get("m16", []) if r.get("build_interval")]
    parts = {}
    sampled = [r for r in res if r["variant"] == "ms16" and r.get("ok")]
    if sampled and os.path.exists(sampled[0]["dir"] + "/parts.json") and "m16" in stats:
        wp = json.load(open(sampled[0]["dir"] + "/parts.json"))["wall_part"]
        tot = sum(wp.values())
        for k, v in wp.items():
            key = k if k in MDIR_PARTS else "other"
            parts[key] = parts.get(key, 0.0) + v / tot * stats["m16"][0]
    gparts = {}
    for v in ("g16", "x16"):
        ds = [r["dir"] for r in by.get(v, [])]
        if ds:
            ps = [gmx_parts(d) for d in ds]
            gparts[v] = {k: statistics.mean(p.get(k, 0.0) for p in ps) for k in GMX_ROWS + ["pair units (M)"]}
    rows[s] = dict(stats=stats, cpu=cpu, build_interval=statistics.mean(builds) if builds else None,
                   parts=parts, gparts=gparts, sites=res[0]["sites"], grid=res[0]["grid"])

print(f"## Rates at 16 threads (ms/step, mean ± sd of the kept rounds), tag {tag}\n")
print("| System | Sites | CPU | MDIR m16 | MDIR, reach cutoff + 1 Å | GROMACS default | GROMACS exact | MDIR / GROMACS default | MDIR / GROMACS exact | MDIR builds every |")
print("|---|---|---|---|---|---|---|---|---|---|")
for s, r in rows.items():
    st = r["stats"]

    def f(v):
        return f"{st[v][0]:.2f} ± {st[v][1]:.2f}" if v in st else "—"
    rr = [v for v in st if re.match(r"m16r", v)]
    ratio = lambda v: f"{st['m16'][0] / st[v][0]:.1f}" if "m16" in st and v in st else "—"
    print(f"| {s} | {r['sites']} | {'/'.join(c.replace('AMD EPYC ', '').replace(' 32-Core Processor', '') for c in r['cpu'])} "
          f"| {f('m16')} | {f(rr[0]) if rr else '—'} | {f('g16')} | {f('x16')} | {ratio('g16')} | {ratio('x16')} "
          f"| {r['build_interval'] or 0:.1f} |")

print("\n## Pairs: entries of MDIR's matrix (reach 10.5 A, 9.5 A on JAC) and the atom pairs of GROMACS's kernels, M per step; ns of one thread per pair\n")
print("| System | MDIR entries | MDIR ns/entry | GROMACS default units | ns/unit | GROMACS exact units | ns/unit |")
print("|---|---|---|---|---|---|---|")
for s, r in rows.items():
    key = "ordered_pairs_9.5" if s == "jac" else "ordered_pairs_10.5"
    e = pairs[s][key] / 1e6
    cells = [f"{e:.1f}", f"{r['parts'].get('pairs', 0) * 16 / e:.2f}" if r['parts'] else "—"]
    for v, f in (("g16", "Force"), ("x16", "Force")):
        g = r["gparts"].get(v)
        cells += [f"{g['pair units (M)']:.1f}", f"{g[f] * 16 / g['pair units (M)']:.2f}"] if g else ["—", "—"]
    print(f"| {s} | " + " | ".join(cells) + " |")

print("\n## MDIR parts at 16 threads (ms/step, from the sampled run's shares times the rate)\n")
print("| System | " + " | ".join(MDIR_PARTS) + " |")
print("|---|" + "---|" * len(MDIR_PARTS))
for s, r in rows.items():
    print(f"| {s} | " + " | ".join(f"{r['parts'].get(k, 0):.2f}" for k in MDIR_PARTS) + " |")

GP = ["pair units (M)", "Force", "Neighbor search", "Domain decomp.", "PME spread", "PME 3D-FFT", "PME solve Elec", "PME gather",
      "NB X/F buffer ops.", "Update", "Constraints", "Rest"]
for v in ("g16", "x16"):
    print(f"\n## GROMACS parts, {v} (ms/step, md.log)\n")
    print("| System | " + " | ".join(GP) + " |")
    print("|---|" + "---|" * len(GP))
    for s, r in rows.items():
        if v in r["gparts"]:
            print(f"| {s} | " + " | ".join(f"{r['gparts'][v].get(k, 0):.3f}" for k in GP) + " |")


def fit(xs, ys):
    X = np.column_stack([np.ones(len(xs[0]))] + [np.array(x, float) for x in xs])
    y = np.array(ys, float)
    beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    dof = max(1, len(y) - X.shape[1])
    resid = y - X @ beta
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov))


water = [s for s in rows if s.startswith("tip3p")]
if len(water) >= 3:
    print("\n## Fits over the TIP3P boxes (ms; N in sites)\n")
    print("| Engine | Part | Law | a (ms) | b | c |")
    print("|---|---|---|---|---|---|")
    N = [rows[s]["sites"] for s in water]
    E = [pairs[s]["ordered_pairs_10.5"] for s in water]
    M = [math.prod(rows[s]["grid"]) for s in water]
    MlogM = [m * math.log2(m) for m in M]

    def line(engine, part, ys, law="a + b N", x=None, unit="ns/site", scale=1e6):
        x = x or [N]
        beta, err = fit(x, ys)
        b = f"{beta[1] * scale:.3g} ± {err[1] * scale:.2g} {unit}"
        c = f"{beta[2] * scale:.3g} ± {err[2] * scale:.2g}" if len(beta) > 2 else ""
        print(f"| {engine} | {part} | {law} | {beta[0]:.3g} ± {err[0]:.2g} | {b} | {c} |")

    for p in MDIR_PARTS[:6]:
        ys = [rows[s]["parts"].get(p, 0) for s in water]
        if p == "pairs":
            line("MDIR", p, ys, "a + b E (E entries of the matrix, reach 10.5 Å)", [E], "ps/entry", 1e9)
        elif p == "PME FFT":
            line("MDIR", p, ys, "a + b M log2 M", [MlogM], "ps/(point·log2)", 1e9)
        else:
            line("MDIR", p, ys)
    line("MDIR", "step", [rows[s]["stats"]["m16"][0] for s in water])
    for v in ("g16", "x16"):
        for p in ("Force", "Neighbor search", "PME spread", "PME 3D-FFT", "PME gather"):
            ys = [rows[s]["gparts"][v].get(p, 0) for s in water]
            if p == "PME 3D-FFT":
                line(v, p, ys, "a + b M log2 M", [MlogM], "ps/(point·log2)", 1e9)
            else:
                line(v, p, ys)
        line(v, "step", [rows[s]["stats"][v][0] for s in water])
json.dump({s: dict(r, cpu=list(r["cpu"])) for s, r in rows.items()}, open(f"{S}/summary-{tag}.json", "w"),
          indent=1, default=str)
