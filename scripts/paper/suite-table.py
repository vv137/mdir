#!/usr/bin/env python3
"""Makes the tables of Section 10 from the runs of run-suite-repeats.sh.

    scripts/paper/suite-table.py LOGS

LOGS holds runs.jsonl, the numbers that bench.py recorded for every run
(--record), and out-VARIANT-N/SYSTEM.out, the output of MDIR for each
system of repeat N, from which the change of the energy and the counts of
builds and prunings are taken. Every rate is over the same steps for both
programs, from the first report of mdin.GPU to its last step
(D327). Rates and times are the mean and the sample standard
deviation over the repeats; a ratio is taken within each repeat, between
runs that ran one after the other on the same device, and then averaged.
Prints three tables in Markdown: the rates (Table 10.2), the share of the
output (Table 10.3), and the time to solution (Table 10.4).

Without runs.jsonl, LOGS is that of the protocol before D327
(mdir-N.log and pmemd-N.log, the summaries of bench.py, and
out-mdir-N/SYSTEM.out), and the table of that protocol is printed.
"""
import glob
import json
import os
import re
import statistics
import sys

SYSTEMS = ["jac_nve", "jac_nve_4fs", "jac_npt", "jac_npt_4fs", "factorix_nve",
           "factorix_npt", "cellulose_nve", "cellulose_npt", "stmv_npt_4fs"]
# Time steps, ps.
TIMESTEP = {name: 0.004 if name.endswith("4fs") else 0.002 for name in SYSTEMS}


def rates(path):
    found = {}
    for line in open(path):
        fields = line.split()
        if len(fields) == 3 and fields[1] in SYSTEMS:
            found[fields[1]] = float(fields[2])
    return found


def spread(values, digits, suffix=""):
    if not values:
        return ""
    mean = statistics.mean(values)
    if len(values) < 2:
        return f"{mean:.{digits}f}{suffix}"
    return (f"{mean:.{digits}f} ± {statistics.stdev(values):.{digits}f}"
            f"{suffix}")


def mdir_details(paths):
    """The atoms, the range of the change of the energy, the mean counts of
    steps between builds and prunings, and the reaches, over MDIR's
    outputs `paths`."""
    atoms, changes, builds, prunes, reaches = "", [], [], [], set()
    for out in paths:
        text = open(out).read()
        m = re.search(r"MDIR: (\d+) particles", text)
        if m:
            atoms = f"{int(m.group(1)):,}"
        m = re.search(r"energy changed by ([0-9.e+-]+)", text)
        if m:
            changes.append(float(m.group(1)))
        m = re.search(r"built \d+ times, every ([0-9.]+) steps", text)
        if m:
            builds.append(float(m.group(1)))
        m = re.search(r"pruned \d+ times, every ([0-9.]+) steps", text)
        if m:
            prunes.append(float(m.group(1)))
        m = re.search(r"^MDIR: skins: the reach ([0-9.]+) Å"
                      r"(?:, pruned to ([0-9.]+) Å)? \((\w+):", text, re.M)
        if m:
            reaches.add(m.group(1) + (f" / {m.group(2)}" if m.group(2) else ""))
    change = (f"{min(changes):.1e} to {max(changes):.1e}" if len(changes) > 1
              else (f"{changes[0]:.1e}" if changes else ""))
    every = (f"{statistics.mean(builds):.1f}, {statistics.mean(prunes):.1f}"
             if builds and prunes else "")
    return atoms, change, every, "; ".join(sorted(reaches))


def symmetric(logs):
    runs = [json.loads(line) for line in open(os.path.join(logs, "runs.jsonl"))
            if line.strip()]
    # By (variant, system, repeat); a run on a shared device is no
    # measurement and is left out, and named.
    by = {}
    for run in runs:
        if run.get("shared"):
            print(f"left out: {run['variant']} {run['system']} repeat "
                  f"{run['repeat']}: the device was shared", file=sys.stderr)
            continue
        if run.get("ns_per_day"):
            by[(run["variant"], run["system"], run["repeat"])] = run
    repeats = sorted({key[2] for key in by})

    def values(variant, name, field="ns_per_day"):
        return [by[(variant, name, r)][field] for r in repeats
                if (variant, name, r) in by
                and by[(variant, name, r)].get(field) is not None]

    def ratios(a, b, name, percent=True):
        return [(100 if percent else 1) * by[(a, name, r)]["ns_per_day"]
                / by[(b, name, r)]["ns_per_day"]
                for r in repeats if (a, name, r) in by and (b, name, r) in by]

    def shares(full, energies, name):
        """The share of a step that the output takes, 1 - (rate with the
        output) / (rate with the energies alone), within each repeat."""
        return [100 * (1 - by[(full, name, r)]["ns_per_day"]
                       / by[(energies, name, r)]["ns_per_day"])
                for r in repeats if (full, name, r) in by
                and (energies, name, r) in by]

    def one_ns(variant, name):
        """Derived: the wall time of 1 ns, in minutes: the time of the run
        outside its steps at the rate of the window, wall - nstlim dt /
        rate, plus 1 ns at that rate."""
        found = []
        for r in repeats:
            run = by.get((variant, name, r))
            if not run or not run.get("wall"):
                continue
            rate = run["ns_per_day"]
            steps = run["steps"] * TIMESTEP[name] * 1e-3 * 86400 / rate
            found.append((run["wall"] - steps + 86400 / rate) / 60)
        return found

    def fixed(variant, name):
        """Derived: the wall time outside the steps, wall - nstlim dt /
        rate, in seconds."""
        found = []
        for r in repeats:
            run = by.get((variant, name, r))
            if run and run.get("wall"):
                found.append(run["wall"] - run["steps"] * TIMESTEP[name]
                             * 1e-3 * 86400 / run["ns_per_day"])
        return found

    print("| System | Atoms | Steps, timed | pmemd.cuda | MDIR, searched "
          "reaches | MDIR / pmemd.cuda | MDIR, default reaches "
          "| MDIR / pmemd.cuda | Searched reaches, Å | Energy changed by "
          "(MDIR) | Builds, prunings: every |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in SYSTEMS:
        outs = sorted(glob.glob(os.path.join(logs, "out-searched-*",
                                             name + ".out")))
        atoms, change, every, reaches = mdir_details(outs)
        window = sorted({by[k].get("window_steps") for k in by
                         if k[1] == name and by[k].get("window_steps")})
        steps = sorted({by[k]["steps"] for k in by if k[1] == name})
        timed = (f"{window[0]:,} of {steps[0]:,}" if len(window) == 1
                 and len(steps) == 1 else "")
        print(f"| `{name}` | {atoms} | {timed} "
              f"| {spread(values('output', name), 1)} "
              f"| {spread(values('searched', name), 1)} "
              f"| {spread(ratios('searched', 'output', name), 1, '%')} "
              f"| {spread(values('default', name), 1)} "
              f"| {spread(ratios('default', 'output', name), 1, '%')} "
              f"| {reaches} | {change} | {every} |")
    print()
    print("| System | pmemd.cuda, energies only | MDIR, energies only "
          "| MDIR / pmemd.cuda | Output's share: pmemd.cuda | MDIR |")
    print("|---|---|---|---|---|---|")
    for name in SYSTEMS:
        print(f"| `{name}` "
              f"| {spread(values('energies', name), 1)} "
              f"| {spread(values('searched-energies', name), 1)} "
              f"| {spread(ratios('searched-energies', 'energies', name), 1, '%')} "
              f"| {spread(shares('output', 'energies', name), 1, '%')} "
              f"| {spread(shares('searched', 'searched-energies', name), 1, '%')} |")
    print()
    print("| System | Wall time of the run, s: pmemd.cuda | MDIR | Of it "
          "outside the steps (derived), s: pmemd.cuda | MDIR | pmemd.cuda's "
          "setup, s | MDIR compiled in, s | Wall time of 1 ns (derived), "
          "min: pmemd.cuda | MDIR |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in SYSTEMS:
        print(f"| `{name}` "
              f"| {spread(values('output', name, 'wall'), 1)} "
              f"| {spread(values('searched', name, 'wall'), 1)} "
              f"| {spread(fixed('output', name), 1)} "
              f"| {spread(fixed('searched', name), 1)} "
              f"| {spread(values('output', name, 'setup_wall'), 0)} "
              f"| {spread(values('searched', name, 'compiled'), 1)} "
              f"| {spread(one_ns('output', name), 2)} "
              f"| {spread(one_ns('searched', name), 2)} |")
    # The check of the clock of the rows against MDIR's own over the
    # second half of its run.
    print()
    worst = []
    for key, run in by.items():
        if run.get("half_ns_per_day") and run.get("half_ns_per_day_by_rows"):
            worst.append((abs(run["half_ns_per_day_by_rows"]
                              / run["half_ns_per_day"] - 1), key))
    if worst:
        print(f"the clock of the rows against MDIR's own over its second half: "
              f"at most {100 * max(worst)[0]:.3f}% ({max(worst)[1]}) over "
              f"{len(worst)} runs")
    print(f"\n{len(repeats)} repeats.")


def short(logs):
    # A repeat counts once both of its summaries are complete.
    repeats = sorted(int(re.search(r"mdir-(\d+)\.log", p).group(1))
                     for p in glob.glob(os.path.join(logs, "mdir-*.log")))
    repeats = [r for r in repeats
               if len(rates(os.path.join(logs, f"mdir-{r}.log"))) == len(SYSTEMS)]
    mdir, pmemd = {}, {}
    for r in repeats:
        mdir[r] = rates(os.path.join(logs, f"mdir-{r}.log"))
        path = os.path.join(logs, f"pmemd-{r}.log")
        pmemd[r] = rates(path) if os.path.exists(path) else {}

    print("| System | Atoms | MDIR, ns/day | pmemd.cuda, ns/day | MDIR / pmemd.cuda "
          "| Energy changed by (MDIR) | Builds, prunings: every | Compiled in "
          "| Reaches, Å |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in SYSTEMS:
        ours = [mdir[r][name] for r in repeats if name in mdir[r]]
        theirs = [pmemd[r][name] for r in repeats if name in pmemd[r]]
        ratios = [100 * mdir[r][name] / pmemd[r][name] for r in repeats
                  if name in mdir[r] and name in pmemd[r]]
        outs = sorted(glob.glob(os.path.join(logs, "out-mdir-*", name + ".out")))
        atoms, change, every, reaches = mdir_details(outs)
        compiled = []
        for out in outs:
            m = re.search(r"compiled in ([0-9.]+) s", open(out).read())
            if m:
                compiled.append(float(m.group(1)))
        lo, hi = (round(min(compiled)), round(max(compiled))) if compiled else (0, 0)
        time = "" if not compiled else (f"{lo} s" if lo == hi else f"{lo}–{hi} s")
        print(f"| `{name}` | {atoms} | {spread(ours, 1)} | {spread(theirs, 1)} "
              f"| {spread(ratios, 1, '%')} | {change} | {every} | {time} "
              f"| {reaches} |")
    print(f"\n{len(repeats)} repeats; energy and counts from "
          f"{len(glob.glob(os.path.join(logs, 'out-mdir-*')))} of them.")


def main():
    logs = sys.argv[1]
    if os.path.exists(os.path.join(logs, "runs.jsonl")):
        symmetric(logs)
    else:
        short(logs)


if __name__ == "__main__":
    main()
