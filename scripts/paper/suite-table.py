#!/usr/bin/env python3
"""Makes the table of Section 10 from the runs of run-suite-repeats.sh.

    scripts/paper/suite-table.py LOGS

LOGS holds mdir-N.log and pmemd-N.log, the summaries of bench.py, and
optionally out-mdir-N/SYSTEM.out, the output of MDIR for each system of
repeat N, from which the change of the energy, the counts of builds and
prunings, and the time of compilation are taken. Rates are the mean and
the sample standard deviation over the repeats; the ratio is that of the
runs of the same repeat, which ran one after the other on the same device.
Prints a table in Markdown.
"""
import glob
import os
import re
import statistics
import sys

SYSTEMS = ["jac_nve", "jac_nve_4fs", "jac_npt", "jac_npt_4fs", "factorix_nve",
           "factorix_npt", "cellulose_nve", "cellulose_npt", "stmv_npt_4fs"]


def rates(path):
    found = {}
    for line in open(path):
        fields = line.split()
        if len(fields) == 3 and fields[1] in SYSTEMS:
            found[fields[1]] = float(fields[2])
    return found


def spread(values, digits):
    mean = statistics.mean(values)
    if len(values) < 2:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def main():
    logs = sys.argv[1]
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
          "| Energy changed by (MDIR) | Builds, prunings: every | Compiled in |")
    print("|---|---|---|---|---|---|---|---|")
    for name in SYSTEMS:
        ours = [mdir[r][name] for r in repeats if name in mdir[r]]
        theirs = [pmemd[r][name] for r in repeats if name in pmemd[r]]
        ratios = [100 * mdir[r][name] / pmemd[r][name] for r in repeats
                  if name in mdir[r] and name in pmemd[r]]
        atoms, changes, builds, prunes, compiled = "", [], [], [], []
        for out in sorted(glob.glob(os.path.join(logs, "out-mdir-*", name + ".out"))):
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
            m = re.search(r"compiled in ([0-9.]+) s", text)
            if m:
                compiled.append(float(m.group(1)))
        change = (f"{min(changes):.1e} to {max(changes):.1e}" if len(changes) > 1
                  else (f"{changes[0]:.1e}" if changes else ""))
        every = (f"{statistics.mean(builds):.1f}, {statistics.mean(prunes):.1f}"
                 if builds and prunes else "")
        lo, hi = (round(min(compiled)), round(max(compiled))) if compiled else (0, 0)
        time = "" if not compiled else (f"{lo} s" if lo == hi else f"{lo}–{hi} s")
        print(f"| `{name}` | {atoms} | {spread(ours, 1)} | {spread(theirs, 1)} "
              f"| {spread(ratios, 1)}% | {change} | {every} | {time} |")
    print(f"\n{len(repeats)} repeats; energy and counts from "
          f"{len(glob.glob(os.path.join(logs, 'out-mdir-*')))} of them.")


if __name__ == "__main__":
    main()
