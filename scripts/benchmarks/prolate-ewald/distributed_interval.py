"""Report the last timed distributed interval of each run log as CSV.

Phase costs are the maximum rank time per step. Missing byte counters in
older logs leave empty cells. Supply each repetition as a separate path.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

PHASES = ("pairs", "migration", "wait", "pme_spread", "pme_overlap",
          "pme_fft", "pme_transpose", "pme_solve", "pme_gather")
COUNTERS = ("pme_overlap_sent_bytes", "pme_transpose_sent_bytes")


def last_interval(path):
    log = path.read_text()
    rates = list(re.finditer(
        r"^MDIR: from step (\d+) to step (\d+), ([\d.]+) ms per step", log, re.M))
    if not rates:
        raise ValueError("missing step timing")
    first, last, rate = rates[-1].groups()
    steps = int(last) - int(first)
    ends = sorted({int(step) for step in re.findall(
        r"^DIST: interval step (\d+) pairs min/mean/max", log, re.M)})
    if steps <= 0 or int(last) not in ends or max(
            (step for step in ends if step < int(last)), default=0) != int(first):
        raise ValueError("phase interval does not match the timed steps")
    prefix = rf"^DIST: interval step {last} "
    row = {"log": path, "first": first, "last": last, "ms_per_step": rate}
    for name in PHASES:
        phase = re.search(prefix + name + r" min/mean/max \S+ \S+ (\S+) s$", log, re.M)
        if not phase:
            raise ValueError(f"missing phase {name}")
        row[name + "_ms"] = 1000 * float(phase[1]) / steps
    for name in COUNTERS:
        counter = re.search(prefix + name + r" (\d+)$", log, re.M)
        if counter:
            row[name + "_per_step"] = int(counter[1]) / steps
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    fields = ["log", "first", "last", "ms_per_step"]
    fields.extend(name + "_ms" for name in PHASES)
    fields.extend(name + "_per_step" for name in COUNTERS)
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    for path in args.logs:
        try:
            writer.writerow(last_interval(path))
        except ValueError as exc:
            parser.exit(1, f"{path}: {exc}\n")


if __name__ == "__main__":
    main()
