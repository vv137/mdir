#!/usr/bin/env python3
"""check-corpus.py: has every parallel region of the host an entry of the
profile map, over the corpus of the equivalence check?

  check-corpus.py MDIR [--only KIND...] [--precisions mixed,double] [-j N]

The profile map (D398, docs/profile-map.md). The corpus is that of
scripts/builder-equivalence/check.py (its control files; the cases written
in Python are left out). Each control file is a case for the CPU target on
two threads, in each precision of --precisions: `mdir emit --stage=lowered
--profile-map=...`, whose lowered program is translated with
`mlir-translate` (beside MDIR's LLVM, or $MLIR_TRANSLATE). A case passes if
the functions `..omp_par` of the LLVM IR and the functions of the entries
are the same, one entry each, and every entry has a source. A case whose
emit fails (as it does with the build of `main`) is counted apart.

Prints a line for each case that fails and the totals; exits with 1 if any
case fails.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "builder-equivalence"))
import check  # noqa: E402  (scripts/builder-equivalence/check.py)

OUTLINED = re.compile(r"^define [^@]*@([\w.$]+\.\.omp_par(?:\.\d+)?)\(", re.M)


def check_case(mdir, translate, case, precision, scratch, index):
    place = os.path.join(scratch, str(index))
    directory, hide = check.mirror(case, place)
    env = dict(os.environ)
    env.update(check.case_env(check.source_text(case)))
    text = check.with_target(check.source_text(case), "cpu")
    text = check.with_execution(text, "threads", "2")
    text = check.with_execution(text, "precision", f'"{precision.upper()}"')
    stem = os.path.splitext(os.path.basename(case.relative))[0]
    control = f"{stem}.profile-map.toml"
    with open(os.path.join(directory, control), "w", encoding="utf-8",
              errors="surrogateescape") as file:
        file.write(text)
    path = os.path.join(place, "map.json")
    try:
        status, out, err = check.run([mdir, "emit", "--stage=lowered", f"--profile-map={path}", control],
                                     directory, hide, env=env)
        if status != 0:
            return "no program", None
        done = subprocess.run([translate, "--mlir-to-llvmir"], input=out, capture_output=True)
        if done.returncode != 0:
            return "fail", "mlir-translate failed"
        functions = set(OUTLINED.findall(done.stdout.decode(errors="replace")))
        regions = json.load(open(path))["regions"]
        named = [region["function"] for region in regions]
        if None in named:
            return "fail", "an entry names no function"
        if len(set(named)) != len(named):
            return "fail", "two entries name one function"
        if set(named) != functions:
            return "fail", (f"without an entry: {sorted(functions - set(named))[:5]}; "
                            f"of no function: {sorted(set(named) - functions)[:5]}")
        unmapped = [region["function"] for region in regions if region["source"] is None]
        if unmapped:
            return "fail", f"no source: {unmapped[:5]}"
        return "pass", (len(regions), Counter(region["source"] for region in regions))
    finally:
        shutil.rmtree(place, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mdir", help="an mdir program or a build tree")
    parser.add_argument("--only", nargs="+", choices=["examples", "tests", "bench", "cases", "extra"])
    parser.add_argument("--bench", help="the prepared Amber suite (MDIR_BENCH_DIR)")
    parser.add_argument("--control", action="append", help="a control file of its own")
    parser.add_argument("--root", default=check.ROOT)
    parser.add_argument("--precisions", default="mixed,double")
    parser.add_argument("--scratch")
    parser.add_argument("--jobs", "-j", type=int, default=8)
    args = parser.parse_args()
    mdir = check.program(args.mdir)
    translate = os.environ.get("MLIR_TRANSLATE") or shutil.which("mlir-translate")
    if not translate:
        sys.exit("check-corpus.py: no mlir-translate; give MLIR_TRANSLATE")
    scratch = args.scratch or tempfile.mkdtemp(prefix="profile-map-")
    os.makedirs(scratch, exist_ok=True)
    cases = [case for case in check.corpus(args, scratch) if case.relative.endswith(".toml")]
    work = [(case, precision) for case in cases for precision in args.precisions.split(",")]
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda item: check_case(mdir, translate, item[1][0], item[1][1], scratch,
                                                        item[0]), enumerate(work)))
    totals, regions, sources = Counter(), 0, Counter()
    for (case, precision), (outcome, detail) in zip(work, results):
        totals[outcome] += 1
        if outcome == "fail":
            print(f"FAIL {case.name} {precision}: {detail}")
        elif outcome == "pass":
            regions += detail[0]
            sources.update(detail[1])
    print(f"{len(work)} cases: {totals['pass']} pass, {totals['fail']} fail, "
          f"{totals['no program']} make no program")
    print(f"{regions} regions, by source: " + ", ".join(f"{k} {v}" for k, v in sorted(sources.items())))
    if not args.scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    sys.exit(1 if totals["fail"] else 0)


if __name__ == "__main__":
    main()
