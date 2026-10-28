#!/usr/bin/env python3
"""check.py: is the program that `mdir emit` prints the same for two builds?

The equivalence check of #568 (docs/driver.md, "The equivalence check of
the builder"). Every stage of the move of the builder from MLIR text to
OpBuilder keeps the generated program as it is; this script shows it over a
fixed corpus of control files.

  check.py compare OLD NEW [corpus options]   the two builds, case by case
  check.py record MDIR --output FILE          the hashes of one build
  check.py verify MDIR --hashes FILE          one build against recorded hashes

OLD, NEW, and MDIR are an `mdir` program or a build tree (its bin/mdir).

The corpus (all by default; --only takes one or more of the kinds):
  examples  the control files (*.toml) of examples/
  tests     the control files of the lit tests: the *.toml files under test/,
            and the parts named *.toml of the tests that split-file divides
            (the lines `#--- NAME` or `//--- NAME`), with the
            other parts of the same test beside them, and the files that
            its RUN lines copy there (`cp %S/PATH... %t[/NAME]`) linked
            under the names they are copied to; the other commands of the
            RUN lines (sed, scripts, environment variables) are not
            followed, so a control file that needs them stops with an error
            in both builds
  bench     the control files (*.toml) of the systems of the prepared Amber
            suite: the directories of MDIR_BENCH_DIR (or --bench) one level
            down; without either, this kind is left out
--control FILE adds a control file of its own (the corpus "extra").

Each control file is a case for each target of --targets (cpu and gpu by
default): a copy of the file with `target` set in [execution]. `mdir emit`
uses no device; the GPU target only changes the program. Each is a case
for each kind of program of --kinds as well: `run`, the program of the
run, and `segments`, the program of segments that a Python simulation of
the same model runs (`mdir emit --segments`, which builds the branches of
the start and the loop over the parts). Both by default; a build of
before `--segments` takes `--kinds run`. A case runs in a
scratch directory that mirrors the directories above the control file with
links, so that the paths of its inputs (relative to the control file or to
the directory it runs in) resolve as they would beside it. The outputs of a
case are its exit status, its standard output, and its standard error, with
the path of the scratch directory replaced by <corpus>.

The forms compared (--forms, both by default):
  raw      the bytes that `mdir emit --stage=STAGE` prints
  printed  for the module stage: the module after a round trip through
           mdir-opt (--printer, by default the one beside NEW or MDIR) with
           MLIR's default printer: SSA names numbered by the printer, no
           locations, no comments, the attributes as MLIR prints them.
           This is the form that a builder that makes IR objects instead
           of text must keep; raw holds only while the text is made the
           same way.
--stages takes module (the default), lowered, or both: lowered is the
program after the passes, as `mdir emit --stage=lowered` prints it, with
the kernels of a GPU target (no device is used, but the CUDA toolkit is).
MLIR's printer already prints it, so it has one form, "printed", which is
compared whatever --forms says.

A case is identical when the status, the output, and the error output are
the same in every form. Failed printed-form diagnostics ignore only generated
buffer positions and redundant source excerpts before complete operation
notes; messages, severities, operation dumps, parser excerpts, and user-file
locations remain checked. Raw failures are byte-exact. A case that stops
with the same error in both builds is identical too, and is counted as such. The
report names cases by their place in the corpus, never by a path of the
machine. The exit status is 0 when every case is identical, 1 when one
differs, and 2 for an error of the script.
"""
import argparse
import difflib
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MARKER = re.compile(r"^(?:#|//)--- (\S+)\s*$")
HASHES_HEADER = "# builder-equivalence hashes 2"


def program(path):
    """An mdir program, or the bin/mdir of a build tree."""
    if os.path.isdir(path):
        path = os.path.join(path, "bin", "mdir")
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        sys.exit(f"check.py: no mdir program at {path}")
    return os.path.abspath(path)


def run_copies(path, directory):
    """The files that the RUN lines of a test copy into %t: (source, name)."""
    with open(path, encoding="utf-8", errors="surrogateescape") as file:
        text = file.read()
    lines, current = [], ""
    for match in re.finditer(r"(?m)^\s*(?:#|//|;)\s*RUN:(.*)$", text):
        line = match.group(1).rstrip()
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        lines.append(current + line)
        current = ""
    copies = []
    for line in lines:
        for command in re.split(r"&&|;|\|", line):
            words = command.split()
            if not words or words[0] != "cp":
                continue
            words = [w for w in words[1:] if not w.startswith("-")]
            if len(words) < 2:
                continue
            *sources, destination = words
            if not destination.startswith("%t"):
                continue
            rest = destination[2:].lstrip("/")
            for source in sources:
                if not source.startswith("%S"):
                    continue
                real = os.path.normpath(directory + source[2:])
                if not rest:
                    copies.append((real, os.path.basename(real)))
                elif len(sources) == 1 and not destination.endswith("/"):
                    copies.append((real, rest))
                else:
                    copies.append((real, os.path.join(rest, os.path.basename(real))))
    return copies


def split_parts(path):
    """The parts of a test that split-file divides: name -> text."""
    parts, name, lines = {}, None, []
    with open(path, encoding="utf-8", errors="surrogateescape") as file:
        for line in file:
            match = MARKER.match(line)
            if match:
                if name is not None:
                    parts[name] = "".join(lines)
                name, lines = match.group(1), []
            elif name is not None:
                lines.append(line)
    if name is not None:
        parts[name] = "".join(lines)
    return parts


class Case:
    """A control file of the corpus: its name in the report, the directory
    whose content it may name (top), and its place under it."""

    def __init__(self, name, top, relative, parts=None, copies=()):
        self.name, self.top, self.relative = name, top, relative
        self.parts, self.copies = parts, copies
        self.text = None


def corpus(args, scratch):
    cases = []
    kinds = set(args.only or ["examples", "tests", "bench", "extra"])
    if "examples" in kinds:
        top = os.path.join(args.root, "examples")
        for directory, _, files in sorted(os.walk(top)):
            for file in sorted(files):
                if file.endswith(".toml"):
                    relative = os.path.relpath(os.path.join(directory, file), top)
                    cases.append(Case(f"examples/{relative}", top, relative))
    if "tests" in kinds:
        top = os.path.join(args.root, "test")
        for directory, subdirectories, files in os.walk(top):
            subdirectories.sort()
            for file in sorted(files):
                path = os.path.join(directory, file)
                relative = os.path.relpath(path, top)
                if file.endswith(".toml"):
                    cases.append(Case(f"test/{relative}", top, relative))
                elif file.endswith(".test"):
                    parts = split_parts(path)
                    controls = [part for part in sorted(parts) if part.endswith(".toml")]
                    copies = run_copies(path, directory) if controls else ()
                    for part in controls:
                        cases.append(Case(f"test/{relative}:{part}", directory, part,
                                          parts=parts, copies=copies))
    bench = args.bench or os.environ.get("MDIR_BENCH_DIR")
    if "bench" in kinds and bench:
        top = os.path.abspath(bench)
        for system in sorted(os.listdir(top)):
            directory = os.path.join(top, system)
            if not os.path.isdir(directory) or os.path.islink(directory):
                continue
            for file in sorted(os.listdir(directory)):
                if file.endswith(".toml") and os.path.isfile(os.path.join(directory, file)):
                    cases.append(Case(f"bench/{system}/{file}", top, f"{system}/{file}"))
    if "extra" in kinds:
        for path in args.control or []:
            path = os.path.abspath(path)
            cases.append(Case(f"extra/{os.path.basename(path)}", os.path.dirname(path),
                              os.path.basename(path)))
    return cases


def with_target(text, target):
    """The control file with `target` of [execution] set."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    setting = f'target = "{target.upper()}"\n'
    header = next((i for i, line in enumerate(lines)
                   if re.match(r"^\s*\[execution\]\s*(#.*)?$", line)), None)
    if header is None:
        return "".join(lines) + "\n[execution]\n" + setting
    end = next((i for i in range(header + 1, len(lines))
                if re.match(r"^\s*\[", lines[i])), len(lines))
    for i in range(header + 1, end):
        if re.match(r"^\s*target\s*=", lines[i]):
            lines[i] = setting
            return "".join(lines)
    lines.insert(header + 1, setting)
    return "".join(lines)


def mirror(case, place):
    """A directory under place in which the control file of the case can be
    written beside links to everything around it; returns the directory and
    the root to hide in the outputs."""
    if case.parts is not None:
        directory = os.path.join(place, "case")
        os.makedirs(directory)
        for name, text in case.parts.items():
            path = os.path.join(directory, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", errors="surrogateescape") as file:
                file.write(text)
        # The files that the RUN lines copy next to the parts are linked
        # there under the names they are copied to.
        for source, name in case.copies:
            target = os.path.join(directory, name)
            if os.path.exists(source) and not os.path.lexists(target):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                os.symlink(source, target)
        return directory, place
    # The directories from the top of the corpus down to the control file
    # are made, and each holds links to the entries of the original.
    root = os.path.join(place, "top")
    original, copy = case.top, root
    os.makedirs(copy)
    steps = os.path.dirname(case.relative).split(os.sep) if os.path.dirname(case.relative) else []
    for step in steps + [None]:
        for entry in os.listdir(original):
            if entry != step:
                os.symlink(os.path.join(original, entry), os.path.join(copy, entry))
        if step is None:
            break
        original, copy = os.path.join(original, step), os.path.join(copy, step)
        os.makedirs(copy)
    return copy, place


def source_text(case):
    """The control file of the case, read once: both builds take this text,
    even if the file changes while the check runs (another run in the
    prepared Amber suite may write its control files)."""
    if case.parts is not None:
        return case.parts[case.relative]
    if case.text is None:
        with open(os.path.join(case.top, case.relative), encoding="utf-8",
                  errors="surrogateescape") as file:
            case.text = file.read()
    return case.text


def run(command, cwd, hide, input=None):
    try:
        done = subprocess.run(command, cwd=cwd, input=input, capture_output=True,
                              timeout=1800)
        status, out, err = done.returncode, done.stdout, done.stderr
    except subprocess.TimeoutExpired:
        status, out, err = "timeout", b"", b""
    hidden = hide.encode()
    real = os.path.realpath(hide).encode()
    out = out.replace(hidden, b"<corpus>").replace(real, b"<corpus>")
    err = err.replace(hidden, b"<corpus>").replace(real, b"<corpus>")
    return status, out, err



GENERATED_DIAGNOSTIC = re.compile(
    rb'^(?P<location>(?:<stdin>|-):[0-9]+:[0-9]+|'
    rb'loc\("(?:<stdin>|-)":[0-9]+:[0-9]+\)):'
    rb'\s*(?P<severity>error|warning|note|remark):')
SOURCE_CARET = re.compile(rb'^[ \t]*[~^]+[ \t\r\n]*$')


def normalize_generated_diagnostics(error):
    """Ignore generated-buffer positions, not diagnostic semantics.

    A source manager may print a source line/caret before its full generic
    operation note. An explicit IR location prints that same operation note
    without the excerpt. Retain the complete note, message, severity, and
    parser excerpts (including an undefined SSA name); user-file locations
    are unchanged. Successful output and raw-form failures stay byte-exact.
    """
    lines = error.splitlines(keepends=True)
    result = []
    index = 0
    while index < len(lines):
        match = GENERATED_DIAGNOSTIC.match(lines[index])
        if not match:
            result.append(lines[index])
            index += 1
            continue
        result.append(b"<generated>: " + lines[index][match.start("severity"):])
        index += 1
        if index + 2 < len(lines) and SOURCE_CARET.match(lines[index + 1]):
            note = GENERATED_DIAGNOSTIC.match(lines[index + 2])
            if (note and note.group("location") == match.group("location") and
                    lines[index + 2][note.start("severity"):].startswith(
                        b"note: see current operation:")):
                index += 2  # Its complete semantic operation dump follows.
    return b"".join(result)


def printed_result(entry):
    status, output, error = entry
    if status != 0:
        error = normalize_generated_diagnostics(error)
    return status, output, error


def outputs(mdir, printer, case, target, stages, forms, scratch, index):
    """The outputs of one build for one case and target:
    {(stage, form): (status, out, err)}."""
    target, _, kind = target.partition(" ")
    place = os.path.join(scratch, f"{index}-{target}{kind}-{hashlib.sha1(mdir.encode()).hexdigest()[:8]}")
    directory, hide = mirror(case, place)
    stem = os.path.splitext(os.path.basename(case.relative))[0]
    control = f"{stem}.equivalence-{target}.toml"
    with open(os.path.join(directory, control), "w", encoding="utf-8",
              errors="surrogateescape") as file:
        file.write(with_target(source_text(case), target))
    segments = ["--segments"] if kind == "segments" else []
    result = {}
    for stage in stages:
        status, out, err = run([mdir, "emit", f"--stage={stage}", *segments, control],
                               directory, hide)
        if stage == "lowered":
            result[(stage, "printed")] = printed_result((status, out, err))
            continue
        if "raw" in forms:
            result[(stage, "raw")] = (status, out, err)
        if stage == "module" and "printed" in forms:
            if status == 0:
                result[(stage, "printed")] = printed_result(
                    run([printer, "-"], directory, hide, input=out))
            else:
                result[(stage, "printed")] = printed_result((status, b"", err))
    shutil.rmtree(place, ignore_errors=True)
    return result


def digest(entry):
    status, out, err = entry
    return (str(status), hashlib.sha256(out).hexdigest(), hashlib.sha256(err).hexdigest())


def summary(cases, rows):
    """rows: (case, target, stage, form, same, failed)."""
    keys = {(case, target) for case, target, *_ in rows}
    different = {(case, target) for case, target, _, _, same, _ in rows if not same}
    failed = {(case, target) for case, target, _, _, same, failed in rows if same and failed}
    stages = sorted({(stage, form) for _, _, stage, form, _, _ in rows})
    print(f"control files: {len(cases)}")
    print(f"cases (control file, target, and kind): {len(keys)}")
    print("compared: " + ", ".join(f"{stage} {form}" for stage, form in stages))
    print(f"identical: {len(keys) - len(different)}"
          f" (of which stop with the same error in both: {len(failed - different)})")
    print(f"different: {len(different)}")
    return 1 if different else 0


def show_difference(name, target, stage, form, old, new, lines):
    print(f"DIFFERENT: {name} [{target}] {stage} {form}")
    if old[0] != new[0]:
        print(f"  status: {old[0]} -> {new[0]}")
    for label, a, b in (("output", old[1], new[1]), ("error output", old[2], new[2])):
        if a != b:
            diff = list(difflib.unified_diff(
                a.decode(errors="replace").splitlines(), b.decode(errors="replace").splitlines(),
                "old", "new", lineterm="", n=1))
            changed = sum(1 for line in diff[2:] if line[:1] in "+-")
            print(f"  {label}: lines removed or added: {changed}")
            for line in diff[:lines]:
                print("    " + line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["compare", "record", "verify"])
    parser.add_argument("programs", nargs="+", help="OLD NEW, or MDIR")
    parser.add_argument("--output", help="record: the file of the hashes")
    parser.add_argument("--hashes", help="verify: the file of the hashes")
    parser.add_argument("--only", nargs="+", choices=["examples", "tests", "bench", "extra"])
    parser.add_argument("--bench", help="the prepared Amber suite (MDIR_BENCH_DIR)")
    parser.add_argument("--control", action="append", help="a control file of its own")
    parser.add_argument("--root", default=ROOT, help="the repository (this one)")
    parser.add_argument("--targets", default="cpu,gpu")
    parser.add_argument("--kinds", default="run,segments",
                        help="the programs: of the run, and of segments (mdir emit --segments)")
    parser.add_argument("--stages", default="module")
    parser.add_argument("--forms", default="raw,printed")
    parser.add_argument("--printer", help="mdir-opt for the printed form")
    parser.add_argument("--scratch", help="where the cases run (a temporary directory)")
    parser.add_argument("--jobs", "-j", type=int, default=8)
    parser.add_argument("--diff-lines", type=int, default=20)
    parser.add_argument("--show-errors", action="store_true",
                        help="name each case that stops with an error, and its first line")
    args = parser.parse_args()

    count = 2 if args.command == "compare" else 1
    if len(args.programs) != count:
        parser.error(f"{args.command} takes {count} mdir program(s)")
    programs = [program(path) for path in args.programs]
    targets = [t for t in args.targets.split(",") if t]
    kinds = [k for k in args.kinds.split(",") if k]
    stages = [s for s in args.stages.split(",") if s]
    forms = [f for f in args.forms.split(",") if f]
    for value, allowed in ((targets, {"cpu", "gpu"}), (kinds, {"run", "segments"}),
                           (stages, {"module", "lowered"}),
                           (forms, {"raw", "printed"})):
        if not value or set(value) - allowed:
            parser.error(f"expected some of {', '.join(sorted(allowed))}")
    printer = None
    if "printed" in forms and "module" in stages:
        printer = args.printer or os.path.join(os.path.dirname(programs[-1]), "mdir-opt")
        if not os.access(printer, os.X_OK):
            sys.exit(f"check.py: no mdir-opt at {printer}: give --printer, or --forms raw")
    if args.command == "record" and not args.output:
        parser.error("record needs --output")
    if args.command == "verify" and not args.hashes:
        parser.error("verify needs --hashes")

    scratch = tempfile.mkdtemp(prefix="builder-equivalence-", dir=args.scratch)
    try:
        cases = corpus(args, scratch)
        for case in cases:
            source_text(case)
        if not cases:
            sys.exit("check.py: the corpus is empty")
        # A case of the program of segments is named "<target> segments".
        work = [(i, case, target + ("" if kind == "run" else " " + kind))
                for i, case in enumerate(cases) for target in targets for kind in kinds]
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            results = list(pool.map(
                lambda item: [outputs(mdir, printer, item[1], item[2], stages, forms,
                                      scratch, item[0]) for mdir in programs], work))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if args.show_errors:
        for (_, case, target), result in zip(work, results):
            for (stage, form), (status, _, err) in sorted(result[-1].items()):
                if status != 0 and (form == forms[0] or stage == "lowered"):
                    first = err.decode(errors="replace").strip().splitlines()[:1]
                    print(f"ERROR: {case.name} [{target}] {stage}: {first[0] if first else status}")

    if args.command == "record":
        with open(args.output, "w") as file:
            file.write(HASHES_HEADER + "\n")
            file.write("# case\ttarget\tstage\tform\tstatus\toutput\terror output\n")
            for (_, case, target), (result,) in zip(work, results):
                for (stage, form), entry in sorted(result.items()):
                    file.write("\t".join([case.name, target, stage, form, *digest(entry)]) + "\n")
        print(f"recorded {sum(len(r[0]) for r in results)} outputs of "
              f"{len(work)} cases ({len(cases)} control files)")
        return 0

    rows = []
    if args.command == "compare":
        for (_, case, target), (old, new) in zip(work, results):
            for key in sorted(new):
                same = old[key] == new[key]
                if not same:
                    show_difference(case.name, target, *key, old[key], new[key], args.diff_lines)
                rows.append((case.name, target, *key, same, new[key][0] != 0))
    else:
        recorded = {}
        with open(args.hashes) as file:
            if file.readline().rstrip("\n") != HASHES_HEADER:
                sys.exit(f"check.py: {args.hashes} is not a file of hashes of check.py")
            for line in file:
                if not line.startswith("#"):
                    name, target, stage, form, *entry = line.rstrip("\n").split("\t")
                    recorded[(name, target, stage, form)] = tuple(entry)
        for (_, case, target), (new,) in zip(work, results):
            for key in sorted(new):
                old = recorded.get((case.name, target, *key))
                same = old == digest(new[key])
                if not same:
                    print(f"DIFFERENT: {case.name} [{target}] {key[0]} {key[1]}"
                          + (": not recorded" if old is None else
                             f": status {old[0]} -> {new[key][0]}" if old[0] != str(new[key][0])
                             else ": the output differs from the recorded hash"))
                rows.append((case.name, target, *key, same, new[key][0] != 0))
    return summary(cases, rows)


if __name__ == "__main__":
    sys.exit(main())
