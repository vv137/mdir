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
  cases     the control files (*.toml) and Python scripts (*.py) of
            scripts/builder-equivalence/cases, written for what no other
            file reaches (#739): the instances of the templates of the
            lowering in single precision, the tiles of edge 4 and 16 of
            PME, and the potential of PME at the particles
--control FILE adds a control file of its own (the corpus "extra").

A control file may name environment variables for its emits, one a line:
`# check.py env: NAME=VALUE`. A case written in Python is run with the
interpreter of --python (by default this one) and the Python package of the
build (<tree>/python) on PYTHONPATH, as `SCRIPT TARGET PRECISION STAGE`
(PRECISION is DEFAULT without --precisions); it prints the module or the
lowered program, and has the program of a run alone.

Each control file is a case for each target of --targets (cpu and gpu by
default): a copy of the file with `target` set in [execution]. `mdir emit`
uses no device; the GPU target only changes the program. Each is a case
for each kind of program of --kinds as well: `run`, the program of the
run, and `segments`, the program of segments that a Python simulation of
the same model runs (`mdir emit --segments`, which builds the branches of
the start and the loop over the parts). Both by default; a build of
before `--segments` takes `--kinds run`. --precisions (none by default) makes each one
a case in each precision given, set in [execution] as the target is, and
each --env NAME=VALUE[,NAME=VALUE] one more case with those variables, beside
the case without them (#739). The label of a case in the report and in a
file of hashes is its target, kind, precision, and variables, for example
"gpu segments mixed env:MDIR_PME_SPREAD=tiles4"; without the two options it
is the label of before. A case runs in a
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
--instances, with the lowered stage, counts for each target the lowered
cases (of the last build) whose program holds each instance of the
templates of the lowering (lib/Runtime/Templates), by the names of their
functions: the neighbor matrix in f64 or f32, particle mesh Ewald by its
types and order (p64c32f32o4), and its variants (triclinic, prolate,
listed, potential, tiles of an edge).

A case is identical when the status, the output, and the error output are
the same in every form. Failed printed-form diagnostics ignore only generated
buffer positions and redundant source excerpts before complete operation
notes; messages, severities, operation dumps, parser excerpts, and user-file
locations remain checked. Raw failures are byte-exact. A case that stops
with the same error in both builds is identical too, and is counted as such. The
report names cases by their place in the corpus, never by a path of the
machine. The exit status is 0 when every case is identical, 1 when one
differs, and 2 for an error of the script.

--expected-errors FILE accepts only the complete old/new failed-output pairs
approved by ruling 10 (#721), keyed by case, target/kind, stage and form. Every
entry must be observed. Approved diagnostic changes are reported separately
from identical cases; successful outputs and all unlisted errors stay exact.
"""
import argparse
import difflib
import hashlib
import json
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


MESSAGE = ("mdir: a program of segments takes restart state through its activation, "
           "not a static restart control\n")


class ExpectedErrors:
    def __init__(self, path=None):
        self.entries = {}
        self.used = set()
        if path is None:
            return
        with open(path) as file:
            manifest = json.load(file)
        if set(manifest) != {"version", "issue", "entries"} or manifest["version"] != 1 or manifest["issue"] != 721:
            raise ValueError("expected-error manifest must describe ruling 10 / #721")
        for entry in manifest["entries"]:
            if set(entry) != {"case", "target", "stage", "form", "old", "new"}:
                raise ValueError("invalid expected-error entry")
            key = tuple(entry[k] for k in ("case", "target", "stage", "form"))
            if key in self.entries or key[1] not in ("cpu segments", "gpu segments") or key[2] not in ("module", "lowered") or key[3] != "printed":
                raise ValueError("duplicate or unsupported expected-error key")
            values = []
            for name in ("old", "new"):
                value = entry[name]
                if set(value) != {"status", "stdout", "stderr"} or type(value["status"]) is not int or value["status"] != 1 or value["stdout"] != "":
                    raise ValueError("only failed emits with empty stdout may be excepted")
                values.append((1, b"", value["stderr"].encode()))
            old, new = values
            if b"operand #0 does not dominate this use" not in old[2] or new[2] != MESSAGE.encode():
                raise ValueError("unapproved diagnostic exception")
            self.entries[key] = (old, new)

    def accept(self, key, old, new, hashes=False):
        expected = self.entries.get(key)
        if expected is None:
            return False
        before, after = expected
        if hashes:
            before = (str(before[0]), hashlib.sha256(before[1]).hexdigest(),
                      hashlib.sha256(before[2]).hexdigest())
        if old != before or new != after:
            return False
        self.used.add(key)
        print(f"EXPECTED-DIAGNOSTIC: {key[0]} [{key[1]}] {key[2]} {key[3]}")
        return True

    def complete(self):
        unused = self.entries.keys() - self.used
        for key in sorted(unused):
            print("UNUSED-EXPECTED-DIAGNOSTIC: " + repr(key))
        return not unused


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
    kinds = set(args.only or ["examples", "tests", "bench", "cases", "extra"])
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
    if "cases" in kinds:
        # The control files and Python scripts written for the corpus
        # (#739), for what no other file of it reaches.
        directory = os.path.join(args.root, "scripts", "builder-equivalence", "cases")
        if os.path.isdir(directory):
            for file in sorted(os.listdir(directory)):
                if file.endswith((".toml", ".py")):
                    relative = os.path.join("scripts", "builder-equivalence", "cases", file)
                    cases.append(Case(f"cases/{file}", args.root, relative))
    if "extra" in kinds:
        for path in args.control or []:
            path = os.path.abspath(path)
            cases.append(Case(f"extra/{os.path.basename(path)}", os.path.dirname(path),
                              os.path.basename(path)))
    return cases


def with_execution(text, key, value):
    """The control file with `key = value` set in [execution]."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    setting = f"{key} = {value}\n"
    header = next((i for i, line in enumerate(lines)
                   if re.match(r"^\s*\[execution\]\s*(#.*)?$", line)), None)
    if header is None:
        return "".join(lines) + "\n[execution]\n" + setting
    end = next((i for i in range(header + 1, len(lines))
                if re.match(r"^\s*\[", lines[i])), len(lines))
    for i in range(header + 1, end):
        if re.match(rf"^\s*{key}\s*=", lines[i]):
            lines[i] = setting
            return "".join(lines)
    lines.insert(header + 1, setting)
    return "".join(lines)


def with_target(text, target):
    """The control file with `target` of [execution] set."""
    return with_execution(text, "target", f'"{target.upper()}"')


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


def run(command, cwd, hide, input=None, env=None):
    try:
        done = subprocess.run(command, cwd=cwd, input=input, capture_output=True,
                              timeout=1800, env=env)
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


def require_materialized(output):
    """A verified emission must not retain a builder's internal connection."""
    if re.search(rb"\bmdir\.term_ir\s*=", output):
        raise RuntimeError("internal term IR connection survived emission")


class Variant:
    """What a case is emitted for: a target, a kind of program, and, when
    asked for, a precision (--precisions) and a set of environment
    variables (--env). Its label names it in the report and in a file of
    hashes; without a precision or an environment it is the label of
    before, "cpu" or "gpu segments", so that older files of hashes still
    apply."""

    def __init__(self, target, kind="run", precision=None, env=()):
        self.target, self.kind, self.precision, self.env = target, kind, precision, tuple(env)

    def label(self):
        words = [self.target]
        if self.kind != "run":
            words.append(self.kind)
        if self.precision:
            words.append(self.precision)
        if self.env:
            words.append("env:" + ",".join(f"{name}={value}" for name, value in self.env))
        return " ".join(words)


def case_env(text):
    """The environment that a control file of the corpus asks for, on lines
    `# check.py env: NAME=VALUE` (scripts/builder-equivalence/cases)."""
    env = []
    for match in re.finditer(r"(?m)^#\s*check\.py env:\s*(\S+)\s*$", text):
        name, _, value = match.group(1).partition("=")
        env.append((name, value))
    return env


def python_dir(mdir):
    """The Python package of the build of an mdir program: <tree>/python."""
    return os.path.join(os.path.dirname(os.path.dirname(mdir)), "python")


def outputs(mdir, printer, case, variant, stages, forms, scratch, index, python=None):
    """The outputs of one build for one case and variant:
    {(stage, form): (status, out, err)}."""
    target, kind = variant.target, variant.kind
    place = os.path.join(scratch, f"{index}-{hashlib.sha1((variant.label() + mdir).encode()).hexdigest()[:12]}")
    directory, hide = mirror(case, place)
    env = dict(os.environ)
    env.update(case_env(source_text(case)))
    env.update(variant.env)
    if case.relative.endswith(".py"):
        # A case written in Python (scripts/builder-equivalence/cases): the
        # script builds a program with the Python package of the build and
        # prints its module or its lowered program.
        env["PYTHONPATH"] = os.pathsep.join(
            [python_dir(mdir)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        script = os.path.basename(case.relative)

        def emit(stage):
            return run([python or sys.executable, script, target.upper(),
                        (variant.precision or "default").upper(), stage],
                       directory, hide, env=env)
    else:
        stem = os.path.splitext(os.path.basename(case.relative))[0]
        control = f"{stem}.equivalence-{target}.toml"
        text = with_target(source_text(case), target)
        if variant.precision:
            text = with_execution(text, "precision", f'"{variant.precision.upper()}"')
        with open(os.path.join(directory, control), "w", encoding="utf-8",
                  errors="surrogateescape") as file:
            file.write(text)
        segments = ["--segments"] if kind == "segments" else []

        def emit(stage):
            return run([mdir, "emit", f"--stage={stage}", *segments, control],
                       directory, hide, env=env)
    result = {}
    for stage in stages:
        status, out, err = emit(stage)
        if stage == "lowered":
            if status == 0:
                require_materialized(out)
            result[(stage, "printed")] = printed_result((status, out, err))
            continue
        if "raw" in forms:
            result[(stage, "raw")] = (status, out, err)
        if stage == "module" and "printed" in forms:
            if status == 0:
                printed = run([printer, "-"], directory, hide, input=out)
                if printed[0] == 0:
                    require_materialized(printed[1])
                result[(stage, "printed")] = printed_result(printed)
            else:
                result[(stage, "printed")] = printed_result((status, b"", err))
    shutil.rmtree(place, ignore_errors=True)
    return result


def digest(entry):
    status, out, err = entry
    return (str(status), hashlib.sha256(out).hexdigest(), hashlib.sha256(err).hexdigest())


TEMPLATE_NAME = re.compile(rb"@(mdrt[._][A-Za-z0-9_.]*)")
PME_SUFFIX = re.compile(r"_(p(?:32|64)c(?:32|64)f(?:32|64)o[0-9]+)(?:_|$)")


def instances(text):
    """The instances of the templates of the lowering
    (lib/Runtime/Templates) that a lowered program holds, by the names of
    their functions: the neighbor matrix in f64 or f32, particle mesh Ewald
    by its types and order, and its variants that are added only where a
    sum asks for them."""
    keys = set()
    for name in {m.decode() for m in TEMPLATE_NAME.findall(text)}:
        if re.match(r"mdrt(\.|_gpu_)build_neighbors_matrix", name):
            keys.add("matrix " + ("f32" if re.search(r"_f32(_|$)", name) else "f64"))
        elif name.startswith("mdrt_gpu_build_neighbors_groups"):
            keys.add("groups triclinic" if "_triclinic" in name else "groups")
        if "pme" not in name:
            continue
        suffix = PME_SUFFIX.search(name)
        if suffix:
            keys.add("pme " + suffix.group(1))
        if name.startswith("mdrt_gpu_pme") and "_triclinic_p" in name:
            keys.add("pme triclinic")
        if name.startswith("mdrt_gpu_pme_prolate"):
            keys.add("pme prolate")
        if "_listed_p" in name:
            keys.add("pme listed")
        if "pme_potential" in name:
            keys.add("pme potential")
        tiles = re.search(r"_spread_tiles([0-9]+)_", name)
        if tiles:
            keys.add(f"pme tiles {tiles.group(1)}")
    return keys


def report_instances(work, results):
    """For each target and instance, the cases (of the last build) whose
    lowered program holds it."""
    counts = {}
    for (_, case, variant), result in zip(work, results):
        status, out, _ = result[-1].get(("lowered", "printed"), (1, b"", b""))
        if status != 0:
            continue
        for key in instances(out):
            counts[(variant.target, key)] = counts.get((variant.target, key), 0) + 1
    print("instances (lowered cases that hold each):")
    for (target, key), count in sorted(counts.items()):
        print(f"  {target} {key}: {count}")


def summary(cases, rows, expected=None):
    """rows: (case, target, stage, form, same, failed)."""
    keys = {(case, target) for case, target, *_ in rows}
    changed = {(case, target) for case, target, _, _, same, _ in rows if not same}
    accepted = expected.used if expected else set()
    different = {(case, target) for case, target, stage, form, same, _ in rows
                 if not same and (case, target, stage, form) not in accepted}
    failed = {(case, target) for case, target, _, _, same, failed in rows if same and failed}
    stages = sorted({(stage, form) for _, _, stage, form, _, _ in rows})
    print(f"control files: {len(cases)}")
    print(f"cases (control file and variant): {len(keys)}")
    print("compared: " + ", ".join(f"{stage} {form}" for stage, form in stages))
    print(f"identical: {len(keys) - len(changed)}"
          f" (of which stop with the same error in both: {len(failed - changed)})")
    if accepted:
        print(f"expected diagnostic changes: {len({key[:2] for key in accepted})} cases / {len(accepted)} outputs")
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
    parser.add_argument("--only", nargs="+", choices=["examples", "tests", "bench", "cases", "extra"])
    parser.add_argument("--bench", help="the prepared Amber suite (MDIR_BENCH_DIR)")
    parser.add_argument("--control", action="append", help="a control file of its own")
    parser.add_argument("--root", default=ROOT, help="the repository (this one)")
    parser.add_argument("--targets", default="cpu,gpu")
    parser.add_argument("--kinds", default="run,segments",
                        help="the programs: of the run, and of segments (mdir emit --segments)")
    parser.add_argument("--precisions", default="",
                        help="single, mixed, double: each case in each, set in [execution]")
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE[,NAME=VALUE]",
                        help="a set of environment variables: each case with it as well")
    parser.add_argument("--python", help="the interpreter of the cases written in Python")
    parser.add_argument("--instances", action="store_true",
                        help="count the cases whose lowered program holds each template instance")
    parser.add_argument("--stages", default="module")
    parser.add_argument("--forms", default="raw,printed")
    parser.add_argument("--printer", help="mdir-opt for the printed form")
    parser.add_argument("--scratch", help="where the cases run (a temporary directory)")
    parser.add_argument("--jobs", "-j", type=int, default=8)
    parser.add_argument("--diff-lines", type=int, default=20)
    parser.add_argument("--show-errors", action="store_true",
                        help="name each case that stops with an error, and its first line")
    parser.add_argument("--expected-errors", help="exact approved diagnostic manifest (#721)")
    args = parser.parse_args()
    if args.expected_errors and args.command == "record":
        parser.error("--expected-errors is only for compare or verify")
    try:
        expected = ExpectedErrors(args.expected_errors)
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.error(str(error))

    count = 2 if args.command == "compare" else 1
    if len(args.programs) != count:
        parser.error(f"{args.command} takes {count} mdir program(s)")
    programs = [program(path) for path in args.programs]
    targets = [t for t in args.targets.split(",") if t]
    kinds = [k for k in args.kinds.split(",") if k]
    stages = [s for s in args.stages.split(",") if s]
    forms = [f for f in args.forms.split(",") if f]
    precisions = [p for p in args.precisions.split(",") if p]
    if set(precisions) - {"single", "mixed", "double"}:
        parser.error("--precisions takes some of double, mixed, single")
    environments = [()]
    for value in args.env:
        pairs = [pair for pair in value.split(",") if pair]
        if not pairs or any("=" not in pair or not pair.split("=", 1)[0] for pair in pairs):
            parser.error(f"--env takes NAME=VALUE[,NAME=VALUE], not '{value}'")
        environments.append(tuple(tuple(pair.split("=", 1)) for pair in pairs))
    if args.instances and "lowered" not in stages:
        parser.error("--instances needs the lowered stage (--stages lowered)")
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
        # A case of the program of segments is named "<target> segments";
        # a precision and a set of variables add to the name (Variant). A
        # case written in Python has the program of a run alone.
        work = [(i, case, Variant(target, kind, precision, env))
                for i, case in enumerate(cases) for target in targets for kind in kinds
                for precision in (precisions or [None]) for env in environments
                if not (kind == "segments" and case.relative.endswith(".py"))]
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            results = list(pool.map(
                lambda item: [outputs(mdir, printer, item[1], item[2], stages, forms,
                                      scratch, item[0], args.python) for mdir in programs], work))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if args.show_errors:
        for (_, case, variant), result in zip(work, results):
            for (stage, form), (status, _, err) in sorted(result[-1].items()):
                if status != 0 and (form == forms[0] or stage == "lowered"):
                    first = err.decode(errors="replace").strip().splitlines()[:1]
                    print(f"ERROR: {case.name} [{variant.label()}] {stage}: {first[0] if first else status}")

    if args.command == "record":
        with open(args.output, "w") as file:
            file.write(HASHES_HEADER + "\n")
            file.write("# case\ttarget\tstage\tform\tstatus\toutput\terror output\n")
            for (_, case, variant), (result,) in zip(work, results):
                for (stage, form), entry in sorted(result.items()):
                    file.write("\t".join([case.name, variant.label(), stage, form, *digest(entry)]) + "\n")
        print(f"recorded {sum(len(r[0]) for r in results)} outputs of "
              f"{len(work)} cases ({len(cases)} control files)")
        if args.instances:
            report_instances(work, results)
        return 0

    rows = []
    if args.command == "compare":
        for (_, case, variant), (old, new) in zip(work, results):
            target = variant.label()
            for key in sorted(new):
                same = old[key] == new[key]
                if not same and not expected.accept((case.name, target, *key), old[key], new[key]):
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
        for (_, case, variant), (new,) in zip(work, results):
            target = variant.label()
            for key in sorted(new):
                old = recorded.get((case.name, target, *key))
                same = old == digest(new[key])
                if not same and not expected.accept((case.name, target, *key), old, new[key], hashes=True):
                    print(f"DIFFERENT: {case.name} [{target}] {key[0]} {key[1]}"
                          + (": not recorded" if old is None else
                             f": status {old[0]} -> {new[key][0]}" if old[0] != str(new[key][0])
                             else ": the output differs from the recorded hash"))
                rows.append((case.name, target, *key, same, new[key][0] != 0))
    complete = expected.complete()
    result = summary(cases, rows, expected)
    if args.instances:
        report_instances(work, results)
    return result if complete else 1


if __name__ == "__main__":
    sys.exit(main())
