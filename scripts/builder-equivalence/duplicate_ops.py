#!/usr/bin/env python3
"""duplicate_ops.py: the costly ops that fused kernels still compute twice.

The measurement of #681. The pipeline (lib/Compiler/Compile.cpp) fuses the
loops over the pairs of one neighbor structure and over the tuples of one
incidence (md-exec-fuse-loops, D139, D150), merges their destinations
(md-exec-accumulate-destinations, D157), and leaves to `canonicalize,cse`
what the fused kernels compute twice. `cse` merges only ops that are equal
as written. This script counts, in each kernel of a `md_exec.pair_for` or a
`md_exec.tuple_for`, the costly ops and those among them that a cheap
rewrite would share, and gives the reason why the loops of one block that
are left apart were not fused.

  duplicate_ops.py corpus MDIR [corpus options]  the corpus of check.py
  duplicate_ops.py count FILE.mlir...             modules already lowered

MDIR is an `mdir` program or a build tree; mdir-opt (--printer, by default
the one beside MDIR) runs the passes. The corpus is that of check.py
(--only examples tests bench extra, --bench, --control, --root), and the
kind "ml": the frozen model in stages of the tests (painn-scalar-export.mlir)
beside the classical terms of the peptide of the tests, in a box. Each
control file is a case for each target of --targets and each precision of
--precisions (both of each by default): a copy with `target` and
`precision` set in [execution]. For each case the script takes the module
(`mdir emit --stage=module`) and the pipeline (`mdir emit --stage=pipeline`)
and runs the passes of the pipeline up to the cut of --cut:

  cse      after `md-exec-fold-tables,canonicalize,cse`, the second `cse`
           after the fusion, as #681 names it: no precision is assigned
           by the passes yet, but the module that the driver builds may
           already differ with the precision (the constraints do)
  md-exec  the end of the md_exec stage: after `md-exec-assign-algebra`,
           with the precision assigned and, with fast math, the
           approximations and the radial tables (md-exec-approximate,
           md-exec-expand-radial) and their `cse` (the default)

The module is printed with unique SSA names (--mlir-print-unique-ssa-ids).

Costly ops: math.exp, math.erfc, math.sqrt, math.rsqrt, arith.divf,
math.powf, table lookups (md.lookup, and the loads of the radial tables
that md-exec-expand-radial makes, from a `memref.get_global` of a table),
and, beside these, math.fpowi (an integer power, a chain of products).

Duplicates among them, by kind (each counts the ops a rewrite would save):
  same      the same op on the same operands (in sibling regions, where
            cse does not reach)
  sqrt      sqrt(x) beside rsqrt(x), or beside c / sqrt(x)
  exp       exp(a) beside exp(-a), or exp(a + b) beside exp(a) or exp(b)
  divisor   x / y and z / y with the same y: one reciprocal and products
            (n divisions by one y count n - 1)
  constant  x / c with a constant c: a product by 1/c
  powers    fpowi(x, m) and fpowi(x, n) of one base: one chain of products
            (n powers of one base count n - 1)
  index     loads of n radial tables at the same index: one table of their
            columns side by side, one load (counts n - 1)
Only `same` keeps the bits as they are; the others change the rounding.

Unfused loops: of each two loops of one kind in one block (in the order of
the block), the first condition of md-exec-fuse-loops that they fail:
  structure  another neighbor structure or incidence
  refresh    the same neighbor structure, refreshed between the two
  positions  other positions (another update of them between the two)
  cell       another cell
  cutoff     another cutoff
  policy     another traversal, conflict, centers, images (pairs)
  arity      another arity, disjoint set, flag or overwrite (tuples)
  dependence the second uses a result of the first, or an op between them
             does and the second cannot move up

Kernels are counted once for each distinct body (the same body in the
start, the step, and the segments is one kernel, with its copies), and
named by the set they run over: the incidence of a loop over tuples
(@bonds, @pairs14, ...) or the cutoff of a loop over pairs. The report is
a line per case (--kernels adds a line per kernel), then a line for the
whole corpus for each target and precision, which counts each distinct
kernel once; --json writes every kernel. The exit status is
0, or 2 for an error of the script. emit uses no device.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check  # noqa: E402

COSTLY = ("math.exp", "math.erfc", "math.sqrt", "math.rsqrt", "arith.divf",
          "math.powf", "table", "math.fpowi")
SHORT = {"math.exp": "exp", "math.erfc": "erfc", "math.sqrt": "sqrt",
         "math.rsqrt": "rsqrt", "arith.divf": "div", "math.powf": "powf",
         "table": "table", "math.fpowi": "powi"}
KINDS = ("same", "sqrt", "exp", "divisor", "constant", "powers", "index")
REASONS = ("structure", "refresh", "positions", "cell", "cutoff", "policy", "arity",
           "dependence")

NAME = r"%[\w.$-]+"
USE = re.compile(r"(%[\w.$-]+)(?:#(\d+))?")
OP = re.compile(r"^\s*(?:(" + NAME + r"(?::\d+)?(?:\s*,\s*" + NAME +
                r"(?::\d+)?)*)\s*=\s*)?\"?([a-z_][\w]*\.[\w.]+)\"?(.*)$")
CONSTANT = re.compile(r"^\s*(" + NAME + r")\s*=\s*arith\.constant\s+(?:dense<)?"
                      r"([-+0-9.eE]+|0x[0-9A-Fa-f]+)")
LOOP = re.compile(r"\bmd_exec\.(pair_for|tuple_for)\b")


class Op:
    def __init__(self, results, name, rest):
        self.results, self.name, self.rest = results, name, rest
        # The operands: the uses before the first `:`, `{`, or `(` of a
        # clause, at the outermost level.
        head = re.split(r"\s:\s|\s\{|\bfastmath<", rest)[0]
        self.operands = [m.group(0) for m in USE.finditer(head)]
        attribute = re.search(r"@[\w.$-]+", rest)
        self.symbol = attribute.group(0) if attribute else None


def parse_results(text):
    """The names of the results of an op: `%a`, `%a:3` gives %a#0..2."""
    names = []
    if not text:
        return names
    for part in text.split(","):
        part = part.strip()
        if ":" in part:
            base, count = part.split(":")
            names += [f"{base}#{i}" for i in range(int(count))] + [base]
        else:
            names.append(part)
    return names


def strip_strings(line):
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', line)


class Loop:
    """A pair_for or tuple_for: its header, the ops of its kernel."""

    def __init__(self, kind, header, results, block, depth, parents):
        self.kind, self.header, self.results = kind, header, results
        self.block, self.depth, self.parents = block, depth, parents
        self.ops, self.lines = [], []
        self.uses = set()
        self.closing = ""

    def name(self):
        """The set the loop runs over: the incidence of a loop over tuples
        (@bonds, @pairs14, ...), the cutoff of a loop over pairs."""
        if self.kind == "tuple_for":
            m = re.search(r"!mdrt\.incidence<@[\w.$-]+, \d+, @([\w.$-]+)>", self.closing)
            return "@" + m.group(1) if m else "?"
        m = re.search(r"cutoff\(([^)]*)\)", self.header)
        return f"pairs cutoff {float(m.group(1)):g}" if m else "pairs"


def parse_module(text):
    """The loops of a printed module, with the constants of the module.

    Every line that ends with `{` opens a region; a line that begins with
    `}` closes one, and opens the next region of the same op if it ends
    with `{` as well (`} else {`, `} do {`, `}, {`). An op inside the
    region of a loop belongs to the innermost loop around it."""
    constants = {}
    loops = []
    # The op that defines each value of the module (one dict for each
    # parse: the cases run on threads).
    definitions = {}
    # The open regions, innermost last: (region id, loop or None).
    stack = []
    regions = [0]

    def new_region():
        regions[0] += 1
        return regions[0]

    for raw in text.splitlines():
        line = strip_strings(raw)
        m = CONSTANT.match(line)
        if m:
            value = m.group(2)
            try:
                constants[m.group(1)] = float.fromhex(value) if value.startswith("0x") \
                    else float(value)
            except ValueError:
                pass
        stripped = line.strip()
        loop = next((entry[1] for entry in reversed(stack) if entry[1] is not None), None)
        if stripped.startswith("}"):
            top = stack.pop() if stack else (0, None)
            if top[1] is not None and not any(e[1] is top[1] for e in stack):
                top[1].closing = line
            if stripped.endswith("{"):
                stack.append((new_region(), top[1]))
            continue
        m = OP.match(line)
        if loop is not None:
            loop.uses.update(u.group(0) for u in USE.finditer(line))
        if m is None:
            if stripped.endswith("{"):
                stack.append((new_region(), None))
            continue
        results = parse_results(m.group(1))
        name = m.group(2)
        op = Op(results, name, m.group(3))
        for r in results:
            definitions[r] = op
        if loop is not None:
            loop.ops.append(op)
            loop.lines.append(raw)
        if stripped.endswith("{"):
            new_loop = None
            if name in ("md_exec.pair_for", "md_exec.tuple_for") and loop is None:
                block = stack[-1][0] if stack else 0
                new_loop = Loop(name.split(".")[1], m.group(3), results, block, len(stack), [])
                new_loop.definitions = definitions
                loops.append(new_loop)
            stack.append((new_region(), new_loop))
    return loops, constants


def normalized(loop):
    """The body of a loop with its names numbered in order: equal bodies
    are one kernel."""
    if getattr(loop, "body", None) is not None:
        return loop.body
    names = {}

    def rename(match):
        key = match.group(1)
        if key not in names:
            names[key] = f"%v{len(names)}"
        return names[key] + (f"#{match.group(2)}" if match.group(2) else "")
    loop.body = "\n".join(USE.sub(rename, line.strip()) for line in loop.lines)
    return loop.body


def count_kernel(loop, constants):
    """The costly ops of a kernel and the duplicates among them."""
    defs = {}
    globals_ = {}
    for op in loop.ops:
        for r in op.results:
            defs[r] = op
        if op.name == "memref.get_global" and op.symbol:
            for r in op.results:
                globals_[r] = op.symbol
    local_constants = {}
    for op in loop.ops:
        if op.name == "arith.constant":
            m = re.match(r"\s*(?:dense<)?([-+0-9.eE]+)", op.rest)
            if m:
                for r in op.results:
                    try:
                        local_constants[r] = float(m.group(1))
                    except ValueError:
                        pass
    known = dict(constants)
    known.update(local_constants)

    costly = Counter()
    entries = []  # (kind, key operands, op)
    for op in loop.ops:
        kind = None
        if op.name in COSTLY:
            kind = op.name
        elif op.name == "md.lookup":
            kind = "table"
        elif op.name in ("vector.load", "memref.load") and op.operands and \
                op.operands[0] in globals_ and "table" in globals_[op.operands[0]]:
            kind = "table"
        if kind is None:
            continue
        costly[kind] += 1
        if kind == "table" and op.name != "md.lookup":
            key = (globals_[op.operands[0]],) + tuple(op.operands[1:])
        elif op.name == "math.fpowi":
            key = tuple(op.operands)
        else:
            key = tuple(op.operands)
        entries.append((kind, key, op))

    duplicates = Counter()
    # same: the same op on the same operands.
    seen = Counter((kind, key) for kind, key, _ in entries)
    for (kind, key), n in seen.items():
        duplicates["same"] += n - 1
    unique = {}
    for kind, key, op in entries:
        unique.setdefault((kind, key), op)

    def of(kind):
        return [(key, op) for (k, key), op in unique.items() if k == kind]

    sqrts = {key[0]: op for key, op in of("math.sqrt") if key}
    rsqrts = {key[0] for key, op in of("math.rsqrt") if key}
    divisions = of("arith.divf")
    divisors = {key[1] for key, op in divisions if len(key) == 2}
    # One rsqrt(x) gives sqrt(x) = x rsqrt(x) and 1/sqrt(x): the sqrt and
    # the rsqrt, or the sqrt and a division by it, count once for each x.
    for x, op in sqrts.items():
        if x in rsqrts or any(r in divisors for r in op.results):
            duplicates["sqrt"] += 1
    # Loads of other tables at the same index: one table of their columns
    # side by side would be one load.
    indices = defaultdict(set)
    for (kind, key), op in unique.items():
        if kind == "table" and op.name != "md.lookup":
            indices[key[1:]].add(key[0])
    for tables in indices.values():
        duplicates["index"] += len(tables) - 1

    def negation_of(value):
        """The value whose negation `value` is, if it is one."""
        op = defs.get(value)
        if op is None:
            return None
        if op.name == "arith.negf" and op.operands:
            return op.operands[0]
        if op.name == "arith.mulf" and len(op.operands) == 2:
            a, b = op.operands
            if known.get(b) == -1.0:
                return a
            if known.get(a) == -1.0:
                return b
        if op.name == "arith.subf" and len(op.operands) == 2 and known.get(op.operands[0]) == 0.0:
            return op.operands[1]
        return None
    exps = {key[0] for key, _ in of("math.exp") if key}
    for a in exps:
        b = negation_of(a)
        if b is not None and b in exps:
            duplicates["exp"] += 1
            continue
        op = defs.get(a)
        if op is not None and op.name in ("arith.addf", "arith.subf") and len(op.operands) == 2:
            x, y = op.operands
            if x in exps or (op.name == "arith.addf" and y in exps):
                duplicates["exp"] += 1
            elif op.name == "arith.subf" and any(negation_of(e) == y for e in exps):
                duplicates["exp"] += 1
    by_divisor = defaultdict(set)
    for key, op in divisions:
        if len(key) == 2:
            by_divisor[key[1]].add(key[0])
    for divisor, numerators in by_divisor.items():
        if divisor in known:
            duplicates["constant"] += len(numerators)
        elif len(numerators) > 1:
            duplicates["divisor"] += len(numerators) - 1
    bases = defaultdict(set)
    for key, op in of("math.fpowi"):
        if len(key) == 2:
            bases[key[0]].add(key[1])
    for base, exponents in bases.items():
        duplicates["powers"] += len(exponents) - 1
    return costly, duplicates


def header_of(loop):
    """The operands and clauses of the header of a loop."""
    text = loop.header
    operands = []
    for part in text.split(" ins(")[0].split(" outs(")[0].split(" coordinates(")[0] \
            .split(" tuple(")[0].split(" reduce(")[0].split(" cutoff(")[0].split(","):
        part = part.strip()
        if part.startswith("%"):
            operands.append(part.split()[0])
    clauses = {}
    for m in re.finditer(r"\b([a-z_]+)\(([^()]*(?:\([^()]*\)[^()]*)*)\)", text):
        clauses[m.group(1)] = m.group(2)
    words = set(re.findall(r"\)\s+([a-z_]+)(?=\s|$|\{)", text))
    reduce_types = ""
    m = re.search(r"reduce\([^:]*:\s*([^)]*)\)", text)
    if m:
        reduce_types = m.group(1)
    return operands, clauses, words, reduce_types


def _root(value, definitions):
    """The structure that a chain of refreshes starts from."""
    while value in definitions and definitions[value].name == "md_exec.refresh_neighbors":
        value = definitions[value].operands[0]
    return value


def unfused_reason(first, second):
    a_ops, a_cl, a_words, a_red = header_of(first)
    b_ops, b_cl, b_words, b_red = header_of(second)
    if a_ops[:1] != b_ops[:1]:
        # The same structure, refreshed between the two (or before each).
        definitions = first.definitions
        if a_ops and b_ops and _root(a_ops[0], definitions) == _root(b_ops[0], definitions):
            return "refresh"
        return "structure"
    if a_ops[1:2] != b_ops[1:2]:
        return "positions"
    if a_ops[2:3] != b_ops[2:3]:
        return "cell"
    if first.kind == "pair_for":
        if a_cl.get("cutoff") != b_cl.get("cutoff"):
            return "cutoff"
        for clause in ("policy", "centers", "images", "traversal", "conflict"):
            if a_cl.get(clause) != b_cl.get(clause):
                return "policy"
        if (a_words ^ b_words) & {"ghost_free", "images"}:
            return "policy"
    else:
        if a_cl.get("arity") != b_cl.get("arity") or \
                ("disjoint" in a_words) != ("disjoint" in b_words) or \
                "i1" in a_red or "i1" in b_red or "overwrite" in a_words | b_words:
            return "arity"
    return "dependence"


def analyze(text):
    """The kernels and the unfused loops of a printed module."""
    loops, constants = parse_module(text)
    kernels = {}
    for loop in loops:
        body = normalized(loop)
        if body not in kernels:
            costly, duplicates = count_kernel(loop, constants)
            kernels[body] = {"kind": loop.kind, "name": loop.name(),
                             "body": hashlib.sha256(body.encode()).hexdigest()[:16], "copies": 0, "costly": dict(costly),
                             "duplicates": dict(duplicates), "ops": len(loop.ops),
                             "depth": loop.depth}
        kernels[body]["copies"] += 1
        kernels[body]["depth"] = max(kernels[body]["depth"], loop.depth)
    # Each loop after the first of its kind in a block is left apart from
    # the loops before it; its reason is that against the closest of them,
    # the one that meets the most conditions of the fusion. A pair of
    # bodies counts once, as the kernels do.
    unfused = Counter()
    seen = set()
    by_block = defaultdict(list)
    for loop in loops:
        by_block[(loop.block, loop.kind)].append(loop)
    for (block, kind), members in by_block.items():
        for j in range(1, len(members)):
            reasons = [(REASONS.index(unfused_reason(members[i], members[j])), i)
                       for i in range(j)]
            rank, i = max(reasons)
            key = (kind, rank, normalized(members[i]), normalized(members[j]))
            if key in seen:
                continue
            seen.add(key)
            unfused[(kind, REASONS[rank])] += 1
    return list(kernels.values()), unfused


def totals(kernels):
    costly, duplicates = Counter(), Counter()
    for k in kernels:
        costly.update(k["costly"])
        duplicates.update(k["duplicates"])
    return costly, duplicates


def row(name, kernels, unfused):
    costly, duplicates = totals(kernels)
    pairs = sum(1 for k in kernels if k["kind"] == "pair_for")
    tuples = len(kernels) - pairs
    cost = " ".join(f"{SHORT[k]}={costly[k]}" for k in COSTLY if costly[k])
    dup = " ".join(f"{k}={duplicates[k]}" for k in KINDS if duplicates[k])
    loops = " ".join(f"{kind.split('_')[0]}:{reason}={n}"
                     for (kind, reason), n in sorted(unfused.items()))
    return (f"{name}: kernels pair={pairs} tuple={tuples}; costly {cost or 'none'}; "
            f"duplicates {dup or 'none'}; unfused {loops or 'none'}")


def with_setting(text, key, value):
    """The control file with `key` of [execution] set to `value`."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    setting = f'{key} = "{value}"\n'
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


def cut_pipeline(pipeline, cut):
    if cut == "cse":
        marker = "md-exec-fold-tables,canonicalize,cse"
        index = pipeline.find(marker)
        if index < 0:
            return None
        return pipeline[:index + len(marker)]
    index = pipeline.find("md-exec-assign-algebra")
    if index < 0:
        return None
    end = pipeline.find(",", index + len("md-exec-assign-algebra"))
    # The options of the pass, if any, end at their brace.
    rest = pipeline[index:]
    if rest.startswith("md-exec-assign-algebra{"):
        end = index + rest.index("}") + 1
    return pipeline[:end] if end > 0 else pipeline


ML_CONTROL = """# The frozen PaiNN export of the tests (one feature, no message layers,
# test/Driver/Inputs/model-stages) beside the classical terms of the
# peptide, in a periodic box: a model in stages beside the pair loop.
[input]
topology = "peptide.prmtop"
coordinates = "peptide.inpcrd"
[output]
energy_interval = 1
[energy]
cutoff = 9.0
pairlist_distance = 10.0
electrostatics = "PME"
[[energy.model]]
name = "painn"
backend = "STAGES"
path = "painn.mlir"
[dynamics]
integrator = "VELOCITY_VERLET"
time_step = 0.001
steps = 2
[ensemble]
ensemble = "NVE"
temperature = 300.0
[constraints]
hydrogen_bonds = false
rigid_water = false
[boundary]
type = "PERIODIC"
"""


def ml_cases(root):
    """The case of the kind "ml": the frozen model in stages of the tests
    with the peptide of the tests, in a box of 40 A."""
    inputs = os.path.join(root, "test", "Driver", "Inputs")
    with open(os.path.join(inputs, "gb", "peptide.inpcrd"), encoding="utf-8") as file:
        coordinates = file.read().rstrip() + "\n" + "".join(
            f"{v:12.7f}" for v in (40.0, 40.0, 40.0, 90.0, 90.0, 90.0)) + "\n"
    parts = {"painn.toml": ML_CONTROL, "peptide.inpcrd": coordinates}
    copies = [(os.path.join(inputs, "gb", "peptide.prmtop"), "peptide.prmtop"),
              (os.path.join(inputs, "model-stages", "painn-scalar-export.mlir"), "painn.mlir")]
    return [check.Case("ml/painn.toml", inputs, "painn.toml", parts=parts, copies=copies)]


def run_case(mdir, printer, case, target, precision, cut, scratch, index, keep=None):
    place = os.path.join(scratch, f"case-{index}")
    os.makedirs(place)
    directory, hide = check.mirror(case, place)
    text = with_setting(with_setting(check.source_text(case), "target", target.upper()),
                        "precision", precision.upper())
    if target == "cpu":
        # The groups and their dual list (pruned_distance) are of a device
        # alone; on the CPU the case takes the structure the driver chooses.
        text = re.sub(r'(?m)^\s*neighbor_structure\s*=\s*"GROUPS".*\n', "", text)
        text = re.sub(r'(?m)^\s*pruned_distance\s*=.*\n', "", text)
    control = os.path.join(directory, "duplicate-ops.toml")
    with open(control, "w", encoding="utf-8") as file:
        file.write(text)
    outputs = {}
    for stage in ("module", "pipeline"):
        result = subprocess.run([mdir, "emit", control, f"--stage={stage}"], cwd=directory,
                                capture_output=True, text=True, errors="replace")
        if result.returncode != 0:
            first = (result.stderr.strip().splitlines() or ["error"])[0]
            return None, first.replace(hide, "<corpus>")
        outputs[stage] = result.stdout
    pipeline = cut_pipeline(outputs["pipeline"].strip(), cut)
    if pipeline is None:
        return None, "no md_exec stage in the pipeline"
    module = os.path.join(place, "module.mlir")
    with open(module, "w", encoding="utf-8") as file:
        file.write(outputs["module"])
    result = subprocess.run([printer, f"--pass-pipeline=builtin.module({pipeline})",
                             "--mlir-print-unique-ssa-ids", module], cwd=directory,
                            capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        first = (result.stderr.strip().splitlines() or ["error"])[0]
        return None, "mdir-opt: " + first.replace(hide, "<corpus>")
    if keep:
        name = re.sub(r"[^\w.-]+", "_", f"{case.name}-{target}-{precision}") + ".mlir"
        with open(os.path.join(keep, name), "w", encoding="utf-8") as file:
            file.write(result.stdout)
    return analyze(result.stdout), None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    corpus = sub.add_parser("corpus")
    corpus.add_argument("mdir")
    corpus.add_argument("--printer")
    corpus.add_argument("--root", default=check.ROOT)
    corpus.add_argument("--only", nargs="+", choices=["examples", "tests", "bench", "extra", "ml"])
    corpus.add_argument("--bench")
    corpus.add_argument("--control", action="append")
    corpus.add_argument("--targets", default="cpu,gpu")
    corpus.add_argument("--precisions", default="mixed,double")
    corpus.add_argument("--cut", choices=["cse", "md-exec"], default="md-exec")
    corpus.add_argument("--jobs", type=int, default=8)
    corpus.add_argument("--match", help="only the cases whose name this expression finds")
    corpus.add_argument("--json")
    corpus.add_argument("--keep", help="a directory to write the modules at the cut to")
    corpus.add_argument("--kernels", action="store_true", help="a line for each kernel")
    corpus.add_argument("--errors", action="store_true",
                        help="list the cases that stop with an error")
    count = sub.add_parser("count")
    count.add_argument("files", nargs="+")
    count.add_argument("--kernels", action="store_true", help="a line for each kernel")
    args = parser.parse_args()

    if args.command == "count":
        for path in args.files:
            with open(path, encoding="utf-8") as file:
                kernels, unfused = analyze(file.read())
            print(row(os.path.basename(path), kernels, unfused))
            if args.kernels:
                for k in kernels:
                    print("  " + kernel_line(k))
        return 0

    mdir = check.program(args.mdir)
    printer = args.printer or os.path.join(os.path.dirname(mdir), "mdir-opt")
    args.jobs = min(args.jobs, 8)
    kinds = args.only or ["examples", "tests", "bench", "extra", "ml"]
    args.only = [kind for kind in kinds if kind != "ml"]
    cases = check.corpus(args, None) if args.only else []
    if "ml" in kinds:
        cases += ml_cases(args.root)
    cases = [case for case in cases if not args.match or re.search(args.match, case.name)]
    # A model in stages runs on a GPU only: the case of the kind "ml" has
    # no CPU target.
    work = [(case, target, precision) for case in cases
            for target in args.targets.split(",") for precision in args.precisions.split(",")
            if not (case.name.startswith("ml/") and target == "cpu")]
    records, errors = [], Counter()
    with tempfile.TemporaryDirectory(prefix="duplicate-ops-") as scratch:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_case, mdir, printer, case, target, precision, args.cut,
                                   scratch, i, args.keep)
                       for i, (case, target, precision) in enumerate(work)]
            for (case, target, precision), future in zip(work, futures):
                result, error = future.result()
                name = f"{case.name} {target.upper()} {precision}"
                if error is not None:
                    errors[name] = error
                    continue
                kernels, unfused = result
                print(row(name, kernels, unfused), flush=True)
                if args.kernels:
                    for k in kernels:
                        print("  " + kernel_line(k), flush=True)
                records.append({"case": case.name, "target": target, "precision": precision,
                                "kernels": kernels,
                                "unfused": {f"{k}:{r}": n for (k, r), n in unfused.items()}})
    # The corpus as a whole, for each target and precision: each distinct
    # kernel once, whatever cases hold it, and the unfused loops of the
    # cases.
    for target in args.targets.split(","):
        for precision in args.precisions.split(","):
            chosen = [r for r in records if r["target"] == target and r["precision"] == precision]
            if not chosen:
                continue
            distinct = {}
            unfused = Counter()
            for r in chosen:
                for k in r["kernels"]:
                    distinct.setdefault(k["body"], k)
                for key, n in r["unfused"].items():
                    kind, reason = key.split(":")
                    unfused[(kind, reason)] += n
            print(row(f"corpus {target.upper()} {precision} ({len(chosen)} cases, distinct kernels)",
                      list(distinct.values()), unfused))
    print(f"{len(records)} cases measured, {len(errors)} stopped with an error")
    if args.errors:
        for name, error in errors.items():
            print(f"  {name}: {error}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump({"cut": args.cut, "cases": records,
                       "errors": dict(errors)}, file, indent=1)
    return 0


def kernel_line(k):
    cost = " ".join(f"{SHORT[c]}={k['costly'][c]}" for c in COSTLY if k["costly"].get(c))
    dup = " ".join(f"{d}={k['duplicates'][d]}" for d in KINDS if k["duplicates"].get(d))
    return (f"{k['kind']} {k['name']} x{k['copies']} depth={k['depth']} ops={k['ops']}: "
            f"{cost or 'none'}"
            f"{'; duplicates ' + dup if dup else ''}")


if __name__ == "__main__":
    sys.exit(main())
