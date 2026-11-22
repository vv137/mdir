"""Report static PTX instructions and physical registers for the audit controls."""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path


def counts(ops):
    ops = list(ops)
    return (sum("f64" in op.split(".") for op in ops),
            sum(op.startswith("ld.global.") for op in ops))


def scopes(ptx):
    # Split after conditional branches as well as at labels so that a branch
    # and its fallthrough can have different loop memberships.
    blocks = [[]]
    labels = {}
    branches = {}
    for line in ptx.split(".entry ", 1)[1].split("{", 1)[1].splitlines():
        line = line.strip()
        if line.endswith(":"):
            if blocks[-1]:
                blocks.append([])
            labels[line[:-1]] = len(blocks) - 1
        m = re.fullmatch(r"(@!?%\w+\s+)?([a-z][\w.]*)(?:\s+(.*?))?;", line)
        if not m:
            continue
        pred, op, args = m.groups()
        blocks[-1].append(op)
        if op.startswith("bra") or op == "ret":
            branches[len(blocks) - 1] = (args, bool(pred))
            blocks.append([])

    predecessors = [set() for _ in blocks]
    for i in range(len(blocks)):
        target, falls = branches.get(i, (None, True))
        if target:
            predecessors[labels[target]].add(i)
        if falls and i + 1 < len(blocks):
            predecessors[i + 1].add(i)
    yield "entry", counts(op for block in blocks for op in block)
    # A natural loop has a backedge whose target dominates its source.
    dom = [set(range(len(blocks))) for _ in blocks]
    dom[0] = {0}
    changed = True
    while changed:
        changed = False
        for i, pred in enumerate(predecessors[1:], 1):
            value = {i}
            if pred:
                value |= set.intersection(*(dom[p] for p in pred))
            if value != dom[i]:
                dom[i] = value
                changed = True
    for label, header in labels.items():
        members = {header}
        pending = [p for p in predecessors[header] if header in dom[p]]
        if not pending:
            continue
        while pending:
            p = pending.pop()
            if p not in members:
                members.add(p)
                pending.extend(predecessors[p])
        yield label, counts(op for p in members for op in blocks[p])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="contains {MATRIX,GROUPS}.final.mlir")
    parser.add_argument("--ptxas", default="ptxas")
    args = parser.parse_args()
    baseline = Path(__file__).with_name("registers.tsv").read_text().splitlines()[1:]
    print("control\tmodule\tscope\tf64\tglobal_loads\tregisters/thread\tdelta\treview")
    with tempfile.TemporaryDirectory() as tmp:
        for control in ("MATRIX", "GROUPS"):
            ir = (args.directory / (control + ".final.mlir")).read_text()
            modules = dict(re.findall(r'gpu.binary @(\w+)[^\n]*?assembly = "([^"]*)"', ir))
            for row in baseline:
                kind, module, regs = row.split("\t")
                if kind != control:
                    continue
                ptx = re.sub(r"\\([0-9A-Fa-f]{2})", lambda m: chr(int(m[1], 16)),
                             modules[module]).rstrip("\0")
                path = Path(tmp) / "kernel.ptx"
                path.write_text(ptx)
                result = subprocess.run([args.ptxas, "-arch=sm_89", "-O2", "-v", path,
                                         "-o", Path(tmp) / "kernel.cubin"],
                                        stderr=subprocess.PIPE, text=True)
                if result.returncode:
                    raise SystemExit(result.stderr)
                used = int(re.search(r"Used (\d+) registers", result.stderr)[1])
                delta = used - int(regs)
                for scope, (f64, loads) in scopes(ptx):
                    print(control, module, scope, f64, loads, used, delta,
                          "review" if delta > 4 else "-", sep="\t")


if __name__ == "__main__":
    main()
