"""Audit the canonical text dump after convert-md-alg-to-gpu."""
import re
import sys
from pathlib import Path

SSA = r"%[\w]+"
COORD = re.compile(rf"memref.load ({SSA})\[({SSA}), ({SSA})\] : memref<\?x3xf(?:32|64), 1>")
ENTRY = re.compile(rf"({SSA}) = memref.load .* : memref<\?x(?:\?x)?i32, 1>")


def regions(lines, marker):
    for i, line in enumerate(lines):
        if marker in line and line.endswith("{"):
            indent = len(line) - len(line.lstrip())
            end = i + 1
            while end < len(lines) and not lines[end].startswith(" " * indent + "}"):
                end += 1
            yield i, lines[i + 1:end]


def check(kind, body, prefix):
    def fail(message):
        sys.exit(f"{kind}: {message}")

    box_casts = 0
    for line in body:
        if kind == "GROUPS" and re.search(
            rf"= arith.truncf ({SSA}) fastmath<contract> : vector<3xf64> to vector<3xf32>$", line
        ) and len(line) - len(line.lstrip()) == len(body[0]) - len(body[0].lstrip()):
            box_casts += 1
            if box_casts == 1:
                continue
        if re.search(r"= (arith|math|vector|nvvm)\.", line) and "f64" in line:
            fail("unexpected f64 operation: " + line.strip())
    loads = COORD.findall("\n".join(body))
    if len(loads) != 3:
        fail(f"expected 3 source coordinate loads, found {len(loads)}")
    centers = COORD.findall("\n".join(prefix))
    buffer, index, _ = loads[0]
    if not centers or centers[-1][0] != buffer or any(
        (b, i) != (buffer, index) for b, i, _ in loads
    ):
        fail("source does not use the center's placed position buffer")

    definitions = dict(re.findall(rf"^\s*({SSA}) = (.*)$", "\n".join(body), re.M))
    cast = re.fullmatch(rf"arith.index_cast ({SSA}) : i32 to index", definitions.get(index, ""))
    entries = ENTRY.findall("\n".join(body))
    operand = cast[1] if cast else ""
    if kind == "search":
        valid = definitions.get(operand, "").startswith("arith.minsi ")
    elif kind == "prune":
        valid = len(entries) >= 2 and re.match(
            rf"arith.select {SSA}, {entries[1]}, ", definitions.get(operand, "")
        )
    else:
        slot = int(kind == "GROUPS" and any("arith.divui " in s for s in body[:10]))
        valid = len(entries) > slot and operand == entries[slot]
    if not valid:
        fail("coordinate index does not come directly from the candidate")


def audit(control, lines):
    found = []
    for start, launch in regions(lines, "gpu.launch "):
        header = lines[start]
        parent = next((s for s in reversed(lines[:start]) if "func.func " in s), "")
        search = "@mdrt_gpu_build_neighbors_matrix" in parent
        if not search and not re.search(r"module\(@mdir_run_(pair_for|refresh_neighbors)", header):
            continue
        for i, body in regions(launch, "scf.for "):
            # Only direct coordinate loads identify the owning loop. Audit its
            # entire body, including any nested interaction/rotation loops.
            indent = len(launch[i]) - len(launch[i].lstrip()) + 2
            if not any(COORD.search(s) and len(s) - len(s.lstrip()) == indent for s in body):
                continue
            if not re.search(r" step %c(?:16|32)(?:_| )", launch[i]):
                continue
            if search:
                kind = "search"
            elif " : i32 {" in launch[i]:
                kind = "prune"
            else:
                kind = control
            check(kind, body, launch[:i])
            found.append(kind)
    expected = {"search": 3, control: 4 if control == "MATRIX" else 7}
    if control == "GROUPS":
        expected["prune"] = 9
    counts = {k: found.count(k) for k in set(found)}
    if counts != expected:
        sys.exit(f"{control}: expected {expected}, found {counts}")
    print(control, counts)


if __name__ == "__main__":
    audit(sys.argv[1], Path(sys.argv[2]).read_text().splitlines())
