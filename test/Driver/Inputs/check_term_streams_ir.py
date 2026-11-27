"""D[concurrent-terms]: where md-exec-assign-streams puts the force terms of
an ordinary step, read from the storage form after the pass.

    check_term_streams_ir.py STREAMS_MLIR VALUE

VALUE is the resolved term_streams. For TERMS, some block holds a fork of
the third stream (`md_exec.fork 2`), then a loop over pairs on the first
stream, then loops over tuples marked for the third (md_exec.lane = 2),
issued after the pairs, then a join and the loop over particles that
kicks. For RECIPROCAL, only reciprocal sums are marked. For SINGLE,
nothing is marked and there is no join or fork. Prints what it found.
"""
import re
import sys

text = open(sys.argv[1]).read()
value = sys.argv[2]
lines = text.splitlines()


def ops(lines):
    """Top-level ops of each block: (indent, kind, marked, lane, line)."""
    found = []
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)(?:%[\w#:]+ = )?(?:%[\w#:]+, %[\w#:]+ = )?md_exec\.(\w+)\b", line)
        if not m:
            continue
        found.append((len(m[1]), m[2], i))
    return found


def attributes(index, kind):
    """The attributes of the op that starts at line `index`: on its own line,
    or on the line that closes its region."""
    if kind in ("join",):
        return ""
    if "{" not in lines[index].split(":")[0] and not lines[index].rstrip().endswith("{"):
        return lines[index]
    depth = 0
    for j in range(index, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        if depth <= 0 and j > index:
            return lines[j]
    return ""


found = ops(lines)
marked = [(ind, kind, i, attributes(i, kind)) for ind, kind, i in found]
side = [(k, a) for _, k, _, a in marked if "md_exec.side" in a]
lanes = [(k, a) for k, a in side if "md_exec.lane = 2" in a]
joins = [m for m in marked if m[1] == "join"]
forks = [m for m in marked if m[1] == "fork"]
if value == "SINGLE":
    assert not side and not joins and not forks, (side[:2], len(joins), len(forks))
    print("SINGLE: no op is marked and there is no join")
elif value == "RECIPROCAL":
    assert side and all(k == "reciprocal" for k, _ in side), {k for k, _ in side}
    assert not lanes
    print(f"RECIPROCAL: {len(side)} reciprocal sums marked, no loop over tuples")
else:
    assert value == "TERMS"
    assert lanes and all(k == "tuple_for" for k, _ in lanes), {k for k, _ in lanes}
    windows = 0
    for n, (ind, kind, i, attr) in enumerate(marked):
        if kind != "fork" or "md_exec.fork 2" not in lines[i]:
            continue
        rest = [m for m in marked[n + 1:] if m[0] == ind]
        kinds = [m[1] for m in rest]
        if "join" not in kinds:
            continue
        join = kinds.index("join")
        window = rest[:join]
        pairs = [k for k, m in enumerate(window) if m[1] == "pair_for"]
        tuples = [k for k, m in enumerate(window)
                  if m[1] == "tuple_for" and "md_exec.lane = 2" in m[3]]
        kick = next((k for k, m in enumerate(rest) if m[1] == "particle_for" and k > join), None)
        if pairs and tuples and min(tuples) > max(pairs) and kick is not None:
            windows += 1
    assert windows, "no marked loop over tuples runs beside a loop over pairs"
    print(f"TERMS: {len(lanes)} loops over tuples on the third stream, "
          f"{windows} forked before a loop over pairs and issued after it")
