"""Reads the policy of the md_alg ops (effective.md_alg_policy,
D408) from the manifests given: for the first, the number of its
entries and of its appends and grid plans; for each later one, each entry
in which it differs from the first, by op and function, with the fields
that differ.

Usage: md_alg_policy.py MANIFEST...
"""
import json
import os
import sys


def policy(path):
    with open(path) as f:
        first = json.loads(f.readline())
    return first["effective"]["md_alg_policy"]


base = policy(sys.argv[1])
name = os.path.basename(sys.argv[1])
count = lambda op: sum(1 for e in base if e["op"] == op)
print(f"{name}: {len(base)} entries, append {count('append')}, "
      f"grid_plan {count('grid_plan')}")
for path in sys.argv[2:]:
    other = policy(path)
    assert len(other) == len(base), (len(other), len(base))
    differ = [(a, b) for a, b in zip(base, other) if a != b]
    print(f"{os.path.basename(path)}: {len(differ)} entries differ")
    for a, b in differ:
        fields = ", ".join(f"{k} {a.get(k)} -> {b.get(k)}"
                           for k in sorted(set(a) | set(b))
                           if a.get(k) != b.get(k))
        print(f"  {a['op']} in @{a['function']}: {fields}")
