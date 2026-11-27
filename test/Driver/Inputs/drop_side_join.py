"""Writes the storage form IN to OUT without the first md_exec.join after
the first loop marked for the third stream (md_exec.lane = 2): a program
whose kick reads the tuples before their join, for the negative test of
check_term_streams_ir.py.

    drop_side_join.py IN OUT
"""
import sys

lines = open(sys.argv[1]).read().split("\n")
first = next(k for k, l in enumerate(lines) if "md_exec.lane = 2" in l)
join = next(k for k in range(first, len(lines)) if lines[k].strip() == "md_exec.join")
del lines[join]
open(sys.argv[2], "w").write("\n".join(lines))
