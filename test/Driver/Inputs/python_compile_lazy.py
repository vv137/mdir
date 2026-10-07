"""mdir.compile builds a program without lowering it; `lowered_ir` lowers on
its first read (D224, #151).

  python_compile_lazy.py ROOT TARGET WORK

With the compile cache in WORK/cache, a GPU program's lowering stores the
PTX of its modules there (D214), so an empty directory after `compile`
shows that nothing was lowered. Then `ir`, `pipeline`, and `plan` are read,
`lowered_ir` is read from four threads at once, and all four must be the
one text, equal to that of a later read. On the CPU, where a lowering
stores nothing, only the reads are checked.
"""
import concurrent.futures
import os
import pathlib
import sys

root, target, work = sys.argv[1:4]
cache = pathlib.Path(work) / "cache"
os.environ["MDIR_COMPILE_CACHE_DIR"] = str(cache)
os.environ.pop("MDIR_COMPILE_CACHE", None)

import mdir


def entries():
    return sorted(p.name for p in cache.rglob("*") if p.is_file()) if cache.exists() else []


loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
system, state = loaded.make_system(), loaded.make_state()
system.cutoff, system.pairlist_distance = 0.8, 0.9
system.truncation = mdir.Truncation.None_
system.electrostatics = mdir.Electrostatics.PME
execution = mdir.Execution()
execution.target = getattr(mdir.Target, target)
program = mdir.compile(system, state, mdir.Integrator(), mdir.Ensemble(), execution,
                       mdir.Schedule())
assert program.ir and program.pipeline and program.plan["target"] == execution.target
assert entries() == [], f"compile lowered the program: {entries()}"
with concurrent.futures.ThreadPoolExecutor(4) as pool:
    texts = list(pool.map(lambda _: program.lowered_ir, range(4)))
assert "llvm.func" in texts[0] and all(t == texts[0] for t in texts)
assert program.lowered_ir == texts[0]
if target == "GPU":
    assert entries(), "the lowering of lowered_ir stored no GPU module"
print("lazy lowering passed")
