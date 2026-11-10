"""What of the program misses the host object (D397).

  compile_cache_program.py ROOT TARGET WORK [--child NAME OUT]

The dipeptide in water with PME, SHAKE, and SETTLE, outside the
deterministic mode, compiled by a process of its own a variant on one cache
directory, warm with the base program. Each variant prints whether its
lowered program (`Program.lowered_ir`, with the binaries of its kernels on a
GPU) differs from the base's and what its simulation took from the cache:
a variant whose lowered program differs misses the host object, and one
whose program is the same hits, whatever the setting that made it. The
variants: precision, fast_math, neighbor_capacity, the PME grid, the
switches of the pipeline and of the builder in the environment, and a
start from another cell (D227), whose edges are arguments of the entry.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

root, target, work = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])

VARIANTS = {
    "base": ({}, {}),
    "base-again": ({}, {}),
    "precision": ({"precision": "Double"}, {}),
    "fast-math": ({"fast_math": True}, {}),
    "neighbor-capacity": ({"neighbor_capacity": 600}, {}),
    "pme-grid": ({"pme_grid": [36, 36, 36]}, {}),
    "pme-spread": ({}, {"MDIR_PME_SPREAD": "global"}),
    "tuple-sums": ({}, {"MDIR_TUPLE_SUMS": "atomics"}),
    "pair-sums": ({}, {"MDIR_PAIR_SUMS": "blocks"}),
    "fused-prune": ({}, {"MDIR_FUSED_PRUNE": "off"}),
    "check-pruning": ({}, {"MDIR_CHECK_PRUNING": "1"}),
    "constraint-test": ({}, {"MDIR_CONSTRAINT_TEST": "separate"}),
    "other-cell": ({"scale": 1.0004}, {}),
}


def child(name, out):
    import mdir

    settings = VARIANTS[name][0]
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.rigid_hydrogen_bonds = system.rigid_water = True
    if "pme_grid" in settings:
        system.pme_grid = settings["pme_grid"]
    if "scale" in settings:
        cell = state.cell
        cell.diagonal = np.asarray(cell.diagonal) * settings["scale"]
        state.cell = cell
        state.positions = np.asarray(state.positions) * settings["scale"]
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.0005, 300.0, 271828
    execution.target = getattr(mdir.Target, target)
    execution.precision = getattr(mdir.Precision, settings.get("precision", "Mixed"))
    execution.deterministic = False
    execution.fast_math = settings.get("fast_math", False)
    execution.neighbor_capacity = settings.get("neighbor_capacity", 0)
    program = mdir.compile(system, state.draw_velocities(system, 300.0, 271828),
                           integrator, ensemble, execution, mdir.Schedule())
    lowered = hashlib.sha256(program.lowered_ir.encode()).hexdigest()
    stats = mdir.Simulation(program).compile_stats
    pathlib.Path(out).write_text(json.dumps({"lowered": lowered, "stats": stats}))


if len(sys.argv) > 4 and sys.argv[4] == "--child":
    child(sys.argv[5], sys.argv[6])
    sys.exit(0)

work.mkdir(parents=True, exist_ok=True)
base = None
agree = True
for name, (_, env) in VARIANTS.items():
    environment = dict(os.environ)
    for key in ("MDIR_COMPILE_CACHE", "MDIR_COMPILE_CACHE_MAX_MB",
                "MDIR_COMPILE_CACHE_SALT_OVERRIDE"):
        environment.pop(key, None)
    environment["MDIR_COMPILE_CACHE_DIR"] = str(work / "cache")
    environment.update(env)
    out = work / f"{name}.json"
    subprocess.run([sys.executable, __file__, root, target, str(work), "--child",
                    name, str(out)], env=environment, check=True)
    result = json.loads(out.read_text())
    base = base or result["lowered"]
    s = result["stats"]
    same = result["lowered"] == base
    print(f"{name}: lowered {'same' if same else 'differs'}"
          f", host compiled {s['host_compiled']} hits {s['cache_hits']}",
          flush=True)
    if name != "base":
        agree &= (s["host_compiled"], s["cache_hits"]) == ((0, 1) if same else (1, 0))
print("a variant missed exactly when its lowered program differs:",
      "yes" if agree else "NO")
