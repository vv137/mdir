"""The compile cache of host objects (D212) across processes.

  compile_cache.py ROOT TARGET PRECISION WORK [--damaged] [--misses] [--concurrent]

The dipeptide in water with PME, SHAKE, and SETTLE, deterministic, 20 steps
in two runs of one part each; a Simulation compiles one program. Each
scenario runs in a process of its own on the cache directory WORK/cache:

- off: no cache (MDIR_COMPILE_CACHE=off); the reference state; no
  directory is made.
- cold: the object is generated and stored.
- warm: a hit; nothing is generated.
- with --damaged: truncated, the entry is cut in half, and it is
  rejected, generated again, and stored; flipped, a byte of the stored
  object is flipped, likewise; rewarm, a hit again.
- with --misses: a changed pipeline option (not deterministic) and a
  changed precision miss, and a bound of 0 MiB leaves no entry.
- with --concurrent: four processes start at once on an empty directory;
  each generates or reads the object, and one entry remains, which the
  next process hits.
- with --gpu-damaged (a GPU): a byte of the data of one PTX entry, and of
  one cubin entry if there are any, is flipped; each is rejected,
  generated again, and stored (D214).

On a GPU each scenario prints a second line with what the serialization of
the GPU modules did: modules, PTX and cubins generated and read from the
cache, entries rejected and stored.

The state after the 20 steps of every deterministic scenario must equal
that of `off` bit for bit. Prints one line per scenario.
"""
import concurrent.futures as concurrent_futures
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

STEPS = 20
SEED = 271828


def child(root, target, precision, deterministic, out):
    import mdir

    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.rigid_hydrogen_bonds = system.rigid_water = True
    start = state.draw_velocities(system, 300.0, SEED)
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature, ensemble.seed = 0.0005, 300.0, SEED
    execution.target = getattr(mdir.Target, target)
    execution.precision = getattr(mdir.Precision, precision)
    execution.deterministic = deterministic
    simulation = mdir.Simulation(mdir.compile(system, start, integrator, ensemble,
                                              execution, mdir.Schedule()))
    # One part per run: where parts end is part of the arithmetic.
    simulation.part_seconds = 1e9
    simulation.run(STEPS // 2)
    simulation.run(STEPS - STEPS // 2, energy=True)
    result = simulation.state()
    np.savez(out, positions=np.asarray(result.positions),
             velocities=np.asarray(result.velocities))
    print(json.dumps(simulation.compile_stats))


def main():
    root, target, precision, work = sys.argv[1:5]
    damaged = "--damaged" in sys.argv[5:]
    misses = "--misses" in sys.argv[5:]
    concurrent = "--concurrent" in sys.argv[5:]
    gpu_damaged = "--gpu-damaged" in sys.argv[5:]
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "cache"
    host = cache / "host"

    def run(name, precision=precision, deterministic=True, off=False, env=None):
        environment = dict(os.environ)
        environment.pop("MDIR_COMPILE_CACHE", None)
        environment.pop("MDIR_COMPILE_CACHE_MAX_MB", None)
        environment["MDIR_COMPILE_CACHE_DIR"] = str(cache)
        if off:
            environment["MDIR_COMPILE_CACHE"] = "off"
        environment.update(env or {})
        out = work / f"{name}.npz"
        result = subprocess.run(
            [sys.executable, __file__, "--child", root, target, precision,
             str(int(deterministic)), str(out)],
            env=environment, check=True, stdout=subprocess.PIPE, text=True)
        return json.loads(result.stdout.splitlines()[-1]), out

    def same(a, b):
        a, b = np.load(a), np.load(b)
        return all(a[k].tobytes() == b[k].tobytes() for k in a.files)

    def report(name, stats, out=None, reference=None):
        line = (f"{name}: compiled {stats['host_compiled']} hits {stats['cache_hits']}"
                f" rejected {stats['cache_rejected']} stored {stats['cache_stored']}")
        if out is not None:
            line += " state " + ("same" if same(reference, out) else "DIFFERENT")
        print(line, flush=True)
        if target == "GPU":
            print(f"{name}: gpu modules {stats['gpu_modules']} ptx generated "
                  f"{stats['gpu_ptx_compiled']} hits {stats['gpu_ptx_hits']} cubins "
                  f"generated {stats['gpu_cubin_compiled']} hits {stats['gpu_cubin_hits']} "
                  f"rejected {stats['gpu_cache_rejected']} stored "
                  f"{stats['gpu_cache_stored']}", flush=True)

    def gpu_entries(kind):
        return sorted((cache / "gpu").glob("*." + kind)) if (cache / "gpu").exists() else []

    def entries():
        return sorted(host.glob("*.o")) if host.exists() else []

    stats, reference = run("off", off=True)
    report("off", stats)
    print("off: directory", "made" if cache.exists() else "absent")
    stats, out = run("cold")
    report("cold", stats, out, reference)
    stored = entries()
    print(f"cold: entries {len(stored)}")
    if target == "GPU":
        print(f"cold: gpu entries ptx {len(gpu_entries('ptx'))} cubin "
              f"{len(gpu_entries('cubin'))}")
    stats, out = run("warm")
    report("warm", stats, out, reference)
    if gpu_damaged:
        # One byte of the data of the first entry of each kind. The
        # lowering of mdir.compile in the same process may be the one that
        # rejects and rewrites an entry, so the entries themselves are
        # checked.
        flipped = {}
        for kind in ("ptx", "cubin"):
            for entry in gpu_entries(kind)[:1]:
                data = bytearray(entry.read_bytes())
                data[-10] ^= 0x01
                entry.write_bytes(bytes(data))
                flipped[entry] = bytes(data)
        stats, out = run("gpu-flipped")
        report("gpu-flipped", stats, out, reference)
        rewritten = sum(entry.read_bytes() != data for entry, data in flipped.items())
        print(f"gpu-flipped: rewritten {rewritten} of {len(flipped)}")
        stats, out = run("gpu-rewarm")
        report("gpu-rewarm", stats, out, reference)
    if damaged:
        # A truncated entry, then an object with one byte flipped.
        entry, = stored
        data = entry.read_bytes()
        entry.write_bytes(data[: len(data) // 2])
        stats, out = run("truncated")
        report("truncated", stats, out, reference)
        data = bytearray(entry.read_bytes())
        data[-100] ^= 0x40
        entry.write_bytes(bytes(data))
        stats, out = run("flipped")
        report("flipped", stats, out, reference)
        stats, out = run("rewarm")
        report("rewarm", stats, out, reference)
    if misses:
        stats, _ = run("nondeterministic", deterministic=False)
        report("nondeterministic", stats)
        other = "Double" if precision == "Mixed" else "Mixed"
        stats, _ = run("precision", precision=other)
        report("precision", stats)
        for entry in entries():
            entry.unlink()
        stats, _ = run("bounded", env={"MDIR_COMPILE_CACHE_MAX_MB": "0"})
        report("bounded", stats)
        print(f"bounded: entries {len(entries())}")
    if concurrent:
        for entry in entries():
            entry.unlink()
        with concurrent_futures.ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda i: run(f"concurrent{i}"), range(4)))
        compiled = sum(stats["host_compiled"] for stats, _ in results)
        hits = sum(stats["cache_hits"] for stats, _ in results)
        alike = all(same(reference, out) for _, out in results)
        print(f"concurrent: generated or hit {compiled + hits}, rejected "
              f"{sum(stats['cache_rejected'] for stats, _ in results)}, "
              f"entries {len(entries())}, state "
              + ("same" if alike else "DIFFERENT"))
        stats, out = run("after")
        report("after", stats, out, reference)

if __name__ == "__main__":
    if sys.argv[1] == "--child":
        root, target, precision, deterministic, out = sys.argv[2:7]
        child(root, target, precision, deterministic == "1", out)
    else:
        main()
