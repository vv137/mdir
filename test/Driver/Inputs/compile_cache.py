"""The compile cache of host objects (D212) across processes.

  compile_cache.py ROOT TARGET PRECISION WORK [--damaged] [--misses] [--concurrent]
                   [--bypass] [--clear] [--clear-concurrent]

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
- with --bypass (D217): before the cold run,
  `mdir.compile(..., cache=False)` makes no directory; after the warm run,
  it generates every object (no hit) and leaves the entries and their
  times as they were, so it read none; `Simulation(program, cache=False)`
  after a compile that uses the cache bypasses it likewise.
- with --clear: `mdir.clear_compile_cache()` removes the entries of the
  warm cache and leaves a file of another format and a writer's temporary
  file; the next process misses and stores, and the one after hits;
  `directory=` names another directory, and without a directory it
  removes nothing.
- with --clear-concurrent: a process clears the directory over and over
  while another compiles four simulations, each storing its entries; then
  every entry that remains is well formed, and the next process reads
  them without a rejection.

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


def child(root, target, precision, deterministic, out, mode=""):
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
    repeat = int(mode.split("=")[1]) if mode.startswith("repeat=") else 1
    stored = 0
    for _ in range(repeat):
        program = mdir.compile(system, start, integrator, ensemble, execution,
                               mdir.Schedule(), cache=mode != "bypass-compile")
        simulation = (mdir.Simulation(program, cache=False)
                      if mode == "bypass-simulation" else mdir.Simulation(program))
        stored += simulation.compile_stats["cache_stored"]
    # One part per run: where parts end is part of the arithmetic.
    simulation.part_seconds = 1e9
    simulation.run(STEPS // 2)
    simulation.run(STEPS - STEPS // 2, energy=True)
    result = simulation.state()
    np.savez(out, positions=np.asarray(result.positions),
             velocities=np.asarray(result.velocities))
    stats = simulation.compile_stats
    stats["stored_over_repeats"] = stored
    print(json.dumps(stats))


def clearer(directory, until):
    """Clears the directory over and over until the file `until` exists."""
    import mdir

    clears = removed = 0
    # The entries that the writer stored after the last clear remain.
    while not os.path.exists(until):
        cleared = mdir.clear_compile_cache(directory)
        clears += 1
        removed += cleared["host_entries"] + cleared["gpu_entries"]
    print(json.dumps({"clears": clears, "removed": removed}))


def well_formed(path):
    """The layout of an entry: tag, key, length, time, hash, data."""
    data = path.read_bytes()
    if data[:8] != b"MDIROBJ1":
        return False
    key = int.from_bytes(data[8:16], "little")
    at = 16 + key
    length = int.from_bytes(data[at:at + 8], "little")
    return len(data) == at + 16 + 64 + length


def main():
    root, target, precision, work = sys.argv[1:5]
    damaged = "--damaged" in sys.argv[5:]
    misses = "--misses" in sys.argv[5:]
    concurrent = "--concurrent" in sys.argv[5:]
    gpu_damaged = "--gpu-damaged" in sys.argv[5:]
    bypass = "--bypass" in sys.argv[5:]
    clear = "--clear" in sys.argv[5:]
    clear_concurrent = "--clear-concurrent" in sys.argv[5:]
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "cache"
    host = cache / "host"

    def run(name, precision=precision, deterministic=True, off=False, env=None,
            mode=""):
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
             str(int(deterministic)), str(out), mode],
            env=environment, check=True, stdout=subprocess.PIPE, text=True)
        return json.loads(result.stdout.splitlines()[-1]), out

    def same(a, b):
        a, b = np.load(a), np.load(b)
        return all(a[k].tobytes() == b[k].tobytes() for k in a.files)

    def report(name, stats, out=None, reference=None, bypassed=False):
        line = (f"{name}: compiled {stats['host_compiled']} hits {stats['cache_hits']}"
                f" rejected {stats['cache_rejected']} stored {stats['cache_stored']}")
        if bypassed:
            line += f" bypassed {stats['cache_bypassed']}"
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

    def all_entries():
        return entries() + gpu_entries("ptx") + gpu_entries("cubin")

    def snapshot(part=""):
        """Every file of the cache under `part` with its size and times."""
        if not cache.exists():
            return {}
        return {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in cache.rglob("*") if p.is_file() and part in str(p)}

    def clear_in_child(directory=None, env=None):
        environment = dict(os.environ)
        environment["MDIR_COMPILE_CACHE_DIR"] = str(cache)
        environment.update(env or {})
        code = ("import json, sys, mdir; d = sys.argv[1] or None; "
                "r = mdir.clear_compile_cache(d) if d else mdir.clear_compile_cache(); "
                "r['directory'] = r['directory'] is not None; print(json.dumps(r))")
        result = subprocess.run([sys.executable, "-c", code, directory or ""],
                                env=environment, check=True,
                                stdout=subprocess.PIPE, text=True)
        return json.loads(result.stdout.splitlines()[-1])

    stats, reference = run("off", off=True)
    report("off", stats)
    print("off: directory", "made" if cache.exists() else "absent")
    if bypass:
        stats, out = run("bypass-empty", mode="bypass-compile")
        report("bypass-empty", stats, out, reference, bypassed=True)
        print("bypass-empty: directory", "made" if cache.exists() else "absent")
    stats, out = run("cold")
    report("cold", stats, out, reference)
    stored = entries()
    print(f"cold: entries {len(stored)}")
    if target == "GPU":
        print(f"cold: gpu entries ptx {len(gpu_entries('ptx'))} cubin "
              f"{len(gpu_entries('cubin'))}")
    stats, out = run("warm")
    report("warm", stats, out, reference)
    if bypass:
        # Times long past, so that a read, which touches its entry, shows.
        for p in cache.rglob("*"):
            if p.is_file():
                os.utime(p, ns=(10**18, 10**18))
        before = snapshot()
        stats, out = run("bypass", mode="bypass-compile")
        report("bypass", stats, out, reference, bypassed=True)
        print("bypass: entries", "unchanged" if snapshot() == before else "CHANGED")
        # mdir.compile uses the cache here and may read GPU entries; the
        # simulation reads no host object.
        before = snapshot("/host/")
        stats, out = run("bypass-simulation", mode="bypass-simulation")
        report("bypass-simulation", stats, out, reference, bypassed=True)
        print("bypass-simulation: host entries",
              "unchanged" if snapshot("/host/") == before else "CHANGED")
    if clear:
        kinds = len(entries()), len(gpu_entries("ptx")) + len(gpu_entries("cubin"))
        foreign = host / "foreign.o"
        foreign.write_bytes(b"OTHERFMT" + bytes(64))
        writing = host / "entry.o.tmp-writer"
        writing.write_bytes(b"MDIROBJ1 partial")
        cleared = clear_in_child()
        print(f"clear: directory {cleared['directory']} host {cleared['host_entries']}"
              f" of {kinds[0]} gpu {cleared['gpu_entries']} of {kinds[1]} bytes "
              + ("positive" if cleared["bytes"] > 0 else "0"))
        print(f"clear: entries {sum(p != foreign for p in all_entries())}"
              f" foreign {'kept' if foreign.exists() else 'REMOVED'}"
              f" temporary {'kept' if writing.exists() else 'REMOVED'}")
        foreign.unlink()
        writing.unlink()
        stats, out = run("after-clear")
        report("after-clear", stats, out, reference)
        stats, out = run("rewarm-after-clear")
        report("rewarm-after-clear", stats, out, reference)
        count = len(all_entries())
        cleared = clear_in_child(str(work / "other"))
        print(f"clear-other: host {cleared['host_entries']} gpu {cleared['gpu_entries']}"
              f", entries here " + ("kept" if len(all_entries()) == count else "REMOVED"))
        cleared = clear_in_child(env={"MDIR_COMPILE_CACHE_DIR": ""})
        print(f"clear-none: directory {cleared['directory']} host "
              f"{cleared['host_entries']} gpu {cleared['gpu_entries']}, entries here "
              + ("kept" if len(all_entries()) == count else "REMOVED"))
        cleared = clear_in_child(env={"MDIR_COMPILE_CACHE": "off"})
        print(f"clear-off: host {cleared['host_entries']} entries {len(all_entries())}")
    if clear_concurrent:
        until = work / "writer-done"
        until.unlink(missing_ok=True)
        environment = dict(os.environ)
        environment["MDIR_COMPILE_CACHE_DIR"] = str(cache)
        clearing = subprocess.Popen(
            [sys.executable, __file__, "--clearer", str(cache), str(until)],
            env=environment, stdout=subprocess.PIPE, text=True)
        try:
            stats, out = run("writer", mode="repeat=4")
        finally:
            until.touch()
            cleared = json.loads(clearing.communicate()[0].splitlines()[-1])
        files = all_entries()
        print("clear-concurrent: stored over 4 compiles "
              + ("more than 1" if stats["stored_over_repeats"] > 1 else
                 str(stats["stored_over_repeats"]))
              + ", clears " + ("many" if cleared["clears"] > 1 else "ONE")
              + ", removed " + ("some" if cleared["removed"] > 0 else "NONE")
              + f", malformed {sum(not well_formed(p) for p in files)}"
              + ", state " + ("same" if same(reference, out) else "DIFFERENT"),
              flush=True)
        print(f"clear-concurrent: stored {stats['stored_over_repeats']} clears "
              f"{cleared['clears']} removed {cleared['removed']} entries left "
              f"{len(files)}", file=sys.stderr)
        stats, out = run("after-concurrent-clear")
        print(f"after-concurrent-clear: rejected {stats['cache_rejected']} "
              f"gpu rejected {stats['gpu_cache_rejected']} state "
              + ("same" if same(reference, out) else "DIFFERENT"))
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
        root, target, precision, deterministic, out, mode = sys.argv[2:8]
        child(root, target, precision, deterministic == "1", out, mode)
    elif sys.argv[1] == "--clearer":
        clearer(sys.argv[2], sys.argv[3])
    else:
        main()
