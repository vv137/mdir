"""What misses the GPU entries of `mdir run` (D[compile-cache-default]).

  compile_cache_run.py INPUTS WORK GROUP MDIR

The dipeptide in water with PME, 20 deterministic steps on a GPU (the
program of gpu-module-compile-gpu.test). Each scenario is a run of its own,
in a directory of its own under WORK, on the cache directory it names, and
prints what its manifest records of the compile cache (PTX and cubins
compiled and read) with the entries that are new in the directory after it:
a scenario that misses every module stores as many new entries as the cold
run of the same program on an empty directory, one that hits stores none.
Modules that are the same within a run may read each other's entries, so
the counts compiled and read alone do not tell a miss.

GROUP is one of:

- llvm: cold, warm, an LLVM option of the command line
  (-nvptx-prec-divf32=0) in both orders, the scheduler of the command line
  (-pre-RA-sched=list-ilp), and MDIR_COMPILE_CACHE_SALT_OVERRIDE, which
  stands for another build; the energies of a hit are those of the cold run.
- target: `mdir emit --stage=lowered` for the device's architecture, then
  for another one (MDIR_GPU_ARCH), then as PTX (MDIR_GPU_BINARY=ptx).
- program: the switches of the pipeline and of the builder, fast_math,
  precision, the PME grid, neighbor_capacity, and MDIR_PIPELINE each store
  new entries; a start from another cell (D227) stores none.
- search: the skin search (D281) of a run with the groups: its candidates'
  host objects are stored cold and read again warm, and with an LLVM option
  of the host's code generator (-unroll-threshold=0) none is read.
"""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

inputs, work, group = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
mdir = sys.argv[4]
work.mkdir(parents=True, exist_ok=True)

BASE = """[input]
topology   = "dipeptide.prmtop"
coordinates = "dipeptide.inpcrd"

[output]
energy_interval = 10
manifest = "run.json"

[energy]
cutoff       = 9.0
pairlist_distance = 10.0
electrostatics = "PME"

[dynamics]
integrator    = "VELOCITY_VERLET"
time_step     = 0.001
steps         = 20

[ensemble]
ensemble    = "NVE"
temperature = 300.0

[constraints]
rigid_water = true

[boundary]
type = "PERIODIC"

[execution]
target = "gpu"
deterministic = true
"""


def entries(cache):
    gpu = cache / "gpu"
    return set(p.name for p in gpu.iterdir()) if gpu.is_dir() else set()


def host_entries(cache):
    host = cache / "host"
    return set(p.name for p in host.iterdir()) if host.is_dir() else set()


def count(names, suffix):
    return sum(n.endswith(suffix) for n in names)


energies = {}


def run(name, cache, args=(), env=None, control=BASE, emit=False,
            coordinates=None, same_as=None):
    directory = work / name
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    shutil.copy(inputs / "dipeptide.prmtop", directory)
    (directory / "dipeptide.inpcrd").write_text(
        coordinates or (inputs / "dipeptide.inpcrd").read_text())
    (directory / "run.toml").write_text(control)
    environment = dict(os.environ)
    for key in ("MDIR_COMPILE_CACHE", "MDIR_COMPILE_CACHE_SALT_OVERRIDE",
                "MDIR_COMPILE_CACHE_MAX_MB"):
        environment.pop(key, None)
    environment["MDIR_COMPILE_CACHE_DIR"] = str(work / cache)
    environment.update(env or {})
    before, hosts = entries(work / cache), host_entries(work / cache)
    command = ([mdir, "emit", "run.toml", "--stage=lowered"] if emit else
               [mdir, "run", "run.toml"]) + list(args)
    result = subprocess.run(command, cwd=directory, env=environment,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"{name}: FAILED\n{result.stdout}\n{result.stderr}")
        sys.exit(1)
    new = entries(work / cache) - before
    line = f"{name}:"
    if not emit:
        record = json.loads((directory / "run.json").read_text().splitlines()[0])
        cache_record = record["compile_cache"]
        line += (f" ptx compiled {cache_record['ptx_compiled']} read "
                 f"{cache_record['ptx_read']} cubins compiled "
                 f"{cache_record['cubin_compiled']} read "
                 f"{cache_record['cubin_read']};")
        energies[name] = [l for l in result.stdout.splitlines()
                          if l.startswith("INFO:")]
    line += f" new ptx {count(new, '.ptx')} cubin {count(new, '.cubin')}"
    if same_as:
        line += " energies " + ("same" if energies[name] == energies[same_as]
                                else "DIFFERENT")
    print(line, flush=True)
    return result, record if not emit else None


if group == "llvm":
    run("cold", "one")
    run("warm", "one", same_as="cold")
    run("divf32", "one", args=["-nvptx-prec-divf32=0"])
    run("divf32-again", "one", args=["-nvptx-prec-divf32=0"])
    run("plain-again", "one", same_as="cold")
    run("divf32-first", "two", args=["-nvptx-prec-divf32=0"])
    run("plain-after", "two", same_as="cold")
    run("scheduler", "one", args=["-pre-RA-sched=list-ilp"])
    run("other-build", "one",
        env={"MDIR_COMPILE_CACHE_SALT_OVERRIDE": "other"}, same_as="cold")
    run("other-build-again", "one",
        env={"MDIR_COMPILE_CACHE_SALT_OVERRIDE": "other"}, same_as="cold")
elif group == "target":
    run("device", "one", emit=True)
    pipeline = subprocess.run(
        [mdir, "emit", "run.toml", "--stage=pipeline"], cwd=work / "device",
        text=True, capture_output=True, check=True).stdout
    chip = re.search(r"cubin-chip=(sm_\d+)", pipeline)
    other = "sm_86" if chip and chip.group(1) == "sm_80" else "sm_80"
    run("other-chip", "one", emit=True, env={"MDIR_GPU_ARCH": other})
    run("as-ptx", "one", emit=True, env={"MDIR_GPU_BINARY": "ptx"})
    run("device-again", "one", emit=True)
elif group == "program":
    run("cold", "one")
    for name, env in [("pme-spread-global", {"MDIR_PME_SPREAD": "global"}),
                      ("pme-spread-tiles", {"MDIR_PME_SPREAD": "tiles"}),
                      ("tuple-sums-atomics", {"MDIR_TUPLE_SUMS": "atomics"}),
                      ("tuple-sums-blocks", {"MDIR_TUPLE_SUMS": "blocks"}),
                      ("pair-sums-blocks", {"MDIR_PAIR_SUMS": "blocks"}),
                      ("fused-prune-off", {"MDIR_FUSED_PRUNE": "off"}),
                      ("check-pruning", {"MDIR_CHECK_PRUNING": "1"}),
                      ("constraint-test", {"MDIR_CONSTRAINT_TEST": "separate"})]:
        run(name, "one", env=env)
    for name, edit in [
            ("fast-math", lambda t: t + "fast_math = true\n"),
            ("precision", lambda t: t + 'precision = "mixed"\n'),
            ("neighbor-capacity", lambda t: t + "neighbor_capacity = 600\n"),
            ("pme-grid", lambda t: t + "\n[pme]\ngrid = [36, 36, 36]\n")]:
        run(name, "one", control=edit(BASE))
    # The pipeline that the run takes, with one option changed.
    emitted = subprocess.run(
        [mdir, "emit", "run.toml", "--stage=pipeline"], cwd=work / "cold",
        text=True, capture_output=True, check=True).stdout.strip()
    changed = emitted.replace("explicit-fma=true", "explicit-fma=false", 1)
    assert changed != emitted
    run("mdir-pipeline", "one", env={"MDIR_PIPELINE": changed})
    run("same-pipeline", "one", env={"MDIR_PIPELINE": emitted})
    # Another cell at the start: the edges are arguments of the entry (D227).
    text = (inputs / "dipeptide.inpcrd").read_text().splitlines()
    box = [float(x) for x in text[-1].split()]
    text[-1] = "".join(f"{v * 1.0004:12.7f}" for v in box[:3]) + "".join(
        f"{v:12.7f}" for v in box[3:])
    run("other-cell", "one", coordinates="\n".join(text) + "\n")
elif group == "search":
    control = (BASE.replace("pairlist_distance = 10.0",
                            'pairlist_distance = "AUTO"\npruned_distance = "AUTO"')
               .replace("deterministic = true", 'neighbor_structure = "GROUPS"'))
    env = {"MDIR_SKIN_SEARCH": "always", "MDIR_SKIN_CACHE": "off"}
    for name, args in [("search-cold", []), ("search-warm", []),
                       ("search-unroll", ["-unroll-threshold=0"])]:
        before = host_entries(work / "search")
        _, record = run(name, "search", args=args, env=env, control=control)
        c = record["compile_cache"]
        print(f"{name}: host objects of the search compiled "
              f"{'some' if c['skin_search_host_compiled'] else 'none'}, read "
              f"{'some' if c['skin_search_host_read'] else 'none'}, new "
              f"{'some' if host_entries(work / 'search') - before else 'none'}",
              flush=True)
