"""The toolchain in the keys of the GPU entries (D397).

  compile_cache_toolchain.py ROOT WORK

The program of compile_cache.py (the dipeptide, deterministic, mixed
precision) on a GPU, compiled anew (a Program of its own) in one process,
with a toolkit of its own, WORK/toolkit, whose libdevice and ptxas are
replaced in place between compiles with files of the same size and time of
modification:

- cold: every module is generated; warm: every one is read.
- libdevice: a libdevice of other bytes but the same bitcode (a wrapper
  header and padding that the reader skips): every PTX misses, and the
  cubins of the PTX, which is the same, hit.
- ptxas: a ptxas script of other bytes that runs the same ptxas, so the same
  `--version`: the PTX hits and every cubin misses.
- the real ptxas, in another process, through a toolkit whose ptxas is a link
  to it: every cubin misses again.

The CUDA_ROOT of the environment is the toolkit whose libdevice and ptxas
the copies take. A line a compile: the modules, the PTX and cubins generated
and read, and the GPU entries new in the directory.
"""
import json
import os
import pathlib
import struct
import subprocess
import sys

root, work = sys.argv[1], pathlib.Path(sys.argv[2])
child = len(sys.argv) > 3 and sys.argv[3] == "--child"
real = pathlib.Path(os.environ["CUDA_ROOT"])
toolkit = work / "toolkit"
cache = work / "cache"
libdevice = toolkit / "nvvm" / "libdevice" / "libdevice.10.bc"
ptxas = toolkit / "bin" / "ptxas"


def wrapped(padding):
    """libdevice behind a bitcode wrapper header (magic, version, offset,
    size, CPU type), with four bytes after it that the reader skips."""
    code = (real / "nvvm" / "libdevice" / "libdevice.10.bc").read_bytes()
    return struct.pack("<5I", 0x0B17C0DE, 0, 20, len(code), 0) + code + padding


def script(tag):
    return f'#!/bin/sh\nexec "{real}/bin/ptxas" "$@"\n# {tag}\n'.encode()


def replace(path, data):
    """Writes `data` over the file in place, keeping its size and times."""
    status = path.stat()
    assert len(data) == status.st_size
    with open(path, "r+b") as f:
        f.write(data)
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))


def entries():
    gpu = cache / "gpu"
    return set(p.name for p in gpu.iterdir()) if gpu.is_dir() else set()


if not child:
    libdevice.parent.mkdir(parents=True, exist_ok=True)
    ptxas.parent.mkdir(parents=True, exist_ok=True)
    libdevice.write_bytes(wrapped(b"\0\0\0\0"))
    ptxas.write_bytes(script("a"))
    ptxas.chmod(0o755)
    linked = work / "linked"
    (linked / "nvvm" / "libdevice").mkdir(parents=True, exist_ok=True)
    (linked / "bin").mkdir(parents=True, exist_ok=True)
    (linked / "bin" / "ptxas").unlink(missing_ok=True)
    (linked / "bin" / "ptxas").symlink_to(real / "bin" / "ptxas")
    (linked / "nvvm" / "libdevice" / "libdevice.10.bc").write_bytes(
        wrapped(b"\1\1\1\1"))

os.environ["CUDA_ROOT"] = str(work / "linked" if child else toolkit)
os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"] = ""
os.environ.pop("MDIR_COMPILE_CACHE", None)
os.environ["MDIR_COMPILE_CACHE_DIR"] = str(cache)
import mdir  # noqa: E402


def compile_once(name):
    before = entries()
    loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
    system, state = loaded.make_system(), loaded.make_state()
    system.cutoff, system.pairlist_distance = 0.8, 0.9
    system.truncation = mdir.Truncation.None_
    system.electrostatics = mdir.Electrostatics.PME
    system.rigid_hydrogen_bonds = system.rigid_water = True
    integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
    integrator.timestep, ensemble.temperature = 0.0005, 300.0
    execution.target = mdir.Target.GPU
    execution.precision = mdir.Precision.Mixed
    execution.deterministic = True
    program = mdir.compile(system, state, integrator, ensemble, execution,
                           mdir.Schedule())
    s = mdir.Simulation(program).compile_stats
    new = entries() - before
    print(f"{name}: modules {s['gpu_modules']} ptx generated "
          f"{s['gpu_ptx_compiled']} read {s['gpu_ptx_hits']} cubins generated "
          f"{s['gpu_cubin_compiled']} read {s['gpu_cubin_hits']} new ptx "
          f"{sum(n.endswith('.ptx') for n in new)} cubin "
          f"{sum(n.endswith('.cubin') for n in new)}", flush=True)


if child:
    compile_once("linked-ptxas")
    sys.exit(0)
compile_once("cold")
print("cold: entries ptx {} cubin {}".format(
    sum(n.endswith(".ptx") for n in entries()),
    sum(n.endswith(".cubin") for n in entries())), flush=True)
compile_once("warm")
replace(libdevice, wrapped(b"\1\1\1\1"))
compile_once("libdevice")
replace(ptxas, script("b"))
compile_once("ptxas")
subprocess.run([sys.executable, __file__, root, str(work), "--child"], check=True)
