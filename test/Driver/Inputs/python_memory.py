"""The memory of a persistent simulation across many calls of run (#110):
one activation of the entry runs every part (D215), and what
it allocates is freed when it ends, at an evaluation after the first run,
an update of tunables, or the end of the simulation, so the memory of the
device (GPU) and of the host (CPU) stays that of one activation.

The memory of the host is read as two quantities that do not depend on what
else the machine runs (#228), instead of the resident memory, which read
+3 to +4 MiB alone and up to +8.8 MiB beside the other tests of a suite for
the same 25 simulations with nothing kept:

- the bytes in use of the allocator (`mallinfo2`: the blocks that were
  allocated and not freed, in every arena, and the blocks mapped one by
  one), bounded from step 5 to the last;
- the address space mapped outside the heap of the main arena, which holds
  what does not come from the allocator (the code and the data of a
  compiled engine, stacks, the arenas of the interpreter), bounded by the
  median of its growth per step: a thread or an arena that begins adds its
  mapping once, what every step keeps adds to every step.

Neither counts pages, so the pages that the allocator keeps or returns, and
those that the kernel takes back, change nothing. The thread cache of the
allocator is turned off for the process: a block freed into it still counts
as in use, and the caches of some 130 threads fill over the first hundred
simulations (+4 KiB per simulation from simulation 5 to 40 with 8 threads
that compile and +61 KiB from 5 to 25 with 128, against +0.4 KiB from 5 to
40 without the cache)."""
import ctypes
import gc
import os
import statistics
import subprocess
import sys

root = sys.argv[1]
target_name = sys.argv[2]
on_host = target_name != "GPU"
TCACHE_OFF = "glibc.malloc.tcache_count=0"
if on_host and TCACHE_OFF not in os.environ.get("GLIBC_TUNABLES", ""):
    # The allocator reads its tunables when the process begins.
    os.environ["GLIBC_TUNABLES"] = ":".join(
        filter(None, [os.environ.get("GLIBC_TUNABLES"), TCACHE_OFF]))
    os.execv(sys.executable, [sys.executable] + sys.argv)

import mdir

target = getattr(mdir.Target, target_name)
CALLS = 30
BOUND = 8.0  # MiB of growth of the device from call 5 to call CALLS
IN_USE_BOUND = 0.5  # MiB of growth in use on the host from step 5 to step CALLS
MAPPED_BOUND = 16.0  # KiB per step, median, of growth mapped on the host


def device_mib():
    """The memory of the device that this process uses, as nvidia-smi
    reports it."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader,nounits"], check=True,
                         stdout=subprocess.PIPE, text=True).stdout
    for line in out.splitlines():
        pid, used = (field.strip() for field in line.split(","))
        if int(pid) == os.getpid():
            return float(used)
    raise AssertionError("nvidia-smi does not list this process")


class Mallinfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks",
        "uordblks", "fordblks", "keepcost")]


if on_host:
    libc = ctypes.CDLL(None)
    libc.mallinfo2.restype = Mallinfo2


def host_mib():
    """The bytes in use of the allocator and the address space mapped
    outside the heap of the main arena, of this process, in MiB."""
    gc.collect()
    info = libc.mallinfo2()
    mapped = 0
    with open("/proc/self/maps") as f:
        for line in f:
            fields = line.split()
            if fields[-1] != "[heap]":
                low, high = (int(address, 16) for address in fields[0].split("-"))
                mapped += high - low
    return (info.uordblks + info.hblkhd) / 2**20, mapped / 2**20


def series(step, before=()):
    """Take CALLS steps and return the memory after each from the fifth on
    (on the device after the fifth and the last, a reading taking a
    process), following those of `before`. The first steps compile the
    program of the later segments, fill the pools, and make what a second
    engine of the process keeps; from step 5 on, the memory stays."""
    samples = list(before)
    for call in range(1, CALLS + 1):
        step(call)
        if call >= 5 and (on_host or call in (5, CALLS)):
            samples.append(host_mib() if on_host else device_mib())
    return samples


def check(what, samples):
    if not on_host:
        growth = samples[-1] - samples[0]
        print(f"{target_name}: device memory {what}: {growth:+.1f} MiB (bound {BOUND} MiB)")
        assert growth <= BOUND, growth
        return
    in_use = samples[-1][0] - samples[0][0]
    mapped = 1024 * statistics.median(
        after[1] - before[1] for before, after in zip(samples, samples[1:]))
    print(f"{target_name}: host memory {what}: in use {in_use:+.2f} MiB (bound "
          f"{IN_USE_BOUND} MiB), mapped {mapped:+.1f} KiB per step (bound "
          f"{MAPPED_BOUND} KiB; {samples[-1][1] - samples[0][1]:+.2f} MiB in all)")
    assert in_use <= IN_USE_BOUND, in_use
    assert mapped <= MAPPED_BOUND, mapped


loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
system, state = loaded.make_system(), loaded.make_state()
system.cutoff, system.pairlist_distance, system.switch_distance = 0.8, 0.9, 0.7
system.electrostatics = mdir.Electrostatics.PME
system.rigid_hydrogen_bonds = system.rigid_water = True
state = state.draw_velocities(system, 300.0, 1)
integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
integrator.timestep = 0.002
ensemble.kind = mdir.EnsembleKind.NVT
execution.target, execution.precision = target, mdir.Precision.Mixed
program = mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())
simulation = mdir.Simulation(program)


def run(call):
    assert simulation.run(5) == 5


calls = series(run)
check(f"from call 5 to call {CALLS}", calls)


def evaluate(call):
    """Each evaluation ends the activation and begins another."""
    assert simulation.run(5) == 5
    simulation.run(0, energy=True)


check(f"after {CALLS} evaluations between runs", series(evaluate, calls[:1]))


def end(call):
    """Simulations that end free what their activations held."""
    other = mdir.Simulation(program)
    assert other.run(5) == 5
    del other


check(f"from simulation 5 to simulation {CALLS} that ended", series(end))


def lend(call):
    """Writable borrows (D229): a commit ends the activation
    and begins another; an abandoned borrow keeps it."""
    assert simulation.run(5) == 5
    with simulation.borrow() as borrow:
        capsules = [borrow.positions.__dlpack__(), borrow.velocities.__dlpack__()]
        del capsules
        if call % 2:
            assert borrow.commit() == ("positions", "velocities")


borrowed = series(lend)
assert simulation.versions["positions"] == CALLS // 2
check(f"from borrow 5 to borrow {CALLS}, committed and abandoned in turn", borrowed)
print("memory across calls passed")
