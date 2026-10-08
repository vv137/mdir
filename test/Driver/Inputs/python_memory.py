"""The memory of a persistent simulation across many calls of run (#110):
one activation of the entry runs every part (D215), and what
it allocates is freed when it ends, at an evaluation after the first run,
an update of tunables, or the end of the simulation, so the memory of the
device (GPU) and of the host (CPU) stays that of one activation."""
import os
import subprocess
import sys

import mdir

root = sys.argv[1]
target_name = sys.argv[2]
target = getattr(mdir.Target, target_name)
CALLS = 30
BOUND = 8.0  # MiB of growth from call 5 to call CALLS


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


def host_mib():
    """The resident memory of this process."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


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
measure = device_mib if target_name == "GPU" else host_mib
kind = "device" if target_name == "GPU" else "host"
# The first calls compile the program of the later segments and fill the
# pools; from call 5 on, the memory stays.
for call in range(1, CALLS + 1):
    assert simulation.run(5) == 5
    if call == 5:
        first = measure()
growth = measure() - first
print(f"{target_name}: {kind} memory from call 5 to call {CALLS}: {growth:+.1f} MiB "
      f"(bound {BOUND} MiB)")
assert growth <= BOUND, growth
# Each evaluation ends the activation and begins another.
for call in range(CALLS):
    assert simulation.run(5) == 5
    simulation.run(0, energy=True)
growth = measure() - first
print(f"{target_name}: {kind} memory after {CALLS} evaluations between runs: "
      f"{growth:+.1f} MiB (bound {BOUND} MiB)")
assert growth <= BOUND, growth
# Simulations that end free what their activations held. The first ones
# make what a second engine of the process keeps.
for call in range(1, CALLS + 1):
    other = mdir.Simulation(program)
    assert other.run(5) == 5
    del other
    if call == 5:
        ended = measure()
growth = measure() - ended
print(f"{target_name}: {kind} memory from simulation 5 to simulation {CALLS} that "
      f"ended: {growth:+.1f} MiB (bound {BOUND} MiB)")
assert growth <= BOUND, growth
# Writable borrows (D229): a commit ends the activation
# and begins another; an abandoned borrow keeps it.
for call in range(1, CALLS + 1):
    assert simulation.run(5) == 5
    with simulation.borrow() as borrow:
        capsules = [borrow.positions.__dlpack__(), borrow.velocities.__dlpack__()]
        del capsules
        if call % 2:
            assert borrow.commit() == ("positions", "velocities")
    if call == 5:
        borrowed = measure()
growth = measure() - borrowed
assert simulation.versions["positions"] == CALLS // 2
print(f"{target_name}: {kind} memory from borrow 5 to borrow {CALLS}, committed and "
      f"abandoned in turn: {growth:+.1f} MiB (bound {BOUND} MiB)")
assert growth <= BOUND, growth
print("memory across calls passed")
