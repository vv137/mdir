"""A process that forks after mdir.compile of a GPU program: the child
simulates on the GPU (D214).

  fork_after_compile.py ROOT

Lowering a GPU program asks NVML, not the CUDA driver, for the device's
architecture, so the parent has no CUDA state when it forks; a child of a
parent that had initialized the driver could not use CUDA. Prints the
pipeline's GPU options and one line from the child."""
import os
import sys

import mdir

root = sys.argv[1]
loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
system, state = loaded.make_system(), loaded.make_state()
system.cutoff, system.pairlist_distance = 0.8, 0.9
system.truncation = mdir.Truncation.None_
system.electrostatics = mdir.Electrostatics.PME
system.rigid_hydrogen_bonds = system.rigid_water = True
start = state.draw_velocities(system, 300.0, 271828)
integrator, ensemble, execution = mdir.Integrator(), mdir.Ensemble(), mdir.Execution()
integrator.timestep, ensemble.temperature, ensemble.seed = 0.0005, 300.0, 271828
execution.target = mdir.Target.GPU
program = mdir.compile(system, start, integrator, ensemble, execution, mdir.Schedule())
gpu = [p for p in program.pipeline.split(",") if p.startswith("mdir-gpu-lower-to-nvvm")]
print("parent:", gpu[0] if gpu else "no GPU pipeline", flush=True)

pid = os.fork()
if pid == 0:
    try:
        simulation = mdir.Simulation(program)
        simulation.run(10, energy=True)
        energy = simulation.state().energies["potential"]
        print(f"child: ran 10 steps, potential finite {energy == energy}", flush=True)
        os._exit(0)
    except BaseException as error:  # noqa: BLE001 - reported to the parent
        print(f"child: failed: {error}", flush=True)
        os._exit(1)
_, status = os.waitpid(pid, 0)
print("parent: child exited with", os.waitstatus_to_exitcode(status), flush=True)
