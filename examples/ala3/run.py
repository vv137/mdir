# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy>=1.23"]
# ///
"""The four stages of examples/ala3 from Python.

The stages are those of 1-min.toml to 4-md.toml: steepest descent with the
heavy atoms of the peptide restrained, 50 ps NVT from velocities drawn at
300 K, 100 ps NPT with weaker restraints, and 1 ns of production at
constant pressure with an energy file, a DCD trajectory, and the density
reported from Python. Each stage begins from the state of the one before.

    PYTHONPATH=<build>/python uv run examples/ala3/run.py [options]

The `mdir` module is the extension built with -DMDIR_ENABLE_PYTHON=ON; uv
provides Python 3.12 and NumPy from the header above. Options:

    --out DIR           where the outputs go (ala3-python)
    --cpu               run on the host instead of a GPU
    --threads N         threads on the host (1)
    --precision P       Mixed (the default) or Double
    --steps-scale X     multiply every stage's steps and intervals by X,
                        e.g. 0.01 for a quick check
    --seed N            the seed of the velocities and the baths (314159,
                        that of [dynamics] seed)
    --deterministic     the deterministic mode ([execution] deterministic)

Units: the Python model takes nm, ps, kJ/mol, K, and bar; the control files
give Å, kcal/mol, and atm.
"""
import argparse
import pathlib
import time

import numpy as np

import mdir

HERE = pathlib.Path(__file__).resolve().parent
KCAL_A2 = 4.184 / (0.1 * 0.1)  # kJ/mol/nm^2 per kcal/mol/Å^2
ATM = 1.01325                  # bar
AMU_NM3 = 1.66053906660        # 1000 g/cm^3 per amu/nm^3
HEAVY = "!:WAT & !@H*"         # the heavy atoms of the peptide
COUPLING = 10                  # steps between the actions of the baths

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--out", default="ala3-python", type=pathlib.Path)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--threads", default=1, type=int)
parser.add_argument("--precision", default="Mixed", choices=["Mixed", "Double"])
parser.add_argument("--steps-scale", default=1.0, type=float)
parser.add_argument("--seed", default=314159, type=int)
parser.add_argument("--deterministic", action="store_true")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)


def scaled(steps):
    """`steps` times --steps-scale, a multiple of the coupling period."""
    return max(COUPLING, round(steps * args.steps_scale / COUPLING) * COUPLING)


def total_mass(prmtop):
    """The sum of the %FLAG MASS section of an Amber topology, in amu."""
    lines, values, reading = prmtop.read_text().splitlines(), [], False
    for line in lines:
        if line.startswith("%FLAG"):
            reading = line.split()[1] == "MASS"
        elif reading and not line.startswith("%FORMAT"):
            values += [float(x) for x in line.split()]
    return sum(values)


def volume(state):
    return abs(np.linalg.det(state.cell.vectors))  # nm^3


def density_of(state):
    return mass / volume(state) * AMU_NM3 / 1000  # g/cm^3


# The system: ff19SB tri-alanine in OPC water, as in the control files.
prmtop, inpcrd = HERE / "ala3.prmtop", HERE / "ala3.inpcrd"
loaded = mdir.load_amber(str(prmtop), str(inpcrd))
system, initial = loaded.make_system(), loaded.make_state()
system.cutoff = 0.9                      # cutoff = 9.0 Å
system.pairlist_distance = 1.0           # pairlist_distance = 10.0 Å
system.truncation = mdir.Truncation.None_  # no switch_distance: none
system.electrostatics = mdir.Electrostatics.PME
system.coulomb_modifier = mdir.CoulombModifier.PotentialShift
system.rigid_hydrogen_bonds = True       # hydrogen_bonds = true
system.rigid_water = True                # rigid_water = true
# The reference of the restraints is the coordinates of ala3.inpcrd in
# every stage, as in the control files.
system.restraint_reference = initial.positions
mass = total_mass(prmtop)

execution = mdir.Execution()
execution.target = mdir.Target.CPU if args.cpu else mdir.Target.GPU
execution.precision = getattr(mdir.Precision, args.precision)
execution.threads = args.threads
execution.deterministic = args.deterministic


def restrain(kcal):
    system.restraints = [mdir.Restraint(HEAVY, kcal * KCAL_A2)] if kcal else []


def simulate(name, *model):
    """A simulation of the program of `model`, with what compiling it took.
    With MDIR_COMPILE_CACHE_DIR set, a stage whose host object is in the
    compile cache skips its generation (D212)."""
    began = time.perf_counter()
    program = mdir.compile(*model)
    lowered = time.perf_counter()
    simulation = mdir.Simulation(program)
    s = simulation.compile_stats
    source = "from the cache" if s["cache_hits"] else "generated"
    print(f"{name}: compiled in {time.perf_counter() - began:.1f} s; mdir.compile "
          f"{lowered - began:.1f} s, Simulation: MLIR {s['pipeline_seconds']:.1f} s, "
          f"JIT {s['engine_seconds']:.1f} s, host object {source}")
    return simulation


def dynamics(name, start, kind, steps, energy_period, energy_file):
    """A simulation of `steps` steps of 2 fs from `start` at 300 K."""
    integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
    integrator.method = mdir.IntegratorMethod.VelocityVerlet
    integrator.timestep = 0.002
    ensemble.kind = kind
    ensemble.temperature = 300.0
    ensemble.tau_t = 0.5                 # V-RESCALE, time_constant = 0.5
    ensemble.tau_p = 2.0                 # C-RESCALE, time_constant = 2.0
    ensemble.pressure = 1.0 * ATM
    ensemble.coupling_period = COUPLING  # interval = 10
    ensemble.com_period = COUPLING       # the motion of the center of mass, with it
    ensemble.seed = args.seed
    schedule = mdir.Schedule()
    schedule.steps, schedule.energy_period = steps, energy_period
    simulation = simulate(name, system, start, integrator, ensemble, execution,
                          schedule)
    simulation.reporters.append(mdir.EnergyReporter(str(args.out / energy_file),
                                                    energy_period))
    return simulation


def summary(name, simulation, seconds):
    state = simulation.state()
    e = state.energies
    print(f"{name}: {simulation.step} steps in {seconds:.1f} s; potential "
          f"{e['potential'] / 4.184:.2f} kcal/mol, temperature {e['temperature']:.1f} K, "
          f"density {density_of(state):.4f} g/cm^3"
          if e else f"{name}: {simulation.step} steps in {seconds:.1f} s")
    return state


# 1. Minimization: 2000 steps of steepest descent, restraints at 10
#    kcal/mol/Å² (1-min.toml).
restrain(10.0)
integrator, schedule = mdir.Integrator(), mdir.Schedule()
integrator.minimize = True           # STEEPEST_DESCENT
schedule.steps = scaled(2000)
simulation = simulate("1-min", system, initial, integrator, mdir.Ensemble(),
                      execution, schedule)
began = time.perf_counter()
simulation.minimize()
state = simulation.state()
row = state.minimization
print(f"1-min: {simulation.step} steps in {time.perf_counter() - began:.1f} s; energy "
      f"{row['energy'] / 4.184:.4f} kcal/mol, RMS force {row['rms_force'] / 41.84:.4f} "
      f"kcal/mol/Å, largest {row['max_force'] / 41.84:.4f} kcal/mol/Å")
np.save(args.out / "min-positions.npy", state.positions)

# 2. NVT: velocities drawn at 300 K, 50 ps with the restraints at 10
#    kcal/mol/Å² (2-nvt.toml).
restrain(10.0)
start = mdir.InitialState.from_state(state, velocities=False).draw_velocities(
    system, 300.0, args.seed)
simulation = dynamics("2-nvt", start, mdir.EnsembleKind.NVT, scaled(25000), scaled(2500), "nvt.dat")
began = time.perf_counter()
simulation.run(scaled(25000), energy=True)
state = summary("2-nvt", simulation, time.perf_counter() - began)
simulation.close_reporters()

# 3. NPT: 100 ps at 1 atm, restraints at 1 kcal/mol/Å² (3-npt.toml).
restrain(1.0)
simulation = dynamics("3-npt", mdir.InitialState.from_state(state), mdir.EnsembleKind.NPT, scaled(50000), scaled(5000),
                      "npt.dat")
began = time.perf_counter()
simulation.run(scaled(50000), energy=True)
state = summary("3-npt", simulation, time.perf_counter() - began)
simulation.close_reporters()

# 4. Production: 1 ns at 1 atm without restraints, energies every 10 ps, a
#    frame every 1 ps, and the density from Python every 10 ps
#    (4-md.toml).
restrain(0.0)
period = scaled(5000)
simulation = dynamics("4-md", mdir.InitialState.from_state(state), mdir.EnsembleKind.NPT, scaled(500000), period, "md.dat")
simulation.reporters.append(mdir.TrajectoryReporter(str(args.out / "md.dcd"), scaled(500)))
densities = []


def density(sim, s):
    densities.append(density_of(s))
    print(f"  step {s.step}: density {densities[-1]:.4f} g/cm^3")


simulation.reporters.append(mdir.CallbackReporter(density, period))
began = time.perf_counter()
simulation.run(scaled(500000), energy=True)
state = summary("4-md", simulation, time.perf_counter() - began)
simulation.close_reporters()
print(f"4-md: mean density {np.mean(densities):.4f} g/cm^3 over {len(densities)} reports")
