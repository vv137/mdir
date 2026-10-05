"""Unit quantities at the Python boundary (D[python-units]): each setter
with an OpenMM quantity in other units against the plain value in the
public units, arrays of Vec3 and NumPy magnitudes, wrong dimensions, and
values that have no declared unit."""
import sys

import numpy as np
import mdir

# Plain values do not import OpenMM.
plain_system = mdir.System()
plain_system.cutoff = 0.9
mdir.InitialState().positions = np.zeros((2, 3))
assert "openmm" not in sys.modules, "a plain setter imported openmm"
from openmm import Vec3
from openmm import unit as u

root = sys.argv[1]
target = getattr(mdir.Target, sys.argv[2])


def expect(error, call, *texts):
    try:
        call()
    except error as exc:
        for text in texts:
            assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected {error.__name__}")


def ulps(a, b):
    """The difference of two float64 values in units in the last place."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return int(np.max(np.abs(a.view(np.int64) - b.view(np.int64)), initial=0))


loaded = mdir.load_amber(root + "/dipeptide.prmtop", root + "/dipeptide.inpcrd")
reference_system, reference_state = loaded.make_system(), loaded.make_state()
count = reference_system.particle_count

# Scalars: (class, setter, quantity, plain value in the public unit).
SCALARS = [
    (mdir.System, "cutoff", 8 * u.angstrom, 0.8),
    (mdir.System, "pairlist_distance", 9.5 * u.angstrom, 0.95),
    (mdir.System, "switch_distance", 7 * u.angstrom, 0.7),
    (mdir.System, "pme_spacing", 1 * u.angstrom, 0.1),
    (mdir.System, "pme_alpha", 3.4 / u.angstrom, 34.0),
    (mdir.Integrator, "timestep", 2 * u.femtosecond, 0.002),
    (mdir.Integrator, "minimize_step", 0.1 * u.angstrom, 0.01),
    (mdir.Ensemble, "temperature", 300 * u.kelvin, 300.0),
    (mdir.Ensemble, "tau_t", 1000 * u.femtosecond, 1.0),
    (mdir.Ensemble, "tau_p", 5000 * u.femtosecond, 5.0),
    (mdir.Ensemble, "pressure", 1 * u.atmosphere, 1.01325),
    (mdir.Ensemble, "compressibility", 4.5e-10 / u.pascal, 4.5e-5),
    (mdir.Restraint, "force_constant", 10 * u.kilocalorie_per_mole / u.angstrom**2, 4184.0),
]
worst = 0
for cls, name, quantity, plain in SCALARS:
    obj = cls()
    setattr(obj, name, quantity)
    stored = getattr(obj, name)
    assert type(stored) is float, (name, type(stored))
    other = cls()
    setattr(other, name, plain)
    difference = ulps(stored, getattr(other, name))
    assert difference <= 1, (name, stored, plain)
    worst = max(worst, difference)
    print(f"{cls.__name__}.{name}: {quantity} -> {stored!r}; plain {getattr(other, name)!r}; "
          f"{difference} ulp")
    # A quantity of the wrong dimension names the setter and the unit.
    wrong = 1 * u.kilojoule_per_mole if name != "force_constant" else 1 * u.nanometer
    expect(mdir.InputError, lambda: setattr(obj, name, wrong), f"{cls.__name__}.{name}: expected a quantity in")
    assert getattr(obj, name) == stored

restraint = mdir.Restraint("@CA", 10 * u.kilocalorie_per_mole / u.angstrom**2)
assert ulps(restraint.force_constant, 4184.0) <= 1
expect(mdir.InputError, lambda: mdir.Restraint("@CA", 10 * u.kelvin), "Restraint.force_constant: expected a quantity in kJ/mol/nm^2")
print(f"scalar setters: {len(SCALARS)} within {worst} ulp of the plain values")

# Versions advance on the quantity path, and errors leave them.
system = loaded.make_system()
ensemble = mdir.Ensemble()
expect(mdir.InputError, lambda: setattr(system, "cutoff", 2 * u.femtosecond), "System.cutoff: expected a quantity in nm", "femtosecond")
expect(mdir.InputError, lambda: setattr(ensemble, "pressure", 1 * u.kilojoule), "Ensemble.pressure: expected a quantity in bar")
expect(mdir.InputError, lambda: setattr(system, "pme_tolerance", 1e-5 * u.nanometer), "System.pme_tolerance: takes plain values only")
expect(TypeError, lambda: setattr(system, "cutoff", "0.8"), "System.cutoff: expected a real number")
expect(mdir.InputError, lambda: setattr(system, "cutoff", np.ones(2) * u.nanometer), "System.cutoff: expected a real number in nm")

# Arrays: positions as a list of Vec3 in A and as a NumPy array in A.
positions = np.asarray(reference_state.positions)
in_angstrom = positions * 10.0
state = mdir.InitialState()
state.positions = [Vec3(*row) for row in in_angstrom] * u.angstrom
from_vec3 = np.asarray(state.positions)
state.positions = in_angstrom * u.angstrom
from_numpy = np.asarray(state.positions)
plain = in_angstrom / 10.0
print(f"positions: list of Vec3 within {ulps(from_vec3, plain)} ulp, "
      f"NumPy within {ulps(from_numpy, plain)} ulp of the plain values")
assert ulps(from_vec3, plain) <= 1 and ulps(from_numpy, plain) <= 1
velocities = np.random.default_rng(1).normal(size=(count, 3))  # nm/ps
state.velocities = (velocities * 1e-2) * (u.angstrom / u.femtosecond)
assert ulps(state.velocities, velocities * 1e-2 * 100.0) <= 1, "velocities"
system.restraint_reference = in_angstrom * u.angstrom
assert ulps(system.restraint_reference, plain) <= 1, "restraint_reference"
cell = mdir.Cell()
cell.diagonal = Vec3(30.0, 31.0, 32.0) * u.angstrom
cell.tilt = [0.0, 1.0, 2.0] * u.angstrom
assert ulps(cell.diagonal, [3.0, 3.1, 3.2]) <= 1 and ulps(cell.tilt, [0.0, 0.1, 0.2]) <= 1, "cell"
state.cell = cell
print("velocities, restraint reference, and cell within 1 ulp")
# Shape, dtype, and dimension checks still apply after the conversion.
expect(mdir.InputError, lambda: setattr(state, "positions", [Vec3(1, 2, 3)] * u.angstrom), "InitialState.positions: expected shape")
expect(mdir.InputError, lambda: setattr(state, "positions", in_angstrom.astype(np.float32) * u.angstrom), "expected native float64; found float32")
expect(mdir.InputError, lambda: setattr(state, "velocities", velocities * u.nanometer), "InitialState.velocities: expected a quantity in nm/ps")
expect(mdir.InputError, lambda: setattr(cell, "diagonal", Vec3(3, 3, 3) * u.picosecond), "Cell.diagonal: expected a quantity in nm")
expect(mdir.InputError, lambda: setattr(state, "positions", [list(r) for r in plain]), "lists are not accepted")
term = mdir.TupleTerm()
term.arity = 2
term.particles = np.array([[0, 1]], dtype=np.int64)
expect(mdir.InputError, lambda: setattr(term, "parameters", [("k", np.ones(1) * u.nanometer)]), "TupleTerm.parameters['k']: takes plain values only")
print("array refusals passed")

# The same model built from quantities and from plain values compiles to
# the same IR (values chosen to convert exactly: 8 A, 9 A, 5 A, 1 fs, 1 atm).
def build(quantities, pressure=1.01325, precision=mdir.Precision.Mixed):
    system, state = loaded.make_system(), loaded.make_state()
    integrator, ensemble = mdir.Integrator(), mdir.Ensemble()
    ensemble.kind = mdir.EnsembleKind.NPT
    q = (lambda value, unit: value * unit) if quantities else (lambda value, unit: value)
    system.cutoff = q(8.0, u.angstrom) if quantities else 0.8
    system.pairlist_distance = q(9.0, u.angstrom) if quantities else 0.9
    system.switch_distance = q(5.0, u.angstrom) if quantities else 0.5
    system.electrostatics = mdir.Electrostatics.Cutoff
    integrator.timestep = q(1.0, u.femtosecond) if quantities else 0.001
    ensemble.temperature = q(310.0, u.kelvin)
    ensemble.pressure = q(1.0, u.atmosphere) if quantities else pressure
    ensemble.tau_t = q(500.0, u.femtosecond) if quantities else 0.5
    system.restraints = [mdir.Restraint(
        "!:WAT & !@H*", q(10.0, u.kilocalorie_per_mole / u.angstrom**2) if quantities else 4184.0)]
    system.restraint_reference = ([Vec3(*row) for row in np.asarray(state.positions)] * u.nanometer
                                  if quantities else np.asarray(state.positions))
    inputs[:] = [system, state, integrator, ensemble]
    execution = mdir.Execution()
    execution.target, execution.precision = target, precision
    return mdir.compile(system, state, integrator, ensemble, execution, mdir.Schedule())


inputs = []
program = build(True)
assert program.ir == build(False).ir and program.ir != build(False, 1.0).ir
double = mdir.Precision.Double
assert build(True, precision=double).ir == build(False, precision=double).ir
print(f"{target.name}: model of quantities compiles to the IR of the plain model "
      "in mixed and double, not of another pressure")

# Wall-clock time converts to seconds, not to picoseconds.
simulation = mdir.Simulation(program)
# A quantity advances the version of its input; a refused one does not.
later = build(True)
expect(mdir.InputError, lambda: setattr(inputs[2], "timestep", 1 * u.kelvin))
assert not later.stale
inputs[2].timestep = 2 * u.femtosecond
assert later.stale
later = build(True)
inputs[1].positions = np.asarray(inputs[1].positions) * 10.0 * u.angstrom
assert later.stale
print("versions advance on the quantity path")
simulation.part_seconds = 250 * u.millisecond
assert ulps(simulation.part_seconds, 0.25) <= 1, simulation.part_seconds
expect(mdir.InputError, lambda: setattr(simulation, "part_seconds", 1 * u.nanometer), "Simulation.part_seconds: expected a quantity in s")
print(f"Simulation.part_seconds: 250 ms -> {simulation.part_seconds!r}")
print("unit quantities passed")
