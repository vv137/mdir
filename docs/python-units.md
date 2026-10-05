# Unit quantities at the Python boundary (D200)

Issue #95. The Python model (D191) takes numbers in its public units: nm,
ps, kJ/mol, K, bar, amu, and e. With this decision every setter of a
quantity that has a unit also takes an `openmm.unit.Quantity` and converts
it to that unit at the boundary. Outputs do not change: `state()`, the
getters, and the plans return plain floats and NumPy arrays in the public
units, as D193 rules. The maintainer chose inputs only (#95).

```python
from openmm.unit import angstrom, femtosecond, kilocalorie_per_mole, atmosphere
system.cutoff = 8 * angstrom                                  # 0.8 nm
integrator.timestep = 2 * femtosecond                         # 0.002 ps
ensemble.pressure = 1 * atmosphere                            # 1.01325 bar
restraint = mdir.Restraint("@CA", 10 * kilocalorie_per_mole / angstrom**2)  # 4184 kJ/mol/nm^2
state.positions = pdb.positions                               # a list of Vec3 in nm
```

## Detection and conversion

OpenMM is not a dependency of MDIR. A value is taken as a quantity when it
has the attribute `value_in_unit_system` (duck typing), and only then is
`openmm.unit` imported. The quantity is converted by its own
`value_in_unit` to the unit that the setter declares (the table below).
The declared units are those of OpenMM's `md_unit_system` except for
pressure: `(1 * bar).value_in_unit_system(md_unit_system)` is
$10^{-25}$, not 1: that system has no unit of pressure, and the one it
derives from its base units (amu, nm, ps, and the mole) is not the bar of
MDIR's pressures. Converting to the declared unit, rather than to the unit system,
gives bar for pressure and the same value as `md_unit_system` everywhere
else.

A plain number keeps its meaning in the public units, as before. A
quantity of the wrong dimension (a time where a length is expected) raises
`mdir.InputError` that names the setter and the unit it expects, followed
by OpenMM's own message, for example
`System.cutoff: expected a quantity in nm; Unit "femtosecond" is not
compatible with Unit "nanometer".` A quantity seen when `openmm.unit`
cannot be imported raises `InputError` too.

## Setters and their units

| Setter | Unit | Shape |
|---|---|---|
| `System.cutoff`, `pairlist_distance`, `switch_distance`, `pme_spacing` | nm | scalar |
| `System.pme_alpha` | 1/nm | scalar |
| `System.restraint_reference` | nm | $(N, 3)$ |
| `Restraint.force_constant` (and the constructor's argument) | kJ/mol/nm² | scalar |
| `InitialState.positions` | nm | $(N, 3)$ |
| `InitialState.velocities` | nm/ps | $(N, 3)$ |
| `InitialState.draw_velocities(system, temperature, seed)`: `temperature` | K | scalar |
| `Cell.diagonal`, `Cell.tilt` | nm | $(3,)$ |
| `Integrator.timestep` | ps | scalar |
| `Integrator.minimize_step` | nm | scalar |
| `Ensemble.temperature` | K | scalar |
| `Ensemble.tau_t`, `Ensemble.tau_p` | ps | scalar |
| `Ensemble.pressure` | bar | scalar |
| `Ensemble.compressibility` | 1/bar | scalar |
| `Simulation.part_seconds` | s (wall-clock time) | scalar |

`Simulation.part_seconds` is the length of a part in wall-clock time
(D196), not simulated time, so a quantity converts to seconds there, not
to picoseconds.

Values with no declared unit take plain values only, and a quantity there
raises `InputError` or, from pybind11's conversion of a plain number,
`TypeError`: `System.pme_tolerance` (a relative error), counts and periods
in steps, seeds, flags, the constants of custom pair expressions
(`PairTerm.constants`), and the per-tuple
`TupleTerm.parameters`. Constants of an expression have the units that the
expression gives them, which MDIR does not know, so it cannot convert them.

## Arrays

An array setter strips the quantity first. A magnitude that is a NumPy
array (or another buffer or CPU DLPack object) then passes D193's checks
unchanged: shape, native float64, C order, finite values, and particle
count. A NumPy array of float32 in a quantity stays float32 after
`value_in_unit` and is refused as D193 refuses it. A magnitude that is a
sequence, such as the list of `Vec3` that OpenMM's `positions` holds or a
single `Vec3` for a cell, is first converted to a float64 array by
`numpy.array(value, dtype=float64)`, then checked the same way. Only the
quantity path converts a sequence; a plain list stays refused, as D193
rules.

A quantity is recognized before the buffer and DLPack probes of D193,
because `Quantity` forwards unknown attributes to its magnitude: before this
decision, a quantity wrapping a NumPy array passed the DLPack probe and its
magnitude was stored with its unit dropped, so that positions in Å were
taken as nm. Such a quantity is now converted where the setter declares a
unit and refused where it does not (`TupleTerm.parameters`).

## Validation

`test/Driver/python-units.test` runs `test/Driver/Inputs/python_units.py`
with OpenMM installed. Each setter is set once with a quantity in other
units and once with the plain value in the public unit, and the two stored
values are compared:

| Setter | Quantity | Stored | Plain value | Difference |
|---|---|---|---|---|
| `System.cutoff` | 8 Å | 0.8 | 0.8 | 0 |
| `System.pairlist_distance` | 9.5 Å | 0.9500000000000001 | 0.95 | 1 ulp |
| `System.switch_distance` | 7 Å | 0.7000000000000001 | 0.7 | 1 ulp |
| `System.pme_spacing` | 1 Å | 0.1 | 0.1 | 0 |
| `System.pme_alpha` | 3.4 1/Å | 34.0 | 34.0 | 0 |
| `Integrator.timestep` | 2 fs | 0.002 | 0.002 | 0 |
| `Integrator.minimize_step` | 0.1 Å | 0.010000000000000002 | 0.01 | 1 ulp |
| `Ensemble.temperature` | 300 K | 300.0 | 300.0 | 0 |
| `Ensemble.tau_t` | 1000 fs | 1.0 | 1.0 | 0 |
| `Ensemble.tau_p` | 5000 fs | 5.0 | 5.0 | 0 |
| `Ensemble.pressure` | 1 atm | 1.01325 | 1.01325 | 0 |
| `Ensemble.compressibility` | 4.5e-10 1/Pa | 4.4999999999999996e-05 | 4.5e-05 | 1 ulp |
| `Restraint.force_constant` | 10 kcal/mol/Å² | 4184.0 | 4184.0 | 0 |
| `Simulation.part_seconds` | 250 ms | 0.25 | 0.25 | 0 |

The largest difference is one unit in the last place, the rounding of
the conversion factor; the tolerance is one ulp. Arrays: the positions of
the dipeptide (1168 particles) in Å, as a list of `Vec3` and as a NumPy
array, the velocities in Å/fs, the restraint reference in Å, and a cell in
Å (`Vec3` and a list), against the same values in MD units: within 1 ulp
of each element. A model built from quantities (cutoff 8 Å, pair list 9 Å,
switch 5 Å, time step 1 fs, 1 atm, 500 fs, 310 K, a restraint of
10 kcal/mol/Å², the reference as `Vec3` in nm: values that convert
exactly) compiles to the same IR as the plain model, on CPU and GPU
targets in mixed and double, and the IR of another pressure differs. The
test also checks that a quantity advances the version of its input (a
compiled program becomes stale) and a refused one does not, that each
wrong dimension raises `InputError` naming the setter and the unit, that
a float32 NumPy quantity, a quantity on `TupleTerm.parameters`, and a
quantity on `System.pme_tolerance` are refused, that a plain list is
still refused, and that plain setters do not import `openmm`.
`test/Driver/python-units-gpu.test` repeats it with programs for a GPU.
`test/Driver/python-units-no-openmm.test` hides `openmm` from the
interpreter: MDIR imports, the plain setters work, and an object with
`value_in_unit_system` raises `InputError`. The tests that need OpenMM
require the lit feature `openmm`, present when the configured Python
imports `openmm.unit`.
