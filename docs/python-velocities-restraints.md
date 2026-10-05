# Drawn velocities and positional restraints in the Python model (D[python-velocities-restraints])

Issue #92 is the small model item that the four-stage example (#82) needs
after the persistent simulations of D196 (#85): initial velocities drawn at
a temperature, and positional restraints for the equilibration stages. The
maintainer ruled on PR #86 for option A of both: a method of the initial
state that reuses the CLI's draw, and a typed list on the system that maps
onto `[[restraints]]` (D74, D124). The maintainer accepted the design of
PR #94 on 2026-10-06: the system as an argument of the draw, separate seeds
for the velocities and the thermostat, constants in kJ/mol/nm², a Python
warning for a restraint that selects nothing, and references in the frame
of the compiled state's cell.

## Drawn velocities

```python
loaded = mdir.load_amber("system.prmtop", "system.inpcrd")
system, state = loaded.make_system(), loaded.make_state()
system.rigid_hydrogen_bonds = system.rigid_water = True
drawn = state.draw_velocities(system, 300.0, 314159)   # a new InitialState
```

`InitialState.draw_velocities(system, temperature, seed)` returns a new
`InitialState` with the positions and the cell of `state` and velocities
of shape `(N, 3)` float64 in nm/ps; `state` itself does not change. The
system is an argument because the draw needs what only the system has: the
masses, the constraints that `rigid_hydrogen_bonds`, `rigid_water`, and
`water_residues` select, and the virtual sites.

The draw is the one that `mdir run` makes when its coordinates give no
velocities (`assignVelocities`, D191 shares it): each component from a
normal distribution of variance $k_B T / m_i$, from the generator seeded by
`seed`; no velocity for a particle without mass; the components along the
constraints of SETTLE and SHAKE removed; the motion of the center of mass
removed; and a scaling to
$K = \tfrac12 N_\mathrm{df} k_B T$, with $N_\mathrm{df}$ the degrees of
freedom of the log. The system is prepared by the same path as `compile`
(an NVE ensemble at `temperature`), so the velocities are those of
`mdir run` bit for bit when the control file has the same topology,
coordinates, constraints, `[ensemble] temperature`, and `[dynamics] seed`.
`mdir run` takes its one seed for the velocities and the thermostat; in
Python `Ensemble.seed` is the thermostat's, and a run that should repeat
`mdir run` sets both to the same value. The draw has a generator of its own
and moves no other random stream.

| Argument | Type | Unit | Accepted |
|---|---|---|---|
| `system` | `System` | | what `compile` accepts, with the particle count of the state |
| `temperature` | float | K | finite, at least 0 (0 gives velocities of 0) |
| `seed` | int | | 0 to $2^{63}-1$, as `[dynamics] seed` |

A refusal raises `InputError` (or `UnsupportedError` for what `compile`
does not support) and returns nothing. To repeat `mdir run`, which takes
one seed for both, give `draw_velocities` and `Ensemble.seed` the same
seed. A state without velocities still
starts at rest in a simulation (D196); drawing them is explicit.

## Restraints

```python
heavy = mdir.Restraint()
heavy.selection = "!:WAT & !@H*"
heavy.force_constant = 4184.0          # kJ/mol/nm^2, 10 kcal/mol/Å^2
heavy.reference_scaling = mdir.ReferenceScaling.Center
system.restraints = [heavy]
system.restraint_reference = reference_positions   # optional, (N, 3) nm
```

`System.restraints` is a list of `mdir.Restraint`, one for each
`[[restraints]]` table, with the same keys:

| Attribute | Type | Unit | Control file |
|---|---|---|---|
| `selection` | str | | `selection`: the mask of Amber of Section 22 of [design-m1.md](design-m1.md) |
| `force_constant` | float | kJ/mol/nm² | `force_constant` in kcal/mol/Å²: 1 kcal/mol/Å² is 418.4 kJ/mol/nm² (the public units of D191) |
| `reference_scaling` | `ReferenceScaling.Center` (default) or `.All` | | `reference_scaling = "CENTER"` or `"ALL"` (D124) |

The energy is $k \lVert\mathbf x - \mathbf x^\mathrm{ref}\rVert^2$ for each
selected particle with mass, constants of restraints that select the same
particle add, and a barostat moves the references as D124 says. The
selection, the sum, the reference scaling, and the evaluation are those of
the control file: `compile` turns the list into the control's restraints and
prepares them by the same code. The constant is converted to the control
file's unit and back by that code; the conversion takes the value in
kcal/mol/Å² that gives the constant back exactly where one exists, which is
always so for a constant computed from a control-file value as the CLI
computes it ($k \cdot 4.184 / 0.1^2$), and otherwise keeps it to within one
rounding.

`System.restraint_reference` is the reference $\mathbf r$, the positions of
the file of coordinates in the control file: `(N, 3)` float64 in nm, in
input order, with `N` the particle count of the system. Absent (shape
`(0, 3)`, the default), it is the positions of the `InitialState` given to
`compile`. A stage that begins from the end of another and restrains to the
structure it began with sets it. Its cell is that of the state given to
`compile`: under a barostat the references scale with the cell from there.
`mdir run` scales from the cell of its coordinates file, also when it
begins from another run's checkpoint; the two agree when the stage before
kept the cell (NVT before NPT, as in the standard pipeline). A separate
reference cell can be added if a case needs it.

The list and the reference cross the boundary as the other D191/D193
collections: reading returns copies (a list of new `Restraint` values, a
read-only array), assignment copies, validates the shape and dtype of the
reference (native float64, C-contiguous, finite, `(N, 3)`), and advances the
version of the system, so a program compiled before is stale.

| Refusal | Raised | When |
|---|---|---|
| An empty `selection` | `InputError` | `compile`, `draw_velocities` |
| A `force_constant` that is not positive and finite | `InputError` | `compile`, `draw_velocities` |
| A mask that does not parse (a number 0 or a reversed range among them) | `InputError` | `compile`, `draw_velocities` |
| A particle selected with both `Center` and `All` | `InputError` | `compile`, `draw_velocities` |
| A reference of another shape, dtype, or particle count, or not finite | `InputError` | assignment |
| A list holding anything but `Restraint` | `TypeError` | assignment |

A selection that selects no particle with mass restrains nothing; the
control file warns of it, and `compile` raises a Python `UserWarning` with
the same text. As for `pair_terms`, `system.restraints.append(r)` changes
a copy; assign the list.

## Validation

`test/Driver/python-velocities-restraints.test` (CPU) and its `-gpu` twin
use the dipeptide in water (1168 particles). The oracle is `mdir run` of the
same control file.

- Drawn velocities against those of `mdir run` (its `readSystem` and
  `assignVelocities`, written as raw f64 by `mdir-model-test --drawn`), with
  and without SETTLE/SHAKE: 0 differing bits of 3504 values, tolerance 0.
  The native `python-model` tests compare them in every file case.
- The file and the object model with three overlapping restraints (CENTER at
  10 and 2.5 kcal/mol/Å², ALL at 5) under NPT give identical restraint
  constants, scalings, references, fields, and semantic IR, on CPU and GPU
  targets in mixed and double (`python-model.test`, `python-model-gpu.test`).
- Runs of 20 steps (0.5 fs, V-RESCALE and C-RESCALE every 10 steps) with
  those restraints and drawn velocities against `mdir run`. The row of step
  10 matches every printed digit, except on the GPU in mixed precision,
  where it agrees within 1e-6 of each value; that case differs the same way
  without restraints (#97). The state at step 20, which the model reaches in
  two parts, against the checkpoint of `mdir run` (largest absolute
  difference / tolerance; tolerances as in the test):

| Target | Ensemble | Precision | Positions (nm) | Velocities (nm/ps) | Forces (kJ/mol/nm) |
|---|---|---|---|---|---|
| CPU | NVT | double | 1.3e-15 / 2.5e-14 | 7.8e-13 / 6.1e-12 | 4.3e-11 / 2.2e-9 |
| CPU | NPT | double | 2.4e-15 / 2.5e-14 | 9.6e-13 / 6.1e-12 | 1.3e-11 / 2.2e-9 |
| CPU | NVT | mixed | 2.4e-8 / 1.6e-7 | 1.8e-6 / 3.0e-5 | 2.2e-3 / 0.12 |
| CPU | NPT | mixed | 2.5e-8 / 1.6e-7 | 1.0e-6 / 3.0e-5 | 1.4e-3 / 0.12 |
| GPU | NVT | double | 4.0e-15 / 2.5e-14 | 1.7e-12 / 6.1e-12 | 1.0e-10 / 2.2e-9 |
| GPU | NPT | double | 5.3e-15 / 2.5e-14 | 1.1e-12 / 6.1e-12 | 1.2e-10 / 2.2e-9 |
| GPU | NVT | mixed | 2.0e-8 / 1.5e-7 | 3.5e-6 / 2.4e-5 | 1.0e-2 / 0.12 |
| GPU | NPT | mixed | 2.5e-8 / 1.5e-7 | 1.3e-6 / 2.4e-5 | 2.2e-3 / 0.12 |

  In double the tolerance is $c\,\epsilon_{64} \max_i\lvert q_i\rvert$ with
  $c = 40$ for positions and $c = 4(N-1)$ for velocities and forces, as in
  [python-segments.md](python-segments.md); in mixed it is $3E$, with $E$
  the difference of `mdir run` in mixed and in double under NVT (under NPT
  the scalings of the cell amplify that difference to 32 kJ/mol/nm in the
  forces, while the model and `mdir run` still differ only by a part
  boundary). The restraints change the potential at step 10 by 0.50 kJ/mol,
  far above the printed digits that match; references given explicitly as
  the state's positions change no bit; references moved by 0.01 nm change
  the potential by more than 1 kJ/mol.
- Refusals: negative, NaN, and infinite temperatures, seeds $-1$ and
  $2^{63}$, a missing system, a state of another particle count; an empty
  selection, constants 0 and NaN, a mask that does not parse, conflicting
  scalings (at `compile` and at the draw); references of another count, in
  float32, as a list, or with NaN; a list of non-restraints (`TypeError`);
  a selection of nothing (`UserWarning`). Copies and versions: editing a
  returned restraint changes nothing, assignment makes a program stale.

## Not in this item

Checkpoints (item 5) will record the restraints and their reference in the
fingerprint as D172 does. Restraints by expressions of absolute positions
(D148) stay outside the D191 subset.
