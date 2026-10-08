# `observe` in the Python model (D[python-observe], D[python-external])

Issue #188, M2a item 8 of [python-m2.md](python-m2.md), the last. A term of
the Python model given by an expression selects constants to observe, as
`observe = [constants]` does in the control file (D189): a Python simulation
writes the columns of `[output] observables` through a reporter and gives the
same values to a script at every step of energy, without reading a file.
Terms of the absolute positions, which the walls of D189 are, enter the
Python model with it (D[python-external]).

## Interface

```python
soft = mdir.PairTerm()
soft.name, soft.expression = "soft", "a*exp(-r/l)"
soft.constants = [("a", 2.0), ("l", 0.05)]
soft.observe = ["l"]                 # the energy of the term and dU/dl
system.pair_terms = [soft]

wall = mdir.ExternalTerm()           # D[python-external]
wall.name, wall.expression = "lower", "0.5*k*max(0, z0 - z)^2"
wall.selection = ":WAT"
wall.constants = [("k", 4184.0), ("z0", 0.6)]
wall.observe = ["z0"]
system.external_terms = [wall]

program = mdir.compile(system, state, integrator, ensemble, execution, schedule)
program.plan["observables"]          # ["soft.energy", "soft.d_l", "lower.energy", "lower.d_z0"]

sim = mdir.Simulation(program)
sim.reporters.append(mdir.ObservablesReporter("prod.obs", period=1000))
sim.reporters.append(mdir.CallbackReporter(lambda s, state: use(state.observables), period=5000))
sim.run(500_000)
sim.run(0, energy=True); sim.state().observables["lower.d_z0"]
```

### Selecting what is observed

- `PairTerm.observe`, `TupleTerm.observe`, and `ExternalTerm.observe` take
  `None` (the default: the term is not observed) or a list of names of
  constants of the term, as the control file's key: the term's energy is
  observed, and its derivative in each name listed; `[]` observes the energy
  alone. Any other value is a `TypeError`.
- A constant that can be observed is one number for the whole term: a
  constant of a pair term or of a term of the positions, and a parameter of a
  tuple term whose values are equal for all its tuples (the control file's
  rule for a parameter given as one number).
- `observe` is a compile input: each observed term adds a potential
  `@observe<k>` to the program (D189), so setting it on the terms of a system
  makes its programs stale. A program whose terms do not observe is the
  program of before, text for text.
- The columns follow the terms in the order `System.pair_terms`,
  `System.tuple_terms`, `System.external_terms`, each list in its order, and
  within a term the energy, then the constants in the order of `observe`:
  `<term>.energy`, `<term>.d_<constant>`. `Program.plan["observables"]`
  lists the names.
- What is observed is what D189, D209, and D210 define: the term alone,
  under a plain cutoff the potential that the forces sample (shifted to 0 at
  the cutoff), and for a pair term whose tail is in the correction for the
  dispersion its tail and the estimate of its shift, at the volume of the
  cell, in the energy and in each derivative.

### The reporter

- `ObservablesReporter(file, period)` writes the column file of
  `[output] observables` (D149, D189): the lines `# step time <columns>` and
  `# - ps kcal/mol kcal/mol/<constant> ...`, then a row at each step that is
  a multiple of `period`, with `%.6f` digits, in kcal/mol and kcal/mol per
  unit of the constant, the constant in the unit in which the term of the
  Python model takes it (nm for a length). For the same model, the file is
  that of `mdir run`.
- Its period is its own: it need not be that of the `EnergyReporter`. The
  built-in reporters share the steps of energy inside the parts of a run
  (D207): the interval of the reports is the greatest common divisor of
  their periods, and the host writes the row of each where it is due.
- As the energy file (D207, D223): a simulation takes one; a reporter
  present at the simulation's first run writes the row of step 0, and one
  added later begins with its next row; an existing file is backed up as
  `mdir run` backs up its outputs; rows are appended across `run` calls;
  `close_reporters()` or the simulation's end closes it; a simulation
  continued from a checkpoint with `append=True` keeps the rows through the
  checkpoint's step and appends, and with `append=False` writes
  `<name>.partNNNN<ext>`; with the barostat of Trotter type the period must
  be a multiple of the coupling period.
- With tunables in the program, the last column is `tunables_version`, as in
  the energy file (D213).

### Reading the values

`State.observables` is a dict from the names of the columns to floats, in
kJ/mol and kJ/mol per unit of the constant (the units of `State.energies`),
or `None`. It is set where `State.energies` is and the program observes:
after a run that ended with a step of energy (`run(n, energy=True)`,
`run(0, energy=True)`), and in the state that a `CallbackReporter` is called
with. The values are those of the row of that step: the file holds them
divided by 4.184, to six decimals. No reporter is needed to read them, and
terms may observe without one (the control file refuses `observe` without
`observables`, whose file would be missing).

### Cost

The program evaluates `@observe<k>` at its steps of energy only (the rows of
the reports, the steps of callbacks, `run(n, energy=True)`), as `mdir run`
does at the rows of its log; the steps between them are unchanged. The
program of segments gains the call of D189 at its plain step of energy,
beside those of its start and of the intervals of its reports, which
`mdir run`'s programs have.

## Terms of the positions (D[python-external])

`mdir.ExternalTerm` is the control file's `[[energy.external]]` (D148), the
custom external force of OpenMM [[Eastman2017]](references.md#eastman2017),
prepared and emitted by the code of the control file:

| Attribute | Meaning |
|---|---|
| `name`, `expression` | The energy of one particle in kJ/mol, in `x`, `y`, `z` (nm, never wrapped), the charge `q` (e), and the constants |
| `selection` or `particles` | A mask of Amber, or zero-based indices (a one-dimensional integer array); one of the two |
| `constants` | Ordered `(name, number)` pairs |
| `scaling` | `None` (unset), `ExternalScaling.None_` (fixed in space), or `ExternalScaling.Cell` (in the frame of the cell, D154); a model with a barostat must give it, as the control file must |
| `observe` | As above |

`System.external_terms` is the list, a compile input. Not in this subset,
and refused: a parameter with a value for each particle, and the time `t`.
The fingerprint of a model with such terms (D223) has the entry
`[python] external_terms`, a hash of their names, expressions, particles,
constants, and scaling, so `mdir run --continue` refuses a checkpoint of
such a model and begins a stage from it, as with `[python] pair_terms`.

## Refusals

| Case | Error |
|---|---|
| `observe` names what is not a constant of the term | `InputError`: the term '`<term>`' observes '`<name>`', which is not a constant of the term given as one number |
| `observe` names a parameter of a tuple term whose values differ among its tuples | `InputError`: ... which has a value for each particle or tuple; observe a constant given as one number |
| A name twice in `observe` | `InputError`: the term '`<term>`': '`<name>`' is observed twice |
| `observe` that is not `None` or a list of strings | `TypeError` |
| `observe` on a term of a program that minimizes | `InputError` of `mdir.compile`: the term '`<term>`' gives 'observe', which is evaluated at the energies of a run of dynamics; a minimization does not take it |
| An `ObservablesReporter` on a simulation whose program observes nothing | `InputError`, at the run that would open the file |
| Two `ObservablesReporter`s | `InputError`: a simulation takes one ObservablesReporter |
| A period that is not a multiple of the coupling period, with the barostat of Trotter type | `InputError`, as for the other built-in reporters |
| An observed parameter that a tunable takes with several entries, or leaves to some sites | `InputError` naming `Simulation.tunables.gradient()` |
| An `ExternalTerm` with both or neither of `selection` and `particles`, a particle twice or beyond the system, a constant named `x`, `y`, `z`, `q`, or `t`, a name the expression does not declare, no `scaling` under a barostat | `InputError` |

A minimization follows `mdir run`, which refuses `observables` under
`[minimize]` ("'observables' is written at the energies of a run of
dynamics"): the program that minimizes is compiled from a system whose terms
do not observe. In a pipeline that shares one `System`, set `observe` for the
stages of dynamics.

## Fingerprint and checkpoints

As D189 has it for the control file, `observe` and the file enter neither
the fingerprint nor the checkpoint: observing changes no physics, and a
checkpoint written with `observe` continues without it and the reverse, in
either front end. `model_sha256` is made of the fingerprint and does not
change; `plan_sha256`, which hashes the program's text, does, and is not
compared at a continuation.

## Tunables

The builder refused a program with tunables and `observe`. By the
maintainer's decision on PR #213:

1. Tunables and `observe` share a program. The columns of an observed term
   follow the updates of the tunables that it reads (the charges, another
   constant of the term); its tail and shift estimate are computed anew at
   an update, as the other constants of the energy are (D213); and the row
   says of which values it is (`tunables_version`).
2. An observed constant may itself be a tunable that has one entry which
   every site takes: a constant of a pair term, or a parameter of a tuple
   term whose map sends every tuple to entry 0. The observed potential takes
   an observed constant as a scalar argument (D189); for a tunable the
   argument is a value of the entry (`Program::startValues`, D227) instead
   of a constant of the text, so an update does not change the program and
   the steps are unchanged. The column `d_<c>` equals
   `Simulation.tunables.gradient()[name][0]` (D230): both differentiate the
   potential that the forces sample, with the same tail and shift estimate.
3. A tunable with several entries on an observed parameter, or one whose map
   leaves a tuple its value, is an `InputError` of `mdir.compile` that names
   `gradient()`, which gives the derivative in every entry.

## Validation

`test/Driver/python-observe.test` (CPU) and `python-observe-gpu.test`
(RTX 3090) run `Inputs/python_observe.py`, in the deterministic mode, in
double and mixed precision. The control file of the same model carries the
expression as the Python model hands it to the builder (`r` in nm as
`(r*0.1)`, the energy over 4.184) and the same numbers for its constants.

| Check | Reference | Result (CPU and GPU, double and mixed) | Tolerance |
|---|---|---|---|
| Dipeptide in water, NVE, 20 steps, rows every 5: a pair term $a\,e^{-r/l}$ (`l`, `a`), a term over two bonds (`r0`), the two walls of D189 on the waters (`z0`; `z0`, `k`); plain cutoff and shift | the observables file and the energy file of `mdir run` | equal byte for byte, 5 rows of 10 columns | 0 |
| `State.observables` in a callback at each row and after `run(0, energy=True)`, formatted as the file formats them | the rows of `mdir run`'s file | equal | 0 |
| Mixture of `pair-dispersion.test` (60 A + 60 B, 32 Å, $r_c$ = 12 Å), an NBFIX written as a pair term between A and B (`sig`, `eps`), plain cutoff and shift, NVE | the observables file of `mdir run` | equal byte for byte | 0 |
| What the correction adds to its columns (the correction on less off) | closed forms of the tail and the shift estimate (D209, D210): $-1.921376$ kcal/mol, $-43.801428$ kcal/mol/nm, $-1.291884$ kcal/mol per kJ/mol | equal | $10^{-8}$ relative |
| The same mixture in NPT (C-rescale and V-rescale every 10 steps, 40 steps, rows every 10), whose tails follow the volume | the observables file of `mdir run` | equal byte for byte | 0 |
| Runs of 7 and 13 steps; energies every 4 with observables every 6 over 24 steps | the file of one run; `mdir run` with rows every 6 | equal byte for byte | 0 |
| Continued from a checkpoint at step 10 with `append=True`, after a row at step 15 that the continuation writes again | the file of one run | equal byte for byte; with `append=False`, `ck.part0002.obs` holds the rows that follow | 0 |
| The same checkpoint continued by a program whose pair term does not observe | the uninterrupted run, `lower.d_z0` at step 20 | no warning; within $6.7\times10^{-16}$ (the sums follow the order of the particles of an activation) | $10^{-12}$ |
| Tunable `a` of the observed pair term updated from 2 to 3 | 1.5 times the columns before, tail included | within $10^{-12}$ (double), $10^{-5}$ (mixed); rows 0 and 5 of version 0, row 10 of version 1 | |
| Observed tunable `l` at 0.05 and 0.06 nm: `soft.d_l` | `gradient()["soft_l"][0]` (D230), and a compile with the value | within $1.1\times10^{-16}$ (double), $3.5\times10^{-7}$ (mixed) | $10^{-12}$, $10^{-5}$ |
| The same | central differences of `soft.energy` by updates, $h = 10^{-5}$ nm | within $2.3\times10^{-8}$ (double), $9.9\times10^{-5}$ (mixed) | $10^{-7}$, $10^{-3}$ |
| Observed tunable `r0` of the bonds, one entry for both: `flat.d_r0` | `gradient()["rest"][0]` | equal to the bit | $10^{-12}$, $10^{-5}$ |
| The refusals of the table above | | each with its message | |

The terms of the positions by themselves (D[python-external]), two walls of
4184 kJ/mol/nm² on the 1146 atoms of the waters of the dipeptide:

| Check | Reference | Result (CPU and GPU, double and mixed) | Tolerance |
|---|---|---|---|
| Energies of the walls at the start | NumPy, $\tfrac12k\sum\max(0, z_0 - z)^2$: 5035.071714 and 3389.462793 kcal/mol (the first is the value written out by hand in `observables.test`) | within $8.9\times10^{-16}$ | $10^{-12}$ (double), $2\times10^{-6}$ (mixed) |
| Their forces, the forces with the walls less those without | NumPy, $k\max(0, z_0 - z)$ | within $1.2\times10^{-15}$ of the largest | the same |
| 10 steps, NVE | `mdir run` with the same `[[energy.external]]` terms: its energy file, and the positions, velocities, and forces of its checkpoint | byte for byte, and to the bit | 0 |
| Particles by index against the mask `:WAT` | | energies and observed values equal | 0 |

A model without `observe`: the modules and the lowered IR of `mdir emit` on
JAC (NVE, NPT) and Factor IX (NPT) of the Amber suite, and the module, the
lowered IR, and the program of segments (`plan_sha256`) of five Python
models (plain, with pair and tuple terms, with tunables and their
derivative; NVE, NVT, NPT; CPU and GPU; mixed and double) equal those of
`main` hash for hash, so the rates are `main`'s and none was measured.
