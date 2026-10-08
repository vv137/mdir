# `observe` in the Python model (D[python-observe])

Issue #188, M2a item 8 of [python-m2.md](python-m2.md), the last. A term of
the Python model given by an expression selects constants to observe, as
`observe = [constants]` does in the control file (D189): a Python simulation
writes the columns of `[output] observables` through a reporter and gives the
same values to a script at every step of energy, without reading a file. This
document is the design of the draft; the measurements follow the
implementation.

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
  `@observe<k>` to the program (D189), so setting it makes a program stale.
  A program whose terms do not observe is the program of `main`, text for
  text.
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

### Terms of the positions (D[python-external])

The walls of D189 are terms of the absolute positions, which the Python
model did not have. `mdir.ExternalTerm` is the control file's
`[[energy.external]]` (D148): `name`, `expression` in `x`, `y`, `z` (nm),
the charge `q` (e), the time `t` (ps), and its constants, in kJ/mol;
`selection` (a mask of Amber) or `particles` (zero-based indices), one of
the two; `constants`, ordered `(name, number)` pairs; `scaling`
(`ExternalScaling.None_`, fixed in space, or `ExternalScaling.Cell`, in the
frame of the cell, D154), which a model with a barostat must give, as the
control file must; and `observe`. A parameter with a value for each particle
is not in this subset. The fingerprint (D223) gains the entry
`[python] external_terms`.

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
  present at the simulation's first run writes the row of step 0; an
  existing file is backed up as `mdir run` backs up its outputs; rows are
  appended across `run` calls; `close_reporters()` or the simulation's end
  closes it; a simulation continued from a checkpoint with `append=True`
  keeps the rows through the checkpoint's step and appends, and with
  `append=False` writes `<name>.partNNNN<ext>`; with the barostat of Trotter
  type the period must be a multiple of the coupling period.
- With tunables in the program, the last column is `tunables_version`, as in
  the energy file (D213).

### Reading the values

`State.observables` is a dict from the names of the columns to floats, in
kJ/mol and kJ/mol per unit of the constant (the units of `State.energies`),
or `None`. It is set where `State.energies` is: after a run that ended with
a step of energy (`run(n, energy=True)`, `run(0, energy=True)`), and in the
state that a `CallbackReporter` is called with. The values are those of the
row of that step: the file holds them divided by 4.184, to six decimals. No
reporter is needed to read them.

### Cost

The program evaluates `@observe<k>` at its steps of energy only (the rows of
the reports, the steps of callbacks, `run(n, energy=True)`), as `mdir run`
does at the rows of its log; the steps between them are unchanged.

## Refusals

| Case | Error |
|---|---|
| `observe` names what is not a constant of the term | `InputError`: the term '`<term>`' observes '`<name>`', which is not a constant of the term given as one number |
| `observe` names a parameter of a tuple term whose values differ among its tuples | `InputError`: ... which has a value for each particle or tuple; observe a constant given as one number |
| A name twice in `observe` | `InputError`: '`<name>`' is observed twice |
| `observe` that is not `None` or a list of strings | `TypeError` |
| An `ObservablesReporter` on a simulation whose program observes nothing | `InputError`, at the run that would open the file |
| Two `ObservablesReporter`s | `InputError`: a simulation takes one |
| A period that is not a multiple of the coupling period, with the barostat of Trotter type | `InputError`, as for the other built-in reporters |
| An observed constant that a tunable takes | see below |

A minimization does not observe: its program has no `@observe<k>`, its
`State.observables` is `None`, and it takes no reporters (D207). The control
file refuses `observables` in a minimization because its file would be
empty; a Python `System` is shared by the stages of a pipeline, so `observe`
on its terms is left out of the minimization without an error. Terms that
observe without a reporter are allowed (the control file refuses `observe`
without `observables`): the values are read from the state.

## Fingerprint and checkpoints

As D189 has it for the control file, `observe` and the file enter neither
the fingerprint nor the checkpoint: observing changes no physics, and a
checkpoint written with `observe` continues without it and the reverse, in
either front end. `model_sha256` is made of the fingerprint and does not
change; `plan_sha256`, which hashes the program's text, does, and is not
compared at a continuation.

## Tunables

Today the builder refuses a program with tunables and `observe`. Proposed:

1. Tunables and `observe` in one program are allowed. The columns of an
   observed term follow the updates of the tunables that it reads (the
   charges, another constant of the term), its tail and shift estimate are
   computed anew at an update as the other constants of the energy are, and
   the row says of which values it is (`tunables_version`).
2. An observed constant may itself be a tunable when the tunable has one
   entry that every site takes: a constant of a pair term, or a parameter of
   a tuple term whose map sends every tuple to one entry. Its value reaches
   `@observe<k>` as a value of the entry instead of a constant of the text,
   so an update does not change the program. The column equals
   `Simulation.tunables.gradient()[name][0]` (D230), which differentiates
   the same shifted potential with the same tail and shift estimate.
   A tunable with several entries on an observed parameter is an
   `InputError` of `mdir.compile` that names `gradient()`.

The alternatives, and the question to the maintainer, are in the pull
request.

## Validation (plan)

- The gate of item 8: the file of a Python simulation against that of
  `mdir run` with `observe` on the same model, byte for byte in the
  deterministic mode, on the CPU and a GPU, in mixed and double precision:
  a pair term, the walls of D189, and a tuple term on the dipeptide in
  water; a pair term under a plain cutoff and under the shift with its tail
  and shift estimate (D209, D210) on the mixture of `pair-dispersion.test`.
- `State.observables` in a callback at every row against the rows of the
  file.
- A run in parts, and a run continued from a checkpoint, against the file
  of one run.
- Each refusal with its message.
- A model without `observe`: the module equal to `main`'s, and the rate on
  JAC.
