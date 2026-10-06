# Tunable parameters of a Python model (D[python-tunable])

Issue #130, the M2a gate that D192 left open and the base of M2b
(differentiable simulation, [roadmap](roadmap.md), Section 6.1; D195).
Status: implemented; the questions put to the maintainer on PR #159 are
marked below.

Before this item every change of a parameter of a Python model meant a new
`mdir.compile` and a new `mdir.Simulation`: a lowering and a JIT of the
whole program. M2b evaluates $U_\theta$ at many $\theta$ and samples again
whenever the effective sample size drops; parameter scans and umbrella
windows have the same need. A parameter declared *tunable* is a value of the
compiled program, not a constant of it, and a simulation takes new values
without compiling.

## Interface

```python
loaded = mdir.load_amber("system.prmtop", "system.inpcrd")
system, state = loaded.make_system(), loaded.make_state()
system.tunables = [
    mdir.Tunable("q", "charge", map=charge_map),    # tied charges
    mdir.Tunable("sigma", "sigma"),                 # per Lennard-Jones type
    mdir.Tunable("epsilon", "epsilon"),
    mdir.Tunable("wall_k", "k", term="wall"),       # a constant of a pair term
]
program = mdir.compile(system, state, integrator, ensemble, execution, schedule)
sim = mdir.Simulation(program)
sim.run(10_000)
sim.tunables["q"]                                    # (M,) float64, read-only copy
sim.tunables.update({"sigma": s1, "epsilon": e1})   # one atomic update
sim.tunables["wall_k"] = np.array([2.5])            # an update of one
sim.tunables.version, sim.tunables.history           # 1, [(0, 0), (10000, 1)]
sim.state().tunables_version
sim.run(10_000)
```

`mdir.Tunable(name, parameter, term=None, map=None, values=None,
mixing=None)` declares one tunable. The declarations are a list on
`System`; assigning it advances the version of the system, so a compiled
program goes stale (D192): declaring is a structural change, an update of
values is not.

### What can be tunable

Every tunable is a vector $\theta_\text{name}\in\mathbb{R}^M$ with an index
map from the *sites* of its parameter to its entries.

| `parameter` | `term` | Sites | Unit |
|---|---|---|---|
| `"charge"` | none | the particles, $N$ | e |
| `"sigma"` | none | the Lennard-Jones types, $T$ (`System.type_names`) | nm |
| `"epsilon"` | none | the Lennard-Jones types, $T$ | kJ/mol |
| a constant of a pair term | the term's name | one | that of the expression |
| a parameter of a tuple term | the term's name | its tuples, $n$ | that of the expression |

`map` is an integer array with one entry per site: the entry of $\theta$
that the site takes, or $-1$ for a site that keeps the value of the model.
Without a map each site is an entry of its own ($M$ equals the number of
sites). Every entry from 0 to $M-1$ must be the image of a site. A tied map
(one charge per atom name of a residue, one $\sigma$ for two types) is the
same mechanism. `values` gives the initial $\theta$, shape $(M,)$; without
it the initial value of an entry is the common value of its sites in the
model, and sites of one entry that differ are an `InputError` naming the
entry. Names are unique; a parameter may be declared by one tunable only.

The set of tunables is a flat collection of named one-dimensional arrays,
the shape of their gradients, so that a framework holds it as a dict or a
pytree (M2b). The Python model exposes what maps are built from:
`System.charges` $(N,)$, `System.particle_types` $(N,)$ int64,
`System.type_names`, `System.atom_names`, and `System.residue_names` (one per
particle), all read-only.

### Per-type Lennard-Jones and the combining rule

The topology holds $\sigma_{ab}$ and $\epsilon_{ab}$ for every pair of types.
A tunable `"sigma"` or `"epsilon"` is per type: the program's table is
$\sigma_{ab} = (\sigma_a+\sigma_b)/2$ (Lorentz; `mixing="geometric"`
gives $\sqrt{\sigma_a\sigma_b}$) and
$\epsilon_{ab}=\sqrt{\epsilon_a\epsilon_b}$ (Berthelot), recomputed on the
host from the per-type values. At compile time the initial per-type values
are the diagonal of the table, and the table must follow the rule: an entry
that differs from the rule by more than $10^{-6}$ relative (an NBFIX of
CHARMM or GROMACS, an edited Amber pair) is refused, naming the pair of
types. Amber's tables follow it within $2\times10^{-7}$ (the eight digits of
`ACOEF`/`BCOEF`; JAC, cellulose, factor IX, the dipeptide); the table of the
rule replaces them, so a tunable run differs from one without tunables at
that level. Pairs of types where both $\epsilon$ are 0 keep
$\epsilon_{ab}=0$, whatever their $\sigma$.

The 1-4 pairs keep the $\sigma$ and $\epsilon$ that the topology gives them,
as a compile from edited types would: their parameters are their own
(Amber copies them from the table of the file, GROMACS from `[ pairtypes ]`
or the rule at reading, CHARMM from its 1-4 parameters).

## Derived quantities

Values that depend on a tunable are recomputed on every update, on the
host, by the code that computes them at compile time (the builder): an
update builds the program's values from the model with the new $\theta$
applied, exactly as a compile with `values=` would. What each tunable
reaches:

| Tunable | Runtime values it changes |
|---|---|
| charge | the field `q`; the 1-4 products $fq_iq_js_C$; the self term and the net-charge background of PME (and their virial), or the self term of the reaction field; the tails (D209) and the shift estimate (D210) of pair terms that read `q1`/`q2` (their classes of particles are rebuilt) |
| sigma, epsilon | the tables `lj_sigma` and `lj_epsilon`; the correction for the dispersion $\langle C_6\rangle$ and its shift estimate; the tails of pair terms that read `sigma`/`epsilon` |
| a constant of a pair term | a table of the program's scalars, read by the term's kernel; the term's tail and shift estimate |
| a parameter of a tuple term | that tuple field |

The program takes these as runtime values: per-particle fields, type tables,
and tuple fields are entry buffers already (D196 builds them anew for every
part); a tunable constant of an expression is read from a $1\times K$ table
instead of being an `arith.constant`; and, with a barostat, the virial and
energy of the constant terms that the barostat adds (`%baro_constant`,
`%baro_energy_constant`) become arguments of the entry. The energies that
the host adds to the log and the reports (the dispersion correction, the
PME constants) come from the rebuilt values. Nothing changes for a program
without tunables.

**Not differentiated here.** The sampler is not differentiated (Section 6.1
of the roadmap). The issue proposes an IR prologue so that `Differentiate`
covers the derived quantities. The first consumer of those derivatives is
the M2b frame evaluator, a potential of its own; this item computes them on
the host for the sampler, bitwise as a compile does, and leaves their
derivatives to the evaluator: the per-site quantities (the table of the
rule, the 1-4 products, the maps) and the sums over particles ($\sum q_i^2$,
$\sum q_i$, $\langle C_6\rangle$) in its IR, and the tails, host quadratures
(D209), as runtime scalars with their host Richardson derivatives, an
explicit exception. This is a question to the maintainer on the PR (Q1).

**How an update builds them.** `Simulation` keeps the system that its
program was built from (at its first step) and the control it was built
with. An update copies both, puts the new values into them
(`model::applyTunables`: the charges, the per-type σ and ε and the table
of the rule, the constants of pair terms, the parameters of tuple terms,
and the classes and values of the tails, `driver::recollectPairTails`),
and runs `driver::buildProgram` on the copy with the compiled program's
neighbor width. The new program's text must equal the compiled one's; its
values replace the old, and the next call of the entry builds its buffers
from them as every call does (D196). Everything else in the update is a
check: the cost is that of building the program's text and values, about
10 ms on the host for JAC (23,558 atoms).

## Updates

- `sim.tunables` is a mapping from names to read-only $(M,)$ float64 copies
  (`dict(sim.tunables)` is a plain dict). `update(mapping)` assigns any
  subset at once: every value is checked (known name, shape $(M,)$, finite,
  $\sigma\ge0$ and $\epsilon\ge0$, charges below 100 e under PME), the
  program's values are rebuilt, and only then is anything changed. A
  failure changes nothing.
- An update advances the *value version* (`sim.tunables.version`, 0 at
  creation), not the structure version of D192: the program stays current.
  `sim.tunables.history` lists `(step, version)`: version `v` is in effect for
  the steps after `step`.
- The forces that the next step begins with were those of the old values.
  After an update during a run the simulation evaluates them anew at its
  state: a call of the entry with no steps and `%first_call = 2`, which
  redoes the evaluation of the start without the half kick back of
  leapfrog (a select on `%first_call` in a program with tunables) and
  writes no row of the energy file. `state().forces` is then that of the
  new values; with velocity Verlet `state().energies` holds the energies of
  the state at the new values; with leapfrog, whose velocities are half a
  step behind, it is `None`. An update before the first run evaluates
  nothing: the first call does. If the evaluation fails, the update is
  undone. Neighbor structures are built anew at every part anyway.
- `run(0, energy=True)` makes the same evaluation without an update
  ([python-segments.md](python-segments.md#evaluations-without-a-step)):
  the energies of the state at the current values, for scans and finite
  differences at fixed positions.
- `State.tunables_version` is the version of its forces and energies. An
  `EnergyReporter` of a simulation with tunables writes a last column,
  `tunables_version`; a trajectory's frames take theirs from the history.
- **Structural changes are refused as an update**: the declarations, maps,
  particle count, constraints, PME grid, cutoff and the expressions are
  inputs of the compile, and changing them makes the program stale. An
  update is accepted only if the program that the new values build is the
  compiled one, text for text; a value that would change the program (a
  pair term whose tail diverges at the new values, for instance) raises
  `InputError` and leaves the simulation as it was.
- Checkpoint provenance (#132) records the declarations and the values;
  Python checkpoints are M2a item 5, so this item exposes them
  (`program.plan["tunables"]`, `sim.tunables`) for it.

## Units and precision

Values are in MD units, as the model's: e, nm, kJ/mol, and for a constant
or parameter of an expression the units of the expression. Setters take
OpenMM quantities for charges, $\sigma$, and $\epsilon$ (D200). Host values
are float64. The device copies follow the precision of other parameters:
fields in the type of parameters (f32 in mixed precision), tables and tuple
fields as the program takes them today. The derivatives that M2b needs are
accumulated in f64 by its evaluator.

## Refused combinations

A tunable `sigma` or `epsilon` with a table that the rule does not
reproduce; an unknown term or parameter; a parameter declared twice; a map
of the wrong length, with entries outside $[-1, M)$, or with an entry no
site takes; `values` of the wrong shape or not finite; a name that is not
an identifier, or used twice; `mixing` for a parameter other than `sigma`.
A tunable needs a model with a topology, and the builder refuses tunables
with `[free_energy]`, `observe`, LJPME, or pulls, which the Python model
does not take yet. An update is refused (`InputError`, nothing changed)
for an unknown name, a shape other than $(M,)$, a value that is not finite,
negative σ or ε, charges of 100 e or more under PME, a pair term whose tail
would enter or leave the correction for the dispersion at the new values,
and any value that would change the program's text; a simulation whose
program declares no tunables refuses every update.

## Validation

Tests `python-tunable-*.test` (`Inputs/python_tunable.py`), on the
dipeptide in water (1,168 atoms) with PME, a pair term
`a*exp(-r/l)` and springs over three pairs of atoms; the tunables are the
charges tied by a map (one entry per atom name of the waters), σ and ε of
the 9 types, the constant `a`, and the force constants of the springs tied
in two.

| Check | Reference | Result | Tolerance |
|---|---|---|---|
| Forces and energies after an update at step 8, and positions, velocities, forces, energies 12 steps later (velocity Verlet, NVE, deterministic) | A compile with the new values from the state of step 8, evaluated first | Equal to the bit: CPU and GPU, double and mixed | 0 |
| NPT, an update before the first run, 30 steps | A compile with the new values | Equal to the bit, volume included: CPU and GPU, double and mixed | 0 |
| Leapfrog through an update | The state before it | Positions and velocities equal to the bit; forces back at the old values within 2.3e-13 kJ/mol/nm (double), 1.8e-4 (mixed) | 1e-9, 1e-3 of the largest force |
| An update and its inverse | The energy of the state before | Equal (CPU, double and mixed) | 1e-9, 1e-4 relative |
| Tunables at the model's values, 20 steps | The model without tunables | Potential within 3e-12 (double) and 9e-11 (mixed) relative on the CPU: the table of the rule replaces Amber's, within 2e-7 | 1e-7, 1e-5 relative |
| Energy at new charges, σ, ε, `a`, and spring constants | OpenMM 8.6.1 (Reference) and NumPy, PME β = 2/nm, grid 72 | 1.3e-3 kJ/mol at the new values, 4.6e-4 at the old; the change, -102.03 kJ/mol, within 8.6e-4 | 2e-3, 1e-3 kJ/mol (B-splines of order 4 against 5) |
| Correction for the dispersion at new σ and ε | $-\tfrac{2\pi}{3}N^2\langle C_6\rangle/(Vr_c^3)$ in NumPy | Within 1.0e-14 relative | 1e-9 |
| $\partial U/\partial l$ of the pair term by central differences ($h = 10^{-5}$ nm, two updates) | `observe` of `mdir run` at the same positions, with tail and shift estimate (D189, D209, D210) | 172.744550 against 172.744542 kcal/mol/Å, 4.7e-8 relative | 1e-6 |

Refusals: 18 of declarations (including an NBFIX of `[ nonbond_params ]`
in a GROMACS topology, refused for per-type σ and naming CT and OW), 6 of
updates, each changing neither values nor version; versions in the history,
the states, and the energy file (`tunables_version` 0, 0, 0, 1, 2 at steps
0 to 20 with updates at 10 and 15); the program goes stale when the
declarations change; programs without tunables emit no table of them.

## Performance

See the PR for the measurements on the Amber suite: the default path
(nothing declared) emits the same module and lowered IR as main for every
suite system, and its rate is that of main.
