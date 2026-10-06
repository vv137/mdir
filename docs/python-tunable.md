# Tunable parameters of a Python model (D[python-tunable])

Issue #130, the M2a gate that D192 left open and the base of M2b
(differentiable simulation, [roadmap](roadmap.md), Section 6.1; D195).
Status: design of the draft PR; the measurements follow the
implementation.

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
explicit exception. This is a question to the maintainer on the PR.

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
  After an update the simulation evaluates them anew at its state (a call of
  the entry with no steps, which redoes the evaluation of the start without
  the half kick of leapfrog), so that `state().forces` is that of the new
  values. With velocity Verlet, `state().energies` then holds the energies
  of the state at the new values (the row of the start); with leapfrog,
  whose velocities are half a step behind, it is `None`. Neighbor structures
  are built anew at every part anyway.
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
site takes; `values` of the wrong shape or not finite. A pair term's
constant whose term is left out of the correction for the dispersion stays
left out after an update; one whose tail diverges at the new values is an
update refused.

## Validation plan

- Update against recompile: a simulation updated during a run against one
  compiled with `values=` from the state at the update, velocity Verlet,
  NVE: bit for bit on the CPU and on a GPU (deterministic mode), mixed and
  double. Against a program without tunables whose model has the new values
  (the edited table): within the term's tolerance.
- Oracle: OpenMM 8.6.1 energies at the new charges and per-type
  Lennard-Jones on the dipeptide in water; NumPy for a pair term's constant.
- Derivative: central differences of $U$ in a tunable constant of a pair
  term, by two updates at fixed positions, against the existing scalar
  derivative of `observe` (D189) from `mdir run` at the same positions.
- Refusals and versions: every refusal above; versions across updates, runs,
  reporters, and states.
- Speed: the default path against main on the Amber suite on GPU 0, and the
  cost with tunable charges and per-type Lennard-Jones (and the cost of an
  update).
