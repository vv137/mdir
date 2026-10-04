# Python compilation interface (D[python-compile])

The next M2 contribution after D191, tracked in issue #76, exposes the
owned native model and explicit compiler lowering. It is an optional build,
not a wheel or a persistent simulation API. M2 remains incomplete.

## Public contract

`mdir.load_amber`, `load_gromacs`, and `load_charmm` return owned loaded data.
`make_system()` and `make_state()` return independent objects in MD units.
`mdir.compile(system, state, integrator, ensemble, execution, schedule)`
validates and lowers the shared semantic IR through the CLI's pipeline.
The immutable returned `Program` exposes `ir`, `lowered_ir`, `pipeline`,
and a copied `plan` dictionary. No runtime libraries are loaded, no CUDA
execution context is initialized, and no report or reproducer is written.
JIT ownership, runtime device selection and execution follow with segments.
The logical device is recorded in the plan, not selected during lowering.

Typed enums select target, precision, integrator, ensemble, electrostatics,
truncation and dispersion. Mutable properties on all compile inputs advance
versions. `program.stale` and `program.check_current()` compare those versions;
the latter raises `StaleProgramError` until explicitly recompiled.
Programs retain input owners. Collection getters return copies: edit a
copy and assign it back to commit a change. This includes positions,
velocities, cell, custom terms, tuple parameters and members. Imported
topology is owned and read-only at this first binding boundary.

`InputError`, `UnsupportedError`, and `CompileError` preserve native error
messages; lowering errors retain MLIR locations and diagnostics. Existing
CLI control-file keys, formats and overwrite behavior are unchanged.
The temporary `Schedule` retains D191's fixed schedule; it is not `run(n)`.
Declared tunable runtime buffers are not implemented by this contribution.

## Build and validation

Enable `MDIR_ENABLE_PYTHON=ON` with Python 3.10–3.13 and pybind11 3.0.1
installed. CLI-only builds require neither. CMake places the importable
development module in `python/` under the build tree; set `PYTHONPATH` to
that directory. Installation places it in a configurable Python destination.
The manylinux_2_28 wheel remains the package gate of M2.

Validation checks file/object semantic IR and pipeline parity, CPU/GPU
mixed/double lowering, copied nested inputs, stale detection, repeated
compilation/destruction, typed failures and absence of output side effects.
The unchanged numerical builder retains its independent term-oracle suite;
this interface makes no Python simulation or performance claim.

## Configuration reference

All compile arguments are explicit and required. Construct `Integrator()`,
`Ensemble()`, `Execution()` and `Schedule()` for their native defaults;
obtain System and InitialState from the loaders. Public property names use
snake case; defaults and the supported physics subset are those of
[python-model.md](python-model.md) and `include/mdir/Driver/Model.h`.

| Object | Properties |
|---|---|
| System | `periodic`, `cutoff`, `pairlist_distance`, `switch_distance`, `truncation`, `electrostatics`, `dispersion`, `pme_alpha`, `pme_tolerance`, `pme_spacing`, `pme_grid`, `pme_order`, `rigid_hydrogen_bonds`, `rigid_water`, `flexible_water`, `water_residues`, `pair_terms`, `tuple_terms`; read-only `particle_count` |
| InitialState | `positions`, `velocities`, `cell` |
| Integrator | `method`, `timestep`, `minimize`, `minimize_step` |
| Ensemble | `kind`, `temperature`, `tau_t`, `pressure`, `tau_p`, `compressibility`, `coupling_period`, `com_period`, `seed` |
| Execution | `target`, `precision`, `device`, `threads`, `deterministic`, `reorder`, `fast_math` |
| Schedule | `steps`, `energy_period` |

Positions and velocities are flattened lists of $3N$ Python floats in input
order, with an empty velocity list denoting absent velocities. `Cell` has
three diagonal lengths and three tilts in nm, with read-only `vectors`
following the reduced lower-triangular convention. Its properties are
copies when obtained from a state: assign the edited cell back to the state.
No NumPy or framework tensor dependency is required.

`PairTerm` has `name`, `expression`, `constants` (name/value pairs) and
`groups` (selection masks). `TupleTerm` has `name`, `expression`, `arity`,
`particles` (flattened zero-based IDs), and `parameters` (name/list pairs).
Expressions use nm, radians and kJ/mol as in the native model. System owns
copies on assignment. To edit a nested term, obtain `system.tuple_terms`,
edit a term or its parameter copy, then assign the entire collection back.
Collection edits without reassignment affect only the returned copy.

The plan records target, precision, logical device, threads, determinism,
particle reordering, entry name, state/force dtypes and PME grid. It does
not claim runtime device resolution or executable JIT ownership. The
lowered IR embeds PTX on GPU targets; lowering needs the CUDA toolkit's
libdevice, found through `CUDA_ROOT` when set. Python compilation ignores
CLI debugging environment overrides (`MDIR_PIPELINE`, `MDIR_PRINT_AFTER`,
`MDIR_REPRODUCER`), so it neither silently changes the plan nor writes files.
The GIL remains held during compilation to prevent input mutation racing
with snapshot/version capture. Persistent execution will define GIL release.

CMake uses the documented pybind11
[package discovery and module helper](https://pybind11.readthedocs.io/en/stable/cmake/index.html).
Only Python 3.10 is validated in this contribution; 3.11–3.13 and portable
wheels remain installation gates.
