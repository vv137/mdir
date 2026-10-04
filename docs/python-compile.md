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
