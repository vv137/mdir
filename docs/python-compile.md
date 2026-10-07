# Python compilation interface (D192, D193)

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
The keyword `cache=False` lowers without the compile cache, and the
program's simulations inherit it (D[compile-cache-controls],
[compile-cache.md](compile-cache.md#controls-from-python)).

Typed enums select target, precision, integrator, ensemble, electrostatics,
truncation and dispersion. Mutable properties on all compile inputs advance
versions. `program.stale` and `program.check_current()` compare those versions;
the latter raises `StaleProgramError` until explicitly recompiled.
Programs retain input owners. Numeric array getters return read-only NumPy
copies: call `.copy()`, edit the copy and assign it back to commit a change.
Cell and custom-term objects are also copied; assign edited objects back to
their owner. Imported topology is owned and read-only at this first binding
boundary.

`InputError`, `UnsupportedError`, and `CompileError` preserve native error
messages; lowering errors retain MLIR locations and diagnostics. Existing
CLI control-file keys, formats and overwrite behavior are unchanged.
The temporary `Schedule` retains D191's fixed schedule; it is not `run(n)`.
A persistent `mdir.Simulation` ([python-segments.md](python-segments.md))
ignores it and runs the steps that `run(n)` asks for.
Declared tunable runtime buffers followed in D213
([python-tunable.md](python-tunable.md)): `System.tunables` is a compile
input like the others, and `Program.plan["tunables"]` lists them.

## Build and validation

Enable `MDIR_ENABLE_PYTHON=ON` with Python 3.10–3.13, pybind11 3.0.1 and
NumPy >=1.23 installed. Configuration checks the selected interpreter can
import a supported NumPy; module import checks it again. CLI-only builds
require none of these Python dependencies. CMake places the importable
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
| System | `periodic`, `cutoff`, `pairlist_distance`, `switch_distance`, `truncation`, `electrostatics`, `coulomb_modifier`, `dispersion`, `pme_alpha`, `pme_tolerance`, `pme_spacing`, `pme_grid`, `pme_order`, `rigid_hydrogen_bonds`, `rigid_water`, `flexible_water`, `water_residues`, `pair_terms`, `tuple_terms`, `restraints`, `restraint_reference`, `tunables` (D213); read-only `particle_count`, `charges`, `particle_types`, `type_names`, `atom_names`, `residue_names`, `residue_indices` |
| InitialState | `positions`, `velocities`, `cell` |
| Integrator | `method`, `timestep`, `minimize`, `minimize_step` |
| Ensemble | `kind`, `temperature`, `tau_t`, `pressure`, `tau_p`, `compressibility`, `coupling_period`, `com_period`, `seed` |
| Execution | `target`, `precision`, `device`, `threads`, `deterministic`, `reorder`, `fast_math` |
| Schedule | `steps`, `energy_period` |

Positions and velocities are native float64 NumPy arrays of shape $(N, 3)$
in input order; absent velocities have shape $(0, 3)$. Inputs must export
buffers or CPU DLPack and have C-contiguous storage and native dtype.
Flat coordinates, lists, float32, nonnative byte order, nonfinite values,
strided arrays and wrong particle counts raise property-specific `InputError`.
Loaded states retain their particle count. A fresh InitialState establishes
its count on its first nonempty coordinate or velocity assignment.

`Cell` diagonal/tilt properties use float64 arrays of shape $(3,)$ and
read-only `vectors` has shape $(3, 3)$ in the reduced lower-triangular
convention. Assign an edited cell back to its state to commit the change.
Every returned numeric array is an independent read-only NumPy copy.

`PairTerm` retains `name`, `expression`, `constants` (name/value pairs) and
`groups` (selection masks). `TupleTerm` has `name`, `expression`, `arity`,
`particles` (shape $(n, \mathrm{arity})$, zero-based int64 IDs), and
`parameters` (ordered name/1-D float64 array pairs). Particle inputs accept
native int32 or int64. Expressions use nm, radians and kJ/mol as in the
native model. System owns copies on assignment. To edit a nested term,
obtain `system.tuple_terms`, copy and edit an array, assign it to the term,
then assign the entire collection back. Failed assignments change neither
values nor versions. See [python-arrays.md](python-arrays.md).

CPU DLPack inputs are copied and non-CPU devices are refused. No framework
or gradient adapter is supplied; device views and leases remain later M2 work.

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

## Recorded validation

Python 3.10.12 with pybind11 3.0.1 passes 16 front-end parity cases per
CPU/GPU target (eight loading/ensemble cases in mixed and double).
Semantic IR and pipeline text match the CLI exactly (difference zero,
tolerance zero). Ownership, copied nested terms/cells, every compile input's
version, immutable inspection, error recovery and retained owners are checked.
The native compiler diagnostic test retains its IR location, recovers on a
subsequent compilation, and leaves no reproducer file.

CLI-only compilation passes with Python and CUDA disabled. A CUDA-disabled
extension passes the CPU checks and refuses GPU compilation with
`UnsupportedError`. The installed module passes the CPU checks outside
source and build trees with GPUs hidden. The extension has no CUDA driver
or MDIR runtime dependency; runtime loading is deferred.

Independent finite differences of the existing tabulated-term energies
(`test/Driver/Inputs/check_functions.py`) provide a numerical regression of
the unchanged CLI pipeline. The reference maximum force magnitude is
4.51391065 kcal/mol/Å. These checks exercise existing CLI kernels, not a
Python simulation, and do not extend the initial Python physics subset.

| Target and precision | Maximum component force difference (kcal/mol/Å) | Tolerance (kcal/mol/Å) |
|---|---:|---:|
| CPU double | $1.78911164\times10^{-9}$ | $4.51391065\times10^{-6}$ |
| GPU double | $1.78911197\times10^{-9}$ | $4.51391065\times10^{-6}$ |
| CPU mixed | $7.88052008\times10^{-6}$ | $9.02782129\times10^{-5}$ |
| GPU mixed | $1.26336992\times10^{-5}$ | $9.02782129\times10^{-5}$ |

The CLI pipeline generator is a verbatim extraction from main. No runtime,
step schedule or device kernels change, so no per-step timing or device
sanitizer run is added. Python execution timing remains a segment-API gate.

The complete default suite, run sequentially under the GPU 1 lock after
checking the device, passes 263 tests with six optional tests unsupported
and zero failures (269 discovered, 2045.79 s). Issues #22 and #26 are closed;
their `not-numbers-gpu.test` and `free-energy-reorder-gpu.test` regressions
pass in this single suite run, with no failures or retries.

D193 repeats the CPU/GPU mixed/double parity matrix with strict
host arrays: all 32 case/target/precision combinations match native loader
array bytes (0 differing bytes, tolerance 0) and CLI semantic IR/pipelines
exactly. The full local suite passes with 265 passed, 6 unsupported and
0 failures. See [host-array validation](python-arrays.md#validation-environment)
for the dependency, ownership and malformed-input checks.
