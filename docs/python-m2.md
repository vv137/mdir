# Python API preparation (M2)

Status: implementation plan with maintainer rulings, not implemented functionality.
D169 fixes the milestone's scope; D[m2-python-plan] records this preparation.
The work item is [#46](https://github.com/vv137/mdir/issues/46).
The maintainer approved the contracts in Section 6 on 2026-10-04,
[relayed in PR #47](https://github.com/vv137/mdir/pull/47#issuecomment-5975221457).
Remaining implementation details are specified in each implementation PR.
The sequence describes prospective PRs, not assigned work.

## 1. Scope and acceptance

M2 gives Python and the control file one validated model and one IR builder.
It includes Amber and GROMACS loading; physics separated from execution;
explicit compilation and stale detection; tunable parameters; persistent
runs and reporters; host arrays and device DLPack views; typed errors;
and the driver's H5MD checkpoint. It needs an installed Python package
and an example following minimization, NVT equilibration, NPT with
restraints, then production.

An OpenMM System importer follows later, as Section 6 of the roadmap says.
Learned potentials and distributed execution belong to M3 and M4 (D169).
M2 exposes an explicitly documented initial subset of classical terms.
The first model PR must enumerate that subset and its CPU/GPU support;
everything outside it is refused with a typed unsupported error.
Completion requires a manylinux_2_28 pip wheel for Python 3.10–3.13,
the validation gates below and a local suite pass. Preparation does not
close M2 or any implementation item.

## 2. Existing code and prerequisites

Initial inventory: main at `5dc0c94`, read on 2026-10-04. The preparation
was rebased on release 0.1.0 (`78acb94`); implementation must account for
the merged free-energy work (D176) and portable binary packaging (D177).

| Code | Reusable part | Work required for embedding |
|---|---|---|
| `include/mdir/Driver/Topology.h`, `lib/Driver/Amber.cpp`, `Gromacs.cpp` | Topology/coordinate readers and GROMACS includes/defines | Return loaded data without selecting a run, drawing velocities, or compiling |
| `include/mdir/Driver/Control.h`, `lib/Driver/Control.cpp`, `System.cpp` | Options, diagnostics, model construction and unit conversions | Separate file parsing from validation of an in-memory model |
| `include/mdir/Driver/System.h` | Particle arrays, topology, cell and restart state | Separate model and mutable state; the current System mixes them |
| `lib/Driver/Builder.cpp`, `include/mdir/Driver/Builder.h` | Semantic IR and metadata for fields, tables, tuples, precision and ordering | Separate model/step construction from the fixed CLI schedule |
| `tools/mdir/Run.cpp` | Pipeline, diagnostics, runtime loading, JIT and buffer setup | Extract reusable compilation and execution ownership; retain CLI policy in the command layer |
| `lib/Driver/Output.cpp`, `include/mdir/Driver/Output.h` | Writers and input-order gathering | Replace the global current-output pointer on the embedded path and return errors/stop status instead of exiting |
| `runtime/mdrt.c`, `runtime/mdrt_cuda.c` | Numerical helpers, random draws, streams and allocation cache | Audit global mutable state and abort/exit paths; isolate simulation state and pin exported allocations |
| `lib/Driver/Fingerprint.cpp`, `Checkpoint.cpp` | D172 provenance and D173 durable, checked format-1 checkpoint | Fingerprinting reopens TOML; add an in-memory path that records explicit Python options and model/plan hash entries |
| `CMakeLists.txt`, `tools/mdir/CMakeLists.txt`, `packaging/` | Version and installed runtime discovery (D174) | Discover runtime libraries relative to the package rather than the Python executable |

Builder's schedule uses fixed counts derived from `numSteps`, output
periods, and coupling periods. Repeated calls to that complete entry are
insufficient for arbitrary `run(n)` boundaries. A reusable segment entry
must accept a count and absolute step, retain buffers and coupling phase,
and execute remainders exactly once. Continuity includes random sequence,
carried forces, bath, thermostat chain, barostat state, time, and ordering.
Spatial sorting currently happens at run/segment boundaries; a new host
boundary must not silently alter coupling timing.

`mgpuMemFree` returns allocations to a size-based cache. Merely keeping a
Python object alive does not prevent their reuse. The CUDA runtime also
keeps a shared context and streams. Releasing the GIL does not make those
globals or the output pointer thread-safe. Persistent ownership and
embedded error handling are prerequisites, not just binding details.

## 3. Public contract and implementation requirements

`load_amber` and `load_gromacs` return an owned loaded-data object containing
topology, parameters, positions, optional velocities and cell in MD units,
with no CUDA initialization or invented run. System and initial state are
built separately from that object. Loaded data outlive the files; record
included topology contents for provenance.

System contains terms, exclusions, constraints, cutoff, PME, custom
expressions and parameter declarations. Integrator and ensemble are
separate objects. Execution uses typed target, device, precision, threads,
and deterministic options. Host numbers use nm, ps, kJ/mol, bar, amu and K.
Convert at the existing control-file boundary, whose lengths, energies and
pressures use different units. One C++ validation path checks dimensions,
shapes, finite values, particle identity and supported combinations.

Explicit `mdir.compile(...)` returns an immutable program with inspectable
IR and resolved plan information. Compilation does not create or back up
report files. Structural model changes make dependent programs stale;
the next run fails until explicitly recompiled. Define tracking for nested
terms and arrays so writes cannot bypass this check. A simulation owns its
buffers, step/time, runtime resources, output state and program lifetime.

Declared tunable coefficients use runtime buffers without recompilation.
Updates invalidate dependent forces and intermediate results. Reject
structural tuning (particle count, constraints, grid, cutoff or expression
structure) as though it were a scalar update. Checkpoint provenance must
include tunable declarations and values.

`sim.run(n)` advances n additional steps and ends segments at reporter
boundaries or the remaining count. Reporters due together share requested
state. C++ execution and built-in writers release the GIL; Python callbacks
run after completion with it held. Callback failures return to Python with
the completed step recorded. Poll stops between segments and retain the
CLI's D131 exit-status policy in the CLI. Use a typed MDIR reporter request
following OpenMM's scheduling model; final reports occur only when due.
Without reporters, bound segment lengths to keep stop latency at most
about 1 s on the Amber suite, measure it, and document the chosen length.
The exact stop return convention belongs to the segment implementation PR.

Refuse overlapping operations on one simulation. Multiple simulations
in one process must have independent state when run serially. Audit global
output, RNG/coupling helpers, CUDA streams and the allocator before
promising broader concurrency.

`state()` returns owned host copies in input order. Specify shapes, dtypes,
periodic cell convention, velocity time offset and available quantities.
Forces, energies and virials must be current, evaluated through the shared
program when necessary, or explicitly unavailable. Snapshots survive the
next run and simulation destruction.

`view()` exposes execution order and matching input particle IDs with the
actual dtype: mixed-mode forces are f32 and state is f64. Leases block
conflicting runs and mutations while views are alive. Define logical
validity separately from storage lifetime and outstanding consumer work.

The [DLPack Python specification](https://dmlc.github.io/dlpack/latest/python_spec.html)
requires producer-owned storage and coordination with the consumer stream.
Implement `__dlpack__` and `__dlpack_device__`, negotiate supported protocol
versions, retain allocation ownership through the managed tensor deleter,
and document device identity under `CUDA_VISIBLE_DEVICES`. Test a different
consumer stream and exactly-once release. The first design needs only the
Python exchange protocol.

An exported tensor cannot be revoked through a Python validity flag.
The initial interface is read-only; borrowing pins storage and blocks
conflicting runs or mutations while consumer aliases remain live. Release must account for
all managed tensor owners and outstanding consumer work before storage is
recycled. Tracked writable borrows follow later within M2; M2 is incomplete
without them. Writable borrowing needs an explicit completion/commit operation
that advances changed field versions (P16) and invalidates dependent forces,
neighbors and derived fields. Untracked external writes cannot be detected
automatically. Choose a policy for consumers that ignore read-only intent.

Map structured failures to input, compile, unsupported and simulation
exceptions, preserving file/line/term, IR location, feature/target, or
step/particle/quantity. Python exceptions cannot unwind through the C
runtime ABI: record failures and stop the segment safely. Distinguish
recoverable input/callback errors from a poisoned CUDA execution context.

Reuse H5MD format 1, integrity checks, `.prev` rotation and durable
replacement. Retain D172's distinction between same-run continuation and
starting a stage with changed physics. Same-mode deterministic continuation
must be exact; target/precision changes are portable but need not preserve
bits. Missing HDF5 raises an unsupported error. Python fingerprints record
only explicitly given options, matching D172's written-key policy. Model
and plan hashes are additional entries in format 1, not a new format.
Claim CLI/Python continuation only after a test in each direction passes.
Post-0.1.0 changes to control-file keys or checkpoint format require a
Changed changelog entry; checkpoint format changes also require migration.

## 4. Proposed implementation sequence

Each row is a prospective issue/PR with its own decision label and design
draft. Changes to Builder, Control and runtime should proceed sequentially.

| Order | Proposed deliverable | Gate |
|---|---|---|
| 1 | `D[python-model]`: in-memory validation, loaded data, System/integrator/ensemble/Execution and shared model-to-IR path | File and object models produce equivalent physics/IR and consistent validation failures |
| 2 | `D[python-compile]`: compilation service and optional native extension, immutable program, plan inspection, stale detection | CPU/GPU mixed/double compilation, structured diagnostics, repeated destruction, no output side effects |
| 3 | `D[python-segments]`: persistent simulation and arbitrary counts, embedded status/errors, runtime ownership | Segmented/uninterrupted agreement, retained coupling phase, stop/error survival, independent simulation state |
| 4 | `D[python-reporters]`: scheduling, C++ writers, Python callbacks, host snapshots | Nondividing/coincident periods, final remainder, callback errors, GIL behavior, file lifecycle, input order |
| 5 | `D[python-checkpoints]`: shared in-memory provenance and cross-front-end continuation | Both checkpoint directions, stage changes, corruption rejection, tunable restoration and output continuation |
| 6 | `D[python-dlpack]`: read-only leases, IDs and stream handoff, followed by tracked writable borrows within M2 | Alias lifetime/completion, allocator reuse, dtype/order and version invalidation; read-only support alone does not close M2 |
| 7 | `D[python-package]`: Python 3.10–3.13 manylinux_2_28 pip wheel, reference and four-stage tutorial | Import/run outside the source/build tree, runtime discovery, CPU-only operation, suite and performance report; conda is deferred |

Use an optional pybind11 extension. Classical CLI builds require neither
Python nor NumPy. Support Python 3.10–3.13; a manylinux_2_28 pip wheel built
on D177's packaging baseline gates M2. Conda follows later. Pin the binding
dependency and specify layout in the package implementation PR.
PR #45's portability work and PR #10's free-energy work have merged as D177
and D176. Check other open changes, including tabulated-value loading,
and rebase before extracting overlapping model/Builder interfaces.

## 5. Validation and performance gates

Front-end parity checks integration; it is not an independent physics
oracle. Each numerical implementation PR records quantity, reference value,
MDIR value, difference and tolerance. Use existing independently validated
term tolerances and measured precision behavior; this plan does not invent
unmeasured thresholds.

| Area | Required evidence |
|---|---|
| Shared model | Small Amber/GROMACS fixtures, custom terms, constraints, defaults, units, triclinic cells and unsupported features; existing OpenMM/NumPy energy and force references |
| Compilation/tuning | Equal semantic IR; resolved precision/target; nested and array stale tracking; coefficient updates without recompilation checked against an independent expression and finite differences |
| Segments | CPU/GPU mixed/double; NVE, stochastic dynamics, periodic coupling, restraints and NPT; partitions such as 1+7+13 versus 21, including boundaries inside coupling periods; compare full checkpoint state and absolute random sequence |
| Exactness | Same-target/precision bitwise continuation where ordering and deterministic contracts guarantee it; explicit tolerances when sorting or reduction order differs, rather than a blanket bitwise claim |
| Reporters/state | Periods 7 and 11 over 25 steps, repeated runs and restart; due steps, counts, shared requests, copy independence, IDs, velocity offset; backups, append/part filenames; four-stage tutorial |
| Errors/resources | Bad input, compile failure, unsupported target, nonfinite state, callback/write errors and stop leave Python alive; cleanup, serial simulation independence, rejected overlapping operations; measured stop latency about 1 s or less without reporters |
| Checkpoints | CLI-to-Python and Python-to-CLI; same run versus changed stage; topology, masses, restraints, thermostat/barostat, tunables, fingerprints and integrity; missing HDF5; one format |
| DLPack | Independent CPU/CUDA consumer; pointer sharing, IDs/dtypes, stream handoff, retained aliases, destruction order, outstanding consumer work, blocked conflicting runs, writes and forced allocation reuse |
| Installation | CLI-only build, pip wheel on Python 3.10–3.13 and manylinux_2_28, import outside the tree, common version source, package-relative libraries, CPU import without CUDA initialization and packaged GPU execution |

Run required local suites per implementation change, including sanitizer
for changed GPU execution and short Amber runs for schedule/kernel changes.
GPU tests use `*-gpu.test` and `REQUIRES: cuda`. Record repeats/failures for
intermittent tests. Existing issue #22 and reordered intermediate-field
issue #26 need an explicit validation status, not unrelated fixes here.

Measure main and Python on the same Amber systems under the GPU 0 lock
after checking usage. Separate compile time from ms/step or ns/day;
distinguish long segments, host-copy reporters and device consumers.
State every slowdown and assess D114's target against pmemd.cuda. GPU
tests use the GPU 1 lock. This design-only PR has no numerical CPU/GPU
results or timing measurements.

## 6. Adopted maintainer rulings and remaining details

The maintainer approved these answers on 2026-10-04 in the linked PR comment:

1. Owned loaded-data objects; System and initial state built separately;
   MD units and typed execution; a documented initial classical subset with
   typed unsupported errors for the rest.
2. Optional pybind11; Python 3.10–3.13; manylinux_2_28 pip wheel on the D177
   baseline required for M2; conda deferred.
3. Refuse overlapping operations on one simulation; independent serial
   simulations; audit shared runtime state before stronger promises.
4. Typed requests following the
   [OpenMM scheduling model](https://docs.openmm.org/latest/api-python/generated/openmm.app.simulation.Simulation.html),
   final reports only when due; bounded segments without reporters with
   stop latency at most about 1 s on the Amber suite.
5. Execution-order views plus input IDs; initial read-only leases block
   conflicting operations; tracked writable borrows are a later required
   step within M2.
6. Fingerprint explicit Python options; model/plan hashes as additional
   format-1 entries; a continuation test in each direction before claiming
   CLI/Python interoperability.

The seven implementation PRs proceed one at a time through the label
workflow. They specify the initial term list, exact signatures/mutation
tracking, stop return convention, measured segment length, consumer
completion and write-commit mechanism, hash comparison details, and package
layout. These details do not reopen the approved choices above. Released
control-key or checkpoint-format changes carry a Changed changelog entry,
and checkpoint changes include migration when the format changes.
