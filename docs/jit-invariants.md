# JIT memory and unwind ownership (D[jit-invariants])

Issue #89 supersedes D196's section-placement lifetime guarantee with checks
at the final host object boundary. No control-file
keys, file formats, defaults, or overwrite behavior change.

The simulation owns an ORC object memory manager. It reserves a
contiguous allocation per object, inspects allocated executable sections and
relocated exception-frame records before registration, and rejects layouts
whose frame ranges leave their executable sections or whose bounding code
interval overlaps another live object. Section names alone are not evidence.
Late ORC initialization and deinitialization functions pass through this same
boundary. Registration is deferred until validation and memory finalization
succeed. Destruction deregisters frames before the allocation is released,
including failed construction. Creation, initialization, execution, and
teardown share the process runtime mutex.

The owned lifecycle is allocation/linking, validation, frame registration,
initialization, execution, deinitialization, deregistration, and release.
Validation failures publish no entry pointer. LLVM owns relocation correctness,
constructor/destructor synthesis, and object removal; libgcc owns exception
search and the implementation of its registration index. These assumptions
are checked with negative boundary tests and seeded lifecycle tests, rather
than claimed as formal verification of either dependency.

Validation covers CPU/GPU, mixed/double, import orders, retained live
engines, construction errors, and delayed exceptions, with ASLR retained.
A separate host exception sanitizer baseline distinguishes interceptor
failures from JIT failures. Numerical results are compared with an independent
analytic pair force and the existing segment regressions.

## Front-end boundary and follow-up

The CLI remains on MLIR's `ExecutionEngine` in this PR. `mdir run` owns one
engine in its own process, so it does not exercise the retained multi-engine
lifetimes that motivated #89. Moving it here would also require accounting
for its entry ABI, symbol registration, runtime-library protocol, diagnostics,
and object dumping, broadening this fix beyond simulation ownership.

The intended direction is to move the CLI to the owned engine in a separate
change, tracked by [#99](https://github.com/vv137/mdir/issues/99), so both front
ends share library loading, symbols, initialization, and final-object checks.
Both already share `compiler::createHostMachine()` from D197. Until migration,
CLI/Python comparisons remain necessary to detect drift. Separate JIT paths
are a possible source of differences, not an established explanation for #97.

D[jit-invariants] amends D196: section placement is superseded as the lifetime
guarantee by contiguous owned allocation and validation of actual code and
unwind ranges before registration. The old MLIR transformer in
`Simulation.cpp` is removed. Its section assignment remains in
`JITEngine.cpp` for locality (`.ltext` for x86-64's large model, `.text`
otherwise); section names and that hint provide no correctness proof. Late
ORC-generated functions are checked regardless of their section placement.

## Enforced transitions and dependency boundary

Each object starts in `Linking`. RuntimeDyld allocates and relocates sections,
then requests frame registration. MDIR queues that request without calling the
unwinder. `finalizeMemory` checks actual allocated addresses, requires the
executable hull inside one owned mapping, requires exactly one unwind table
inside allocated data, parses its relocated CIE/FDE records with LLVM's DWARF
parser, and requires every FDE interval inside an executable section. Under the
process frame-registry mutex it rejects overlap with another live executable
hull. It applies permissions and only then registers the table and enters
`Registered`. Failure enters `Rejected` and registers nothing. A second
finalization or registration after release is rejected. Cleanup is idempotent:
it deregisters, removes the registry entry, and enters `Released`. The mapper
refuses storage release while the registration flag is still set.

The memory manager owns its mapper by base-class ordering: deregistration in
the derived destructor precedes SectionMemoryManager's release callbacks,
and the mapper outlives those callbacks. ORC removal also calls deregistration;
failed linking destroys the manager even when no resource was published.
The entry pointer is exposed only after initialization and successful lookup.
Initialization and symbol-resolution failures are returned as owned error
strings, which survive destruction of the ORC session.

Creation, continuation compilation, entry execution, and destruction take the
same process mutex. Polling Python happens between parts after releasing it.
A caller must keep a simulation alive while one of its methods runs; native
callers cannot destroy an object concurrently with its own active method.
The Python binding retains the object for its method call. Independent
simulations can remain live together. Their runtime operations serialize.

LLVM owns section sizing, overflow-safe reservation, relocation and machine
code correctness, frame encoding/decoding, permissions, constructor/destructor
synthesis and invocation, and object removal. MDIR verifies the resulting
layout rather than depending on section names or on a reservation promise.
LLVM's DWARF parser and libgcc's registration routines must agree on emitted
ELF frame records; MDIR does not validate arbitrary hostile object files.
The OS must keep simultaneously owned mappings disjoint. Contiguous ownership
also excludes unrelated libraries from the bounding interval, beyond the MDIR
registry. libgcc owns its index and deregistration implementation. CUDA owns
kernel module initialization and runtime cleanup. The supported host format
is ELF x86-64 or AArch64; other formats fail closed. Validation on this machine
covers x86-64; AArch64 is an unexercised dependency configuration.

The wrapper uses LLVM 23.1.2 ORC LLJIT and RuntimeDyld public interfaces.
Its packed entry matches the ABI documented by MLIR ExecutionEngine; the
implementation is independent and creates only the simulation's void wrapper.
No LLVM installation files are modified.

## Reproduction and sanitizer configuration

`test/Driver/jit-memory.test` exercises the production memory manager directly,
including malformed relocated records, out-of-section FDEs, noncontiguous
code allocations, overlap with a retained object's interval, duplicate
finalization, registration after release, and release before deregistration.
The native engine fixture includes late ORC initialization/deinitialization
functions under x86-64 small and large code models and a missing-symbol
initialization failure followed by a C++ throw after destruction. Successful
engines also call a native thrower through the packed JIT entry and catch the
exception in the host, proving live unwinding works in both code models.

`python-simulation-lifetime{,-gpu}.test` each run fresh processes in NumPy-first,
MDIR-first, and runtime-first orders (both runtimes first on the GPU). Each
process runs the original 24 retained-engine cycles, then seeds 89, 196, and
20261006 once each for 32 operations, retaining up to five simulations with
both first and continued engines. Six additional lifetimes run on three host
threads. Failures print the seed, import order, and operation sequence.
`validation.txt` under each test's output directory records the independent
force comparison and the seed outcomes. ASLR is retained.

A sanitizer exception baseline can be built from
`test/Driver/Inputs/host_exception_baseline.cpp` with GCC 11,
`-fsanitize=address,undefined -fno-omit-frame-pointer`; it catches 32 throws
without a JIT or GPU. The native JIT boundary executable is also run in the
instrumented build with `ASAN_OPTIONS=detect_leaks=0:allow_user_poisoning=0`.
The uninstrumented LLVM libraries require disabled user poisoning.
`-fno-sanitize=vptr` avoids checks requiring RTTI from a no-RTTI LLVM build.
Neither test disables ASLR. Instrumented generated machine code and LLVM's
own allocations are outside the host sanitizer's coverage.

The Python sanitizer configuration is tracked separately in #98: GCC 11's
instrumented extension references unavailable LLVM RTTI and fails at import,
before JIT execution. Preloading libasan and libstdc++ together changes the
prior exception-interceptor recursion into a clean import diagnostic, but
does not make this configuration a usable Python sanitizer baseline. Release
Python lifecycle tests and native sanitizer tests provide distinct evidence;
Python sanitizer lifecycle success is not claimed.

## Validation results

The release build on x86-64, LLVM 23.1.2, passed the complete GPU-enabled
suite: **272 passed, 7 unsupported, 0 failed**, 638.18 s under the GPU 1 lock.
The unsupported tests are the six opt-in device sanitizer tests and the
external Amber scale suite. No device code changes, so compute-sanitizer
was not run. The strengthened live-unwinding native test subsequently passed
both normally and under host ASan/UBSan.

In the complete lifecycle matrix, each seed (89, 196, 20261006) was run once
per target and import order: 18 seed sequences, 576 seeded operations,
0 failures. Both precisions are retained in each process. Each process also
passes 24 interleaved lifetimes and six threaded lifetimes. ASLR is enabled.
The independent oracle is the analytic Lennard-Jones pair force, evaluated
in NumPy at the returned coordinates with $\sigma=0.3$ nm and
$\epsilon=0.1$ kJ/mol, below the switching interval. The maximum absolute
force difference is compared with $256\varepsilon_{64}\lvert F\rvert$
in double and $64\varepsilon_{32}\lvert F\rvert$ in mixed; these budgets
cover arithmetic and parameter/coordinate conversion rounding for this pair.
The floating-point constants here denote machine epsilon, not the potential's
energy parameter.

| Target | Precision | Reference x-force (kJ/mol/nm) | Maximum difference | Tolerance |
|---|---|---|---|---|
| CPU | double | 0.6877722941605 | 1.110223e-16 | 3.909533e-14 |
| CPU | mixed | 0.6877722941590 | 9.906765e-8 | 5.247286e-6 |
| GPU | double | 0.6877722941605 | 1.110223e-16 | 3.909533e-14 |
| GPU | mixed | 0.6877722941582 | 9.906841e-8 | 5.247286e-6 |

The original focused GPU validation had one infrastructure failure in two
tests: the CLI was launched while a concurrent relink replaced its executable,
producing permission denied. The lifecycle test passed. Builds were completed
before running the full suite above. This is not evidence of intermittent
numerical behavior and was not retried until green. Sanitizer setup failures
(interceptor recursion, unsupported RTTI linkage, and missing LLVM poisoning
options) are recorded separately from the passing native sanitizer run; #98
tracks the Python configuration.

Performance was measured on idle GPU 0 under its lock, one RTX 3090 at
300 W, against main 143c650, with the same two-particle input, 200,000 steps
per precision after compilation and warmup. One long run per cell:

| Precision | Main ms/step | This change ms/step | Change | Main compile/warmup (s) | This change (s) |
|---|---|---|---|---|---|
| double | 0.020776762 | 0.021375896 | +2.9% | 1.947048 | 1.946182 |
| mixed | 0.023662715 | 0.022072166 | -6.7% | 1.386909 | 1.633006 |

These launch-bound pair measurements are single runs, not a statistical
speedup claim or an Amber suite performance claim. Device kernels and the
integration schedule are unchanged. AArch64 and Python sanitizer lifecycle
validation remain unexercised configurations; the dependency guarantees are
limited as stated above.
