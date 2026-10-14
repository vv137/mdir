# JIT memory and unwind ownership (D199)

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

D199 amends D196: section placement is superseded as the lifetime
guarantee by contiguous owned allocation and validation of actual code and
unwind ranges before registration. The old MLIR transformer in
`Simulation.cpp` is removed. Its section assignment remains in
`JITEngine.cpp` for locality (`.ltext` for x86-64's large model, `.text`
otherwise); section names and that hint provide no correctness proof. Late
ORC-generated functions are checked regardless of their section placement.

## Cached host objects (D212)

The compile cache ([compile-cache.md](compile-cache.md)) is the
`llvm::ObjectCache` of the engine's compiler. It holds relocatable objects
before linking, and a hit replaces only code generation: the object goes
through the same allocation, linking, validation, registration, and release
as a generated one, in memory the engine owns. No linked code, mapping, or
frame is shared between engines or processes, and an entry is copied into
memory rather than mapped. `jit-memory.test` runs the checks of this
document without a cache, with a cold one, and with a warm one.

The PTX and cubins of the GPU modules (D214) are data: a
constant of the host module, which the CUDA driver loads into memory of its
own. Whether they were generated, read from the cache, or compiled by
`ptxas` changes only those bytes, before the host object is generated;
no host code comes from them, and the checks above are unchanged. The
lifetime tests run with the cubins and the cache of the suite.

## Activations that outlive a call (D215)

A Python simulation's entry does not return between parts
([python-segments.md](python-segments.md#resident-buffers)). It runs on a
stack that the simulation maps (64 MiB, committed as it is used, with a
page below it that is not mapped) and waits at the end of each part inside
`mdrtPartBoundary`, a function of the host that the code calls through its
C interface, while the host goes on on its own stack (`swapcontext`). The
frames of the activation therefore hold addresses of the engine's code and
data between calls of `run`. The engine, its object memory, and the
registration of its frames outlive the activation: the simulation ends its
activation before it destroys its engine, under the process runtime mutex.

An activation never returns and is never unwound. It ends at a boundary:
the simulation frees what the activation allocated, which the records of
the runtimes (`mdrtActivationClose`, `mdrtDeviceActivationClose`) and of
the memory of the host that its code takes list, and unmaps its stack.
Since the entry never returns, memory of the stack that it took in a loop
would never be given back: its depth at a boundary must not depend on the
steps taken. On the CPU `hoist-static-allocas` moves every allocation of a
constant size to the start of its function where the lowering ends (D117,
amended by #305, where the rows of the observed columns grew the stack by
32 bytes a step of energy); on a GPU the lowering releases the stack of
each iteration of a loop of the host. An allocation whose size is not a
constant is one that the pass cannot move; no program has one, and the
lowering of a program of frames, whose rows stand in its loop over frames
(#309), refuses a module that would keep one outside the first block of a
function (`hoist-static-allocas{refuse-rest=true}`). `python-stack-rows.test`
and `python-frames-loop.test` read the depth (`Simulation._stack_in_use`,
private) to the byte. The
generated code holds no other resource on its stack (no destructors, locks,
or registrations), and the two frames of the host on that stack, the start
of the activation and the boundary, hold none either; no exception crosses
the frames of the code. An activation may be resumed on a thread other
than the one it began on: the code keeps no address of thread-local
storage across a boundary (glibc's `swapcontext` does not switch the
thread pointer), and the runtime of the device makes its context current in
each of its calls. One activation runs at a time in the process, under the
run mutex; several may wait at their boundaries, one for each live
simulation, and their records keep their allocations apart.

`python-simulation-lifetime-*.test` retain up to five simulations that have
run more than one part, each with a live activation, and run six lifetimes
on three threads, whose activations wait side by side.
`python-resident-threads*.test` resume one activation on threads other than
the one it began on, with four threads of execution on the CPU, and give the
state of a run on one thread to the bit.

## What a simulation keeps across activations (D253)

An activation ends by freeing what it allocated (above), and the next one
begins from the state of the host. One thing outlives it, since #294: the
structures of the tuples of the topology, which the simulation keeps in a
store of the runtime (`runtime/mdrt_tuples.h`) from its first activation
to its end. The rest of this section is what holds for that store.

**What is kept.** For each tuple set and each order of the particles that
an activation asked for:

- the members of the set renumbered to the places of the particles
  (`md_exec.renumber`), on the host;
- the incidence structure built from them (`md_exec.build_incidence`), on
  the host for the CPU target and on the device for a GPU;
- on a GPU outside the deterministic mode, the copy of the renumbered
  members that the loops over tuples evaluated once read;
- and, to decide whether these may be taken again, a copy of both of their
  inputs: the members in the order of the files, as the entry was given
  them, and the identity of the particle at each place.

A program in the order of the input (that of a frame evaluator, or
`Execution.reorder = False`) has no identities to compare: its structures
are kept under the members alone.

**How much, and its bound.** The store has one entry for each pair of
members and kind of order (identities, or the order of the files) that an
activation has asked for, which is one for each tuple set of the program:
an order that differs from the kept one does not add an entry, it replaces
the kept one in its memory (I2). A second entry of a tuple set exists only
while the code of one activation holds two orders of it at once (the
anchors of the constraint groups, which the order is computed from before
the particles are permuted), and it is reused in the same way at the next
start. The number of entries therefore does not grow with the starts, the
steps, or the orders that a simulation passes through
(`python-memory*.test`, and the counts of `python-kept-tuples*.test`). The
bytes, on the three explicit systems of the Amber suite:

| System | Atoms | Incidence structures and the copy of the members (device on a GPU, host on the CPU), MB | Kept inputs and renumbered members (host), MB |
|---|---|---|---|
| JAC | 23,558 | 42.6 + 0.7 | 1.9 |
| Factor IX | 90,906 | 164.4 + 2.2 | 6.3 |
| Cellulose | 408,609 | 696.3 + 19.9 | 48.0 |

The first column is what an activation held before as its own, and gave
back to the pool of device blocks when it ended (the pool kept the blocks
for the next activation, so the device memory of the process is what it
was). The second is new: two copies of the members (as given, and
renumbered) and one of the identities for each tuple set. On a GPU the
host never holds a whole incidence structure: its rows are written a few
megabytes at a time and uploaded (`tuplesBuildOnDevice`), in 16 MiB of
the store that are taken once, where each was built whole in new memory
of the host before.

**What the comparison costs.** A start that takes its structures compares
the identities and the members of every tuple set with the kept copies: on
the CPU target 0.08 ms on JAC, 0.24 ms on Factor IX, and 1.7 ms on
Cellulose for the five sets together, against the 2 ms, 14 ms, and 315 ms
that building them took; on a GPU the identities are read from the device
once for each set as before, and the whole is 0.13, 0.43, and 2.1 ms (one
RTX PRO 6000) of starts of 2.0, 6.1, and 30 ms.

**I1. A kept structure is taken only where it is the one that would be
built.** The renumbered members and the incidence structure are functions
of the two inputs and of nothing else: not of the positions, the cell, the
values of the tunables, the step, or the precision. Where the code asks
for them, the runtime compares the identities that the program has just
computed (read from the device on a GPU) and the members that the entry
was given with the kept copies, element for element, and takes the kept
structure only if both are equal. Nothing is inferred from what the host
did since the last activation: no call of the simulation marks the store
valid or invalid, so no call can forget to. What invalidates a kept
structure is therefore exactly one of:

- the particles are in another order than the one it was built for (one
  identity differs): after steps, after a commit of a borrow that moves a
  particle into another cell of the order, in a simulation that continues
  a checkpoint;
- the members that the entry is given differ in one element from those it
  was built for, by whatever path they changed.

An update of the tunables, a commit of velocities, tunables, or a cell, an
evaluation, `gradient()`, and a checkpoint invalidate nothing unless they
change the order, and then by the comparison alone.

**I2. The code that runs never has its structures built over.** The
runtime counts what the code of the activation holds of an entry (taken
and not given back, `mdrtTuplesRelease`). A structure that is not found
is built in the memory of an entry of the same tuple set that is not held,
whose sizes are those needed (the width of a row does not depend on the
order), and otherwise in new memory. When an activation begins, the one
before it has ended at a boundary without giving anything back, and
nothing is held.

**I3. The kept memory is not the activation's.** It is allocated outside
the records of the activations (on a device, outside the pool of blocks
that `mdrtDeviceActivationClose` returns blocks to), so the end of an
activation leaves it, and it is freed when the simulation is destroyed,
after its last activation and before its engine. The code of the program
does not free it: where it would free a structure it gives it back to the
runtime, which frees only what it did not keep.

**I4. The kept structures are read, never written.** The loops over
tuples and the build of a neighbor structure read an incidence structure
and the members; no kernel and no loop of the host stores into them.
With `MDRT_CHECK_KEPT_TUPLES` set in the environment
([environment.md](environment.md#debugging)) the runtime checks I1 and I4
at every use: each structure that is taken is also built anew from the
inputs given and compared with the kept one (read back from the device),
and a difference ends the program with a message that names the structure.
`python-kept-tuples*.test` run with it.

**Which programs.** Every program takes the structures of its tuple sets
from the runtime: the pipeline sets the option `kept-tuples` of
`convert-md-exec-to-loops` and `convert-md-exec-to-gpu` for all of them.
A Python simulation and the simulation of a frame evaluator
(`Control::segments`) keep a store across their activations, as above.
`mdir run` opens a store of the process for its one call of the entry and
closes it when the entry returns (D[run-tuples-memory], #316): its program
orders the particles where the run begins and where each segment of
checkpoints begins, so each segment asks for its structures in a new
order. I1 and I2 hold as they do across activations, inside one call:

- the first segment begins in the order of the start, with no step
  between them, so its inputs compare equal and it takes every structure
  of the start; every later segment compares its identities, finds them
  different, and builds;
- the structures of the start are held by the code of the entry until it
  returns (they stand in the first block of the entry), and those of a
  segment are given back at the end of its iteration. A later segment
  therefore builds in the memory of the entry of the segment before, which
  is not held, and the number of entries does not grow with the segments:
  those of the start, which its code holds, and those that the segments
  pass on to one another. On a device these are the structures that the run held on
  `main` as well, where the start's stood until the end and each segment
  took the blocks of the one before from the pool; on the host the store
  adds the kept inputs and the renumbered members of the two entries and
  the 16 MiB in which a device's rows are written, where `main` built each
  structure whole in new memory of the host and freed it. The peak
  resident memory of a run on a GPU is therefore less: 418 to 361 MB on
  JAC, 506 to 420 MB on Factor IX, 1,112 to 786 MB on Cellulose, and
  2,581 to 1,495 MB on STMV (one RTX 4090, mixed precision).

With `MDRT_CHECK_KEPT_TUPLES` set, `mdir run` writes at the end of its log
how many structures were built and how many taken (`MDIR: the structures
of the tuples were built N times and taken as they were kept M times`);
without it the log is that of before (the maintainer's ruling on PR
#327). `run-kept-tuples.test` and its GPU twin check the counts of a run
of four segments under the variable, that a run without it prints no
counts and ends in the same state, and that the run equals, to the bit, the same run continued at every
checkpoint, whose processes build in new memory. The structure of tuples
that a program finds at run time (the triplets of a three-body term) is
not kept in any program. Code that runs without a store entered (a tool
that executes a module without opening one) gets structures of its own
from the same functions and frees them.

**Tests.** `python-kept-tuples.test` and `-gpu.test`: the store called as
the code calls it, against a renumbering and an incidence structure
written in NumPy, with one identity or one member changed; a kept
structure written over by the test, which ends the program under the
check; and a sequence of starts (evaluations, an update of the tunables,
steps that order the particles anew, a commit of a borrow, a checkpoint,
a run continued from it, and on the CPU a commit that moves a particle)
whose every state, energy, and derivative equals, to the bit, that of a
simulation that builds every structure anew (`Simulation._keep_tuples`,
private), in double and mixed precision, with the numbers of structures
taken and built at each start (`Simulation._kept_tuples`, private): a
start that should have built and took, or the reverse, fails the test.

## What an update of the tunables keeps (D[update-cost])

An update of the tunables (D213) gives the program of a simulation new
values: it puts the values into the model, builds the program from it by
the code of the compile, compares the text with the compiled one, and
begins an activation. Until #324 it copied the model of the compile at
every update, and what it built anew grew with the system and not with the
tunable: 143 ms for 408,609 atoms on the host of an RTX 4090, to change one
number, of which a sort that asks whether each tuple set is disjoint 108 ms
and the two copies of the model 11 ms; and 114 ms more on 23,558 atoms
where a pair term has its tail in the correction for the dispersion, for
quadratures that most of the pairs of classes share. Since D[update-cost]
a simulation keeps, from its first update to its end, the model that the
values are put into. The rest of this section is what holds for it.

**What is kept.** A copy of the control and the system of the compile with
a topology that nothing else holds, made at the first update. An update
writes the values into it in place (`model::applyTunablesInPlace`) and
builds the program from it. Everything of the program is built at every
update, as before: every array, every constant that the host adds, and
the whole text, which is compared with the compiled one as before (D213):
a value that would change the program is refused as before.

**U1. The model that the values are put into is the model that
`applyTunables` makes from the model of the compile.** The sites that an
update writes are fixed by the declarations, not by the values: every site
of every tunable, every pair of types that the combining rule or a tunable
of the table gives, the weights of the seeds, and the tails, which are
collected anew. Each is written from the values given and from the values
of the types that the declarations hold (`TunableSet::typeSigma`,
`typeEpsilon`), never from what the model held there. So what the model
held in those sites before (the values of the last update, or of one that
was refused after it had written them) does not matter, and a refusal
needs no undo. Nothing else of the model is written by anything: it is the
simulation's own, and only an update touches it.

**U2. Checked where it is asked.** With `MDRT_CHECK_KEPT_TUPLES` set in
the environment ([environment.md](environment.md#debugging)) every update
also builds its program as before D[update-cost], from a copy of the model
of the compile with the values put into it by `applyTunables`, and
compares the two programs member for member and to the bit
(`driver::describeDifference`). A difference is a defect of MDIR: the
update fails with a message that names what differs and changes nothing,
and the simulation builds from a copy from then on. Without the variable
nothing is compared, and U1 is what holds. `python-update-cost*.test` run
with it.

**How much, and its bound.** One model for a simulation, whatever the
number of its updates, from its first update on: the copy of the control
and the system with its topology, which every update made and freed
before. A simulation that is never updated holds none.

**Whether a tuple set is disjoint** is asked of its members in one pass
with a mark for each particle (`Program::TupleSet::hasNoRepeat`), where a
copy of the members was sorted: the same answer for every list, at every
build, of a compile and of `mdir run` as of an update.

**The tails of the pair terms** (D209) are constants that every build
makes. Their quadratures are functions of the values of the names that the
expression of the term reads and of the cutoff, and of nothing else of a
pair of classes of particles. A build computes each once for those values,
compared as bits, and takes it for every pair of classes and every point
of a derivative that asks for the same: to the bit the number that it
computed each time before. The memo is the build's: it holds at most one
entry for each pair of classes and each point of a derivative that the one
build asks for, and ends with the build. Nothing of a tail is kept from one update to the
next, so the values that a fit passes through leave nothing behind.

**Not kept.** The arrays of the program, which every update builds from
the topology, most of what the rebuild of 408,609 atoms still costs; and
the arguments of the entry, which a start copies (#326).

**Tests.** `python-update-cost.test` and `-gpu.test`: a sequence on the
dipeptide in water with every kind of tunable (updates of each alone, of
all together, and back; between steps; a commit of a borrow;
`gradient()`; an update that is refused after its values were written,
and the one after it; a checkpoint and a run continued from it; a frame
evaluator beside its sampler), in double and mixed precision, with the
comparison of U2 at every update, whose every state, force, energy, and
derivative equals, to the bit, that of a simulation whose updates copy
the model of the compile (`Simulation._keep_model`, private), with the
counts of the programs built from the kept model and from a copy
(`Simulation._tunable_builds`, private); and a number of the kept model
written over by the test (`Simulation._spoil_kept_model`, private), which
the comparison finds and names.

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

Creation, entry execution, and destruction take the
same process mutex. Since D211 a simulation compiles
one program, whose entry begins the run or continues it as an argument says,
so it owns one engine, and since D215 one activation of its
entry runs its parts (above). Polling Python happens between parts after
releasing the mutex.
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

`python-simulation-lifetime-{numpy,mdir,runtime}-first{,-gpu}.test` each run a
fresh process in the NumPy-first, MDIR-first, or runtime-first order (both
runtimes first on the GPU), one order per file. Each process runs the original 24 retained-engine cycles, then seeds 89, 196, and
20261006 once each for 32 operations, retaining up to five simulations that
have run more than one part. Six additional lifetimes run on three host
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

The reviewer also measured JAC through the Python simulation on idle GPU 0
(RTX 3090, mixed, PME grid $64^3$, rigid hydrogen bonds and water, 0.8 nm
cutoff; 1,000 warmup steps and 10,000 timed steps, one run per branch):
**0.2761 ms/step** for main with #94 (no JIT change), against **0.2766 ms/step**
at b27129d, +0.2%, reported within run-to-run noise. Creation plus the first
1,000 steps took 33.3 s and 25.3 s respectively; the reviewer attributed the
difference probably to run order (baseline first), not to this change. These
are reviewer measurements, not additional repetitions of the pair benchmark.
See the [review timing](https://github.com/vv137/mdir/pull/93#issuecomment-5999524681).
