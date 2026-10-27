# Principles of development

Status: adopted 2026-09-30.

These principles come from defects that reached runs of real systems. Each
cites the case that taught it. They apply to the dialects, the passes, the
lowerings, the runtime, and the driver alike.

## 1. A fact that a pass relies on is in the IR

A pass may rely only on what the IR states: types, attributes, and the
structure of ops. A fact that one pass knows and another assumes, kept in a
map of a pass or in a comment, is a defect waiting for a new input.
An attribute carrying semantic meaning is written through a named dialect
helper or an inherent op accessor and is verified where it is read. A raw
string attribute name in a pass is not a contract (D345,
#625).
The storage-form module declaration is read and changed through a named
dialect contract, with its type, placement, and cast provenance checked
(D349, #636).

| Case | What was implicit | What the IR states now, or should |
|---|---|---|
| A buffer of the members of the tuples of a SHAKE set served as scratch for the particles of Cellulose; kernels wrote beyond it (eaca883) | The storage form `memref<?xf64>` did not say which set a buffer held; the pool reused buffers by type | The storage form now carries `#md_exec.set_layout<@set, rank>` on fields, scratch, tuple members, and incidence. Loop verification rejects a different set, and the pool still keys reuse by set and complete type (D337). A general memory planner remains [planned](#planned-work). |
| The loops of a fused kernel wrote at once to scratch that storage gave to several of them (19fa9ef) | Storage assumed that loops run one after the other | Fusion checks that the loops' buffers are distinct; an effects summary of each loop should state it |
| A tuple set whose tuples share no particle is evaluated once per tuple (eaca883) | — | `disjoint` on `md.tuple_set` and `md_exec.tuple_for`, which the driver proves from the members |
| The groups of the constraints update in a chain, one after another, though they share no atom (D83) | That SETTLE and the sets of SHAKE share no atom was known to the driver only | `md.disjoint_union`, which the driver checks from the members; `md-bypass-updates` relies on it |

The positive side: a fact in the IR is also a license to optimize. The
driver proves `disjoint`; the lowering evaluates each tuple once because the
IR says it may.

## 2. An optimization states its precondition, and the precondition is checked

Every transformation names what must hold for it to preserve meaning, in
terms of the IR, and checks it where it applies. Where the check is not
static, a debug mode checks it at run time.

- Fusion of loops over pairs and tuples: the same positions, no loop reads
  what an earlier one writes, and buffers for sums of their own.
- Deferral of the readbacks of sums: each sum has slots of its own; a copy
  is left out only when nothing that runs on the device lies between it and
  the copy before.
- Reuse of device memory freed by a program: one stream, in order; the
  work of a second stream allocates and frees nothing (D87).
- Work on a second stream: every op before its join is independent of it,
  from declared effects and traced aliases (D87).
- SHAKE by Newton: a fixed number of iterations reaches the rounding for a
  step no larger than the constraints were built for; a debug mode should
  check the residual.
- The flags in mapped memory of the host (D118): a store of a kernel is
  visible to the host after an event, and a store of the host to a kernel
  launched after it; checked on x86-64 with an RTX 3090 and driver 595.84,
  not on a platform whose mapped memory is not coherent, where the copy of
  D113 is the fallback.

### Optimizations that reorder or overlap work

An optimization that changes the order in which work runs, or runs work at
once, is exact or wrong: nothing in a test of a few systems tells a race
that did not happen from one that cannot. Such an optimization carries an
argument, written in its decision: the claim (what is the same as in the
serial program), the proof, and the list of its premises, each with the
place where it is checked, statically in a pass or the lowering, or as a
contract of an op that was audited. The pass that applies it and the
lowering that relies on it check the same condition with the same code. A
bitwise comparison with the serial program in the deterministic mode is
the evidence beside it, not the proof. D87 is the pattern.

An optimization that changes results beyond the rounding of sums, such as
a list of neighbors kept for a fixed number of steps, is not of this kind:
it is a mode or an option, off by default unless it is a documented
choice of the method, it warns when a run chooses it, and the log records
it.

## 3. Every optimization can be turned off, and the two are compared

An optimization has a switch, and the tests compare the results with and
without it. Sums in a fixed order make many of the comparisons bitwise.

| Pair | Compared |
|---|---|
| Fused and separate loops; lanes and one thread per particle; owner-computes and replicated tuples; deferred and immediate readbacks | Bitwise |
| CPU and GPU; OpenMP and serial | To the rounding of the sums |
| Double and mixed precision | To a tolerance set from the precision |

## 4. Numerical assumptions are measured on the hard cases

A convergence argument states the regime it holds in, and a test covers the
edge of the regime. SHAKE by 12 sweeps assumed that `m_H / m_X` is small;
masses of hydrogen repartitioned to 3 amu made it about 1/2, and a step of
4 fs became unstable (eaca883). Tests include repartitioned masses, steps of
4 fs, hot starts, and large cells.

## 5. Scale is a separate axis of testing

Defects in sizes and indices show only where counts differ: more particles
than tuples, counts that are not multiples of a block, rows longer than a
warp. The small systems of the unit tests do not reach them. The Amber
benchmark suite (`scripts/benchmarks/amber`) runs systems from 23,558 to
1,067,095 atoms, and short runs of it belong to the tests of a release.

## 6. Tools check what tests cannot

| Tool | What it finds | Where it runs |
|---|---|---|
| compute-sanitizer memcheck, racecheck | Accesses beyond buffers, races between threads | The GPU tests, and short runs of a large system |
| AddressSanitizer, UndefinedBehaviorSanitizer | Host memory: heap corruption, dangling references (such as an `llvm::Twine` held beyond its temporaries) | A build of the driver and runtime |
| `MDRT_TRACE`, `MDIR_PRINT_AFTER` | Which kernel of which module a report names | Diagnosis |

## 7. Inputs are varied on purpose

Readers are tested with the variants that exist: topologies of the format
before Amber 7, without `ATOMIC_NUMBER`, with repartitioned masses, and
files that converters such as ParmEd write (which took hydrogens of 3 amu
for helium). A reader rejects what it cannot read with a message; it never
indexes past what it read. A comparison with another program reads the
files that the other program reads, not a conversion of them: a topology
that ParmEd wrote again lost 2283 dihedral terms of Factor IX, and every
rate of Factor IX up to D132 was of another model than pmemd's (D133).

## 8. Checklist for a change to a pass or a lowering

- What does the change assume about its input? Is it in the IR, and is it
  verified?
- Does it reuse, alias, or reorder buffers or copies? Under what condition,
  checked where?
- Can it be turned off, and is there a test that compares both ways?
- Does it depend on sizes or counts? Is there a test where they differ?
- Does it change a numerical method? Is the hard case of Section 4 tested?

## 9. The primitives are closed under transposition

Status: a design goal (D292, #450), not yet a checked rule.

MDIR is written as a composition of a small set of primitives: map,
gather, scatter-accumulate, halo copy and halo sum, reduction, broadcast,
scan, permutation, compaction, a linear transform with a declared adjoint
(the FFT first; the maintainer's ruling on #452), and the structural sorts
and builds. The goal is that the adjoint of each primitive is again a
primitive of the set: gather and scatter-accumulate, $G$ and
$G^\mathsf T$, reduction and broadcast, scan and reverse scan, a
permutation and its inverse, compaction and expansion, the forward FFT and
the unnormalized backward FFT, a map and its transposed Jacobian. Then the
derivative of any part is a composition of the same primitives, lowered
by the same strategies, and every strategy written for one direction
serves the other: a privatized scatter (D285) is also the transpose of
every gather that the forward pass makes.

Each primitive states its properties (associative, commutative, exact,
stable, ordered), and a lowering chooses by them, as principle 2 asks of
any precondition. A part that does not fit the set says so and why; that
is a finding, not a failure.

[primitives-map.md](primitives-map.md) gives the set, the properties, and
the map of every part of MDIR to its primitives, with the state of the
code and the gaps, each an issue.

The first executable semantic FFT bridge (D341, PR #563)
preserves the declared adjoint while giving backward execution a private
input buffer. Its staged host-device copies currently synchronize both
runtime streams; eliminating that boundary belongs to #580 under #567.
An explicit dependency token alone does not establish transport overlap.

## 10. Execution contracts

Status: design, D334 (#574). The detailed contract is in
[execution-contracts.md](execution-contracts.md). It connects provenance,
effects, validity, and witnesses across `md`, `dyn`, `md_exec`, and
`md_dist`. Semantic IR says what is computed, verified execution IR says
which access and schedule is legal, and a physical plan selects a legal
implementation for the target.

Invariant meaning belongs in types or static attributes: a buffer's set
(#565) and a distributed feature's representation (#571). State that can
change during a run belongs in SSA values and explicit operands: positions,
cell, ownership epoch, and transfer completion. A witness is an SSA result
of a check and is valid only for the state from which that check made it.
The consumer and its lowering check that the witness names their state.
An attribute that gives an operation semantic meaning is declared on that
operation or accessed through a named, verified dialect helper. Passes do
not read or write such an attribute by a string name outside its dialect;
the op verifier checks that the value satisfies the declared contract.
The batched stage's row count is one such invariant: `md.stage_rows` is
checked against a known tensor dimension before bufferization and retained
as a static function attribute afterward (D351, #642).
An outlined stage body uses the checked `md.stage_body` unit marker only on
a function with a body (D353, #647); lowering reads it through
the MD dialect helper.

A required condition is either proved or established by a sound runtime
guard with a legal recovery. If it remains unknown, the path is not executed
and an unsupported-contract diagnostic is issued. A refuted required
condition, contradictory type, mismatched entity, invalid evidence, or
demonstrated uninitialized read is invalid IR and is refused. An optional
strategy whose requirements are false or unknown uses an independently legal
baseline with an `explain` reason. `Unknown` never means true, zero,
initialized, fresh, or independent. A runtime predicate's availability does
not establish its truth, and an event's availability does not establish
completion. Existing user inputs keep their behavior; `mdir check`
diagnoses input defects. Legality is established before measured cost
chooses a strategy.

Numerical requirements are represented separately from memory and
structural requirements, but are prerequisites of a legal plan. The cost
model cannot weaken them: a race-free atomic sum is illegal when a fixed
order is required. `fixed_order`, `any_order`, exact integer
accumulation, precision mode, `fast_math`, and approximation accuracy are
different guarantees. A behavior-preserving refactor keeps lowered IR and
PTX byte-identical and deterministic results bitwise identical. This design
adds no operation, plan choice, control key, or runtime behavior.

The [boundary rules](execution-contracts.md#8-contract-records-and-subjects)
name the exact state a proof depends on and require a new proof after a
relevant mutation. Initialization, readiness, and contribution completeness
are separate facts. A token's availability is not copy completion; a copy's
completion is not release of a host writer's staging buffer. The first
#567 implementation preserves existing waits while making these
dependencies verifiable. Collective order and numerical policy are checked
before a physical plan is selected.

The [resident FFT strategy](fft-device-resident.md) (D[fft-device-resident],
#580) applies these rules to a bounded composition: target layout is proved
or guarded, backward inputs stay private, and copy completion releases only
the buffers and queue dependencies it actually covers. D358 checks its copy/completion pairs before backend selection; general
wait scheduling remains separate.

## Planned work

1. The storage form carries the set of field, loop-reduction scratch, tuple-member, and
   incidence buffers in an identity layout (D337). Loop verification checks their
   symbolic index domain, and the lowering boundary rejects casts that
   reinterpret one set as another. Next, carry the set of other structural
   buffers, and make memory planning a pass of its own over
   lifetimes and sets. Dynamic row counts still come from the validated
   input and the pass's size map.
2. Copies between host and device are ordered by async tokens in the IR;
   deferral and merging of readbacks become a general scheduling pass, and
   the path to CUDA graphs is open.
   Storage assignment already emits token-bearing copies with immediate
   waits. #567 checks readiness and buffer lifetimes without moving
   them; delayed readbacks are a separate measured step.
3. The `md_exec` loop interface now derives a summary of each operand's
   read, write, or accumulation and its row scope (D342,
   #566). Missing set or algebra declarations and opaque body effects make
   it incomplete, so it proves no optimization. Next, make one checked
   summary serve fusion, D87 stream assignment, and distributed scheduling;
   those consumers retain their existing checks and choices today.
4. A debug lowering checks the bounds of every load and store, and the
   residual of iterative solvers.
5. Done: the tests have a tier under compute-sanitizer (`lit -Dsanitize=1`,
   `test/Sanitizer`) and a tier of short runs of the Amber suite
   (`MDIR_BENCH_DIR`, `test/Scale`). Kernels are named after the op that
   they come from and their ordinal in the lowering, such as
   `mdir_run_particle_for_121`, in profiles and in the reports of the
   sanitizer (D354; until it, the line of the op in the module
   of `mdir emit` stood after the op).

Compiler/runtime buffer roles must come from a shared, versioned declaration (D350); a parallel enum or numeric selector is insufficient. Validate externally supplied selectors before indexing runtime storage, and include ABI compatibility versions in compiled-object cache identity.
