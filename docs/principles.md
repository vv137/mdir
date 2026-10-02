# Principles of development

Status: adopted 2026-09-30.

These principles come from defects that reached runs of real systems. Each
cites the case that taught it. They apply to the dialects, the passes, the
lowerings, the runtime, and the driver alike.

## 1. A fact that a pass relies on is in the IR

A pass may rely only on what the IR states: types, attributes, and the
structure of ops. A fact that one pass knows and another assumes, kept in a
map of a pass or in a comment, is a defect waiting for a new input.

| Case | What was implicit | What the IR states now, or should |
|---|---|---|
| A buffer of the members of the tuples of a SHAKE set served as scratch for the particles of Cellulose; kernels wrote beyond it (eaca883) | The storage form `memref<?xf64>` does not say which set a buffer holds; the pool reused buffers by type | The pools key buffers by set; the storage form should carry the set in its type ([Planned work](#planned-work)) |
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

## Planned work

1. The storage form carries the set of each buffer in its type, so that
   reuse and aliasing across sets cannot be written, and the verifier checks
   the index domains of kernels. Memory planning then becomes a pass of its
   own over lifetimes and sets.
2. Copies between host and device are ordered by async tokens in the IR;
   deferral and merging of readbacks become a general scheduling pass, and
   the path to CUDA graphs is open.
3. Each loop op carries a summary of its effects: what it reads and writes,
   and where (its own element, all members of a tuple, a global sum). Fusion
   decides from it in one place. Begun: the independence of whole ops, from
   their declared effects and traced aliases, decides what runs on a second
   stream (D87); fusion still has checks of its own.
4. A debug lowering checks the bounds of every load and store, and the
   residual of iterative solvers.
5. Done: the tests have a tier under compute-sanitizer (`lit -Dsanitize=1`,
   `test/Sanitizer`) and a tier of short runs of the Amber suite
   (`MDIR_BENCH_DIR`, `test/Scale`). Kernels are named after the op and the
   line of the module of `mdir emit` that they come from, such as
   `mdir_run_particle_for_l44_121`, in profiles and in the reports of the
   sanitizer.
