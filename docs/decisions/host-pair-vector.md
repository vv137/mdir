# D[host-pair-vector]

Recorded at its merge.

**On the CPU target, a loop over the pairs of the neighbor matrix with the policy `(directed, owner_only)` may run each row $W$ candidates at a time in registers of MLIR's `vector` dialect, the strategy `lanes(W)`, which `md-exec-choose-host-pairs` states on the op where one shared precondition holds and either the deterministic mode (on every host) or the host's measured policy selects it, and which the host lowering builds with one shared emitter; every other loop, and outside the deterministic mode every loop on a host without a measured policy of lanes, keeps the scalar row of today.** Issue #554, under #549 (step 4 of D267) and #444. The maintainer ruled on the design on PR #823: one constant $W$ in the deterministic mode, `MDIR_HOST_PAIRS=scalar` pinned in the corpus and the lit suite, f32 tables on the host a separate decision, the one-thread share by the proxy below, and Table 10.9 not measured again.

## The measurement before the choice (#554, the maintainer on PR #555)

The share of the step that the loop over pairs takes, in the NVE protocol of D333, on its systems and thread counts, on `main` (17704a2d). The sampler of D267 (`scripts/benchmarks/cpu-breakdown/sampler.c`, `prof.py`) attributes the samples of the main thread's wall clock to the parallel regions of the entry. Three changes to the tools were needed; they were merged as their own PR (#836, issue #833), so the measurement can be reproduced from the repository: since #655 the executable exports no dynamic symbols, so the sampler finds the descriptor of the GDB JIT interface in the static symbol table instead of through `dlsym`; and the loops over pairs in f64 read no radial table, so `prof.py` classifies a region by its Lennard-Jones powers (`powi`) as well. A program of one thread has no parallel regions (the loops are inlined into the entry), so the one-thread share is measured on the program of two threads run by one OpenMP thread (`OMP_THREAD_LIMIT=1`; a variant `mp1`/`dp1` of `bench.py`), whose step is stated beside the true one-thread rate.

*Measured* on an EPYC 7543, at 1 and 16 threads. There were two jobs per system, submitted together, which ran one after another on the same machine, so they are separated in time, not by machine. Each ran four alternating rounds of `m1`, `m16`, `d1`, and `d16`, with the first round dropped, followed by one sampled run of each sampled variant. MDIR's rate is that of the second half of its steps. The pair share comes from the samples of the main thread's wall clock over the same half, and includes the waits of the loop's barrier. "Bound" is $1/(1-\text{share})$.

| System | Mode | 16 threads: share (jobs a, b) | Bound | 1 thread, proxy share | 1 thread, loop time / true step | Bound |
|---|---|---|---|---|---|---|
| JAC | mixed | 56.61%, 56.54% | 2.30, 2.30 | 65.70%, 65.71% | 69.7%, 68.7% | 3.30, 3.20 |
| JAC | double | 82.97%, 83.32% | 5.87, 6.00 | 88.40%, 88.52% | 93.3%, 92.0% | 15.0, 12.5 |
| TIP3P 103k | mixed | 63.87%, 65.54% | 2.77, 2.90 | 68.88%, 68.84% | 70.5%, 70.5% | 3.40, 3.39 |
| TIP3P 103k | double | 87.74%, 87.80% | 8.16, 8.20 | 90.29%, 90.23% | 93.0%, 94.8% | 14.3, 19.4 |
| Cellulose | mixed | 56.11%, 55.68% | 2.28, 2.26 | 62.94%, 63.07% | 63.9%, 65.5% | 2.77, 2.90 |
| Cellulose | double | 82.92%, 82.73% | 5.86, 5.79 | 86.83%, 86.84% | 89.3%, 90.7% | 9.35, 10.7 |

- **The one-thread proxy.** The proxy is the program of two threads run by one OpenMP thread, which took 1.5 to 6.0% longer a step than the true one-thread program. Its overhead (PME's spreading by atomics, the forks and joins) lies outside the loop over pairs, so the proxy's share understates the one-thread share. The loop's time in the proxy divided by the true step assumes the loop costs the same in both programs. The one-thread share lies between the two columns.
- **The rates of `main`** (ms/step, mean ± sample standard deviation of the three kept rounds, jobs a and b):

  | System | `m16` | `d16` | `m1` | `d1` |
  |---|---|---|---|---|
  | JAC | 10.161 ± 0.020, 10.160 ± 0.032 | 26.33 ± 0.03, 26.52 ± 0.22 | 128.84 ± 0.46, 128.72 ± 0.38 | 360.7 ± 3.3, 359.3 ± 1.2 |
  | TIP3P 103k | 148.75 ± 1.35, 145.84 ± 0.72 | 428.65 ± 0.75, 423.55 ± 0.69 | 2024.7 ± 15.6, 2005.4 ± 9.3 | 5985 ± 143, 5849 ± 9 |
  | Cellulose | 202.17 ± 1.74, 199.45 ± 0.35 | 513.0 ± 5.3, 512.6 ± 2.4 | 2541 ± 63, 2491 ± 6 | 6833 ± 191, 6715 ± 33 |

  The wider spreads are single slow rounds: Cellulose round 1 of job a in `m1` and `d1`, and TIP3P round 2 of job a in `d1`.
- **Compared with D267.** At 16 threads in mixed precision the NVE share of TIP3P 103k, 64 to 66%, is that of D267's NPT configuration, 64%, on another CPU model (EPYC 9354). JAC and Cellulose are lower, 56 to 57%, and have bonded terms (4.5 to 5.8% of the step) that the rigid water box lacks. The rest of the mixed step is the builds of the matrix (about 15% at either thread count), PME (7 to 13%), and the loops over particles.
- **Table 10.8.** These rates are of another CPU model than D333's (EPYC 9354), so they neither confirm nor replace Tables 10.8 and 10.9. The text that cited D267's 64% and its bound now gives these shares (white paper, Section 10).

## The loops the host lowering sees (survey)

Every control file of the corpus of `scripts/builder-equivalence/check.py` (the examples, the tests including the parts that split-file divides, the prepared Amber suite, and the corpus cases), with the target set to the CPU and a GPU choice of groups and a pruned list removed, emitted to the storage form before `convert-md-exec-to-loops`: 156 control files hold 754 loops over pairs; 108 do not emit (tests that expect an error, inputs that RUN lines make, models that run on a GPU only).

- **Policy and flags.** All 754 are `(directed, owner_only)`. None has `centers`, `center_set`, `ghost_free`, or `images(explicit)`; the two-ended `(directed, atomic)` form appears only in programs of `md_dist` through `mdir-opt`.
- **Outputs.** Forces `vector<3xf64>` (315 loops) or `vector<3xf32>` (288), a scalar f64 field (5), or none (146, the loops of energies alone); the global sums f64 and `vector<3xf64>` (the energy and the virial), summed in the kernel's type along a row and widened once.
- **Algebra.** `any_order` on every value, except 30 loops of the deterministic tests, `fixed_order`.
- **Exclusions.** The host's build removes the excluded pairs from the rows (`emitExclusionFilter`); the kernel holds no exclusion logic, and the test $j \ne i$ of the scalar row stays.
- **Kernel signatures.** $(r^2, \mathbf d, t_i, t_j, q_i, q_j)$ in f32 (342) or f64 (322), with real fields in place of the types (52), and without fields (20).
- **Operations in kernels**, by the loops that contain them, and their lane mapping:

| Operations | Loops | Lane mapping |
|---|---:|---|
| `arith`: `divf`, `mulf`, `subf`, `addf`, `cmpf`, `select`, `constant`, `addi`, `subi`, `bitcast`, `index_cast`, `maxsi`, `minsi`, `shli`, `shrui`, `extf`, `truncf`, `fptosi`, `maximumf`, `minimumf` | 754 | The same op on `vector<W x T>` |
| `math`: `sqrt`, `fma`, `exp`, `floor`, `absf` | 408 | The same op on vectors |
| `math`: `erfc` (160), `log` (5) | 165 | The same op on vectors; `convert-math-to-libm` calls the scalar function for each lane (no vector library is linked), so these gain nothing |
| `math.fpowi` with a constant exponent | 753 | The product tree of LLVM's code generator for `powi` with a constant exponent (repeated squaring), which the scalar form gets; MLIR's lowering takes no vector of exponents |
| `vector.extract`, `vector.broadcast`, `vector.from_elements` on `vector<3xT>` and the rows of tables | 724 | Component-major layout: `vector<n x T>` becomes `vector<n x W x T>`, $n$ registers of $W$ lanes |
| `md.lookup` of a table by the two types (`memref<?x?xvector<2xf64>>` in 665 loops; `f64`, `vector<4, 6, 7 x f64>` in the others) | 680 | A gather: one load a lane, then a transpose into components. In mixed precision the table is f64 and truncated, as today |
| `memref.get_global` and `vector.load` of a row of a radial table (1,281 to 5,121 rows of 4 f32) | 346 | A gather: one row load a lane |
| `md_exec.yield` | 754 | — |

No kernel of the corpus holds an op outside these rows, a nested region, a call, or a write to memory. The nine Amber suite systems have 4 loops (NVE) or 8 (NPT) each: forces with the energy and the virial (72 ops), the energy alone (39), forces alone (40), and, under NPT, forces with the virial (51). The legal subset of the first slice therefore covers every loop of the corpus.

## The first slice

**Strategy attribute.** `host(lanes(W))`, an optional attribute of `md_exec.pair_for`, printed only where it is set. Every program without it, and every GPU program, prints as before. `md-exec-choose-host-pairs` sets it on the CPU target only, after `md-exec-assign-algebra` and before `md-exec-assign-storage`, which carries it. Without it, the row is scalar.

**One precondition.** `md_exec::hostpairs::checkLanes(PairForOp, W)` (`include/mdir/Dialect/MDExec/HostPairs.h`) is called by the pass and again by `convert-md-exec-to-loops`, which fails on an attribute that it refuses (principle 2):

1. the policy is `(directed, owner_only)`, without `centers`, `center_set`, `ghost_free`, or `images(explicit)`;
2. the kernel has no regions, and every op in it has a lane mapping (`hostpairs::getLaneMapping`, the table of the survey; `fpowi` needs a constant exponent);
3. it computes in f32 or f64, and its values are numbers or vectors of one dimension;
4. the op states the algebra of what it reduces (D301);
5. $W$ is a power of two from 1 to 64; the verifier of the op also checks this.

With `explain`, the pass remarks the choice, or the first condition that fails, or the policy's reason.

**One emitter.** `kernels::emitPairRowLanes` (`lib/Conversion/MDExecKernels/HostPairLanes.cpp`) replaces `emitPairKernel` in the host's `lowerPairFor` where the attribute is set. `emitPairKernel` is not changed. The code is `vector`, `arith`, `math`, and `scf`, plus one LLVM intrinsic; no intrinsic of a host is written. The details are in [device-primitives.md, Section 7](../device-primitives.md#7-lanes-of-the-hosts-pair-rows-dhost-pair-vector):

- full groups of a row read by `vector.load`, then one masked tail whose padding lanes repeat the row's last candidate;
- every value as its components of `vector<W x T>`, and the shared values as scalars;
- a read a lane for the positions, the fields, `md.lookup`, and the rows of a radial table;
- `math.fpowi` with a constant exponent as `llvm.intr.powi` of the vector with the scalar exponent, which LLVM's code generator expands as it expands the scalar one;
- the minimum image and $r^2$ op for op as in the scalar row.

**Reduction order.** Lane $l$ sums the candidates $k \equiv l \pmod W$ in increasing $k$, in the type of the scalar row's sum. A sum that the kernel widens is widened in each lane. The lanes are combined by halving (`emitLaneReduce`, `ReductionEmitter.h`), the order in which lane 0 of the device's butterfly combines a warp. The destinations and the fixed-order global sums (D171) work as before. A row's result depends on its list, $W$, and the kernel alone.

**$W$.**

- `width`, where given (`MDIR_HOST_PAIRS=lanes:W`, for comparisons and tests).
- In the deterministic mode, 8 in f32 and 4 in f64 on every host (the ruling).
- Otherwise the host's register width over the bits of the kernel's type: 512 bits with `avx512f`, 256 with `avx`, 128 otherwise. The CPU and its features are those of the machine that compiles the host code, read as `JITTargetMachineBuilder::detectHost` reads them, so that the Python front end, which builds its pipeline before it initializes the target, sees the same host. A CPU that `MDIR_HOST_CPU` names has the features of its name and every feature they imply (`x86-64-v3`: AVX2, so 256 bits; `x86-64-v4`: 512).

**The deterministic mode takes the lanes on every host,** measured or not (the maintainer's ruling on PR #823, option 2): one order on every host comes first there, and the rule that an unmeasured host keeps the scalar row is a performance default of the ordinary modes. Its bits are then the same on every host, and they differ from those of the scalar row of before. `MDIR_HOST_PAIRS=scalar` still selects the scalar row.

**Policy.** `hostpairs::findPolicy` is the host column of #433's table, keyed by the CPU that LLVM names; it lists the hosts whose measurement made the lanes their default. Outside the deterministic mode, every other host keeps the scalar row, and the remark of `explain` says so. The entries for `znver3` (EPYC 7543) and `znver4` (EPYC 9354) take the lanes, from the timing below.

**Switches.**

- `MDIR_HOST_PAIRS=scalar|lanes|lanes:W` overrides the policy for comparisons and tests. The driver parses it before anything is lowered; any other value, and a W that is not a power of two from 1 to 64, is an error of the input of a program for the CPU that names it. A program for a GPU does not read it; `mdir doctor` reports the host code whatever it says, and its check of the CPU, a program for the CPU, fails with the error. The lit suite sets `scalar` (`test/lit.cfg.py`), so that `mdir emit` of a CPU program does not depend on the host.
- `MDIR_HOST_CPU=<cpu>` compiles the host code for the CPU that LLVM names, with that CPU's own features (`x86-64`: SSE2 alone). This is how a narrower host is checked here. The host code of every program reads it, so the driver refuses, for the CPU or a GPU, a name that LLVM's X86 back end does not know as a 64-bit CPU, and `mdir doctor` reports the CPU and the AVX features of the machine that compiles the host code. The compile cache keys a host object by the CPU and its features, and the module holds $W$.

No control-file key or file format changes.

## Validation

**Identity.** `scripts/builder-equivalence/check.py compare` of `main` b7d801d9 and the branch at its head over the whole corpus, with `MDIR_HOST_PAIRS=scalar`, at the stages `module` and `lowered`, printed and raw, for the CPU and GPU targets: all 1,070 cases (the runs and the segments) are identical, of which 474 stop with the same error in both builds. The identity is of the scalar row: it shows that the lanes leave every program without them as it was.

**Lane arithmetic, to the bit** (`test/Driver/host-pair-lanes.test`):

- `lanes(1)` gives the scalar row's state after 20 steps on the dipeptide in OPC with PME, in the mixed and double modes.
- `lanes(8)` gives the scalar row's state on dimers of argon whose rows hold some twenty candidates with one within the cutoff, in both modes.

These two cannot see a wrong tail or a wrong order of the combination: `lanes(1)` has no tail and nothing to combine, and a row with one contributing term gives the same bits in any order.

**The order of the combination** is pinned in the IR: lane $l$ with lane $l + W/2$, then $W/4$, down to one, at $W$ = 4 (`host-pair-lanes.mlir`, the energy) and at $W$ = 8, the deterministic width of f32, for the three force components and the energy, which is widened in each lane first (`host-pair-lanes-w8.mlir`). With the halving made to pair each lane with itself (temporarily), both fail. Neither the bound below nor the deterministic properties can see a wrong pairing, since any order of the same terms is within the bound, and the properties compare the lanes with themselves.

**Lanes against the scalar row at the bound of a sum's order.** The lanes add the same terms as the scalar row in another order; the terms are the same numbers, since each op of the kernel is evaluated as the scalar row evaluates it. Two orders of a sum of $n$ terms $t_k$ differ by at most $2\gamma_{n-1}\sum_k \lvert t_k\rvert$, with $\gamma_m = m u/(1 - m u)$ and $u$ the unit roundoff of the row's sum ($2^{-24}$ in mixed precision, $2^{-53}$ in double) [[Higham2002]](../references.md#higham2002). The bound of each atom and component takes $n$ as the candidates of its row within the cutoff and $\sum_k \lvert t_k\rvert$ over them, from a reference of the direct terms of the kernel (Lennard-Jones and $\mathrm{erfc}(\beta r)/r$), without the excluded pairs of the topology, which the list leaves out, times 1.01 for the reference's own rounding. $W$ is pinned: `lanes:8` and `lanes:4`, and the test checks, from the remarks of the chooser, that every loop over pairs takes `host(lanes(W))`.

- *In the suite* (`test/Driver/host-pair-lanes-bound.test`): the dipeptide in water with PME and a reach of 1 nm, ten types, the exclusions of its topology, rows of the list of 118 to 371 candidates (the candidates within the reach that are not excluded, by the reference), 1,017 of 1,168 not a multiple of 8 and 862 not a multiple of 4, so that the tail's padding repeats a candidate. Largest ratio to the bound: 0.016 and 0.017 in double, 0.020 and 0.023 in mixed, for `lanes(8)` and `lanes(4)`. A candidate dropped, doubled, or misgathered adds or removes a whole term, which the bound, a few units of $10^{-5}$ of $\sum\lvert t_k\rvert$ in mixed precision and of $10^{-14}$ in double, leaves out in both modes. With the tail's mask removed from the emitter (temporarily) the largest ratio is 5.7e10; with the chooser refusing every loop, the check of the lanes fails.
- *On the three systems of the timing* (scratch, 16 threads; the 1-thread runs give the same bits):

| System | Mode, $W$ | Candidates within the cutoff | max $\lvert\Delta F\rvert$ / bound | median $\lvert\Delta F\rvert$ / bound | max $\lvert\Delta F\rvert$, kJ/mol/nm | $\lvert\Delta U\rvert$ / bound |
|---|---|---|---|---|---|---|
| JAC | mixed, 8 | 169 to 244 | 0.12 | 1.1e-3 | 1.5e-3 | 1.0e-5 |
| JAC | double, 4 | 169 to 244 | 0.12 | 1.2e-3 | 3.9e-12 | 0 |
| TIP3P 103k | mixed, 8 | 62 to 325 | 0.033 | 8.3e-4 | 1.7e-3 | 1.6e-5 |
| TIP3P 103k | double, 4 | 62 to 325 | 0.017 | 9.4e-4 | 4.6e-12 | 0 |
| Cellulose | mixed, 8 | 115 to 279 | 0.20 | 1.1e-3 | 2.2e-3 | 3.9e-6 |
| Cellulose | double, 4 | 115 to 279 | 0.17 | 1.1e-3 | 4.1e-12 | 0 |

Every force component of every system is within its bound, by a factor of at least 4.9. In double precision the potentials agree to the bit, the rows' differences lying below the total's last place.

**Determinism.** In the deterministic mode with the lanes ($W$ = 8 in f32 and 4 in f64), the states after 20 steps are identical:

- at 1, 2, 4, and 7 threads;
- from run to run;
- with the host code compiled for SSE2 alone (`MDIR_HOST_CPU=x86-64`), where 8 lanes of f32 take two registers; the pipeline then states `cpu=x86-64 vector-bits=128`, and `mdir doctor` reports the machine of the host code as `x86-64` without AVX.

The deterministic properties of the suite run with the lanes as well, with `MDIR_HOST_PAIRS` unset (the shipped default): output independence (`host-pair-lanes-output-independence.test`), Python against `mdir run` in one part and in segments (`-deterministic-run.test`), and the thermostats across checkpoints in both directions (`-thermostats.test`), in mixed and double precision.

**Against an independent oracle.** OpenMM 8.6.1, Reference platform, in double precision on JAC: the same Amber model, PME with MDIR's $\beta$ and the $64^3$ grid, and the dispersion correction. OpenMM's PME order is 5 and MDIR's is 4.

- RMS $\lvert\mathbf F_\text{MDIR} - \mathbf F_\text{OpenMM}\rvert$: 0.81193 kJ/mol/nm for both scalar and lanes in mixed precision, and 0.81182 for both in double. That is 7.9e-4 of the RMS force, 1030 kJ/mol/nm.
- Potential: −302545.45 (mixed scalar), −302545.45 (mixed lanes), and −302545.40 (double, both), against −302492.13 kJ/mol. The offset of 53 kJ/mol is the potential shift of MDIR's Coulomb cutoff and the order of the grid, the same for both rows.

The tolerance is that the lanes leave the agreement with the oracle unchanged: the RMS differences agree to 5 digits in each mode.

## Timing

The NVE protocol of D333 on JAC, TIP3P 103k, and Cellulose, with the cutoffs and grids of the measurement above: `main` 8c00268e against the branch at 65e396c1 on `main` 8c00268e, rebased unchanged as a1eb115c, with `MDIR_HOST_PAIRS=lanes` ($W$ = 8 in f32 and 4 in f64 on AVX2, 16 and 8 on AVX-512); on JAC also the branch's scalar row (`MDIR_HOST_PAIRS=scalar`) at 16 threads. The branch's later commits change documents, tests, and the comment above the policy table in `HostPairs.cpp`; between the two bases of the rebases `main` changed the code of the skins and of the compile cache, none of it the code of the loops over pairs.

- *Rate windows.* Each run writes a row of its log every twentieth of its steps, and the time of each row is taken as it reaches a pseudo-terminal, so a run has 20 windows of its rate. A run's rate is the mean of its windows after the first 3, without the highest and the lowest tenth. The tables give, for each run, the windows' spread ("within") and the change from the first third of the kept windows to the last ("drift").
- *Run lengths.* JAC at 16 threads: 20,000 steps (windows of 1,000). JAC at 1 thread: 2,000 steps (windows of 100). TIP3P 103k and Cellulose at 16 threads: 2,000 steps (windows of 100). The two large systems were not measured at 1 thread at the head: 100,000 steps a run would take 2.1 to 3.6 hours at 1 thread on JAC in mixed precision and 6.7 to 10.3 hours in double (at the kept rates), and 2.4 to 14 hours at 16 threads on the large systems.
- *Jobs.* At least two jobs a system on each CPU model whose runs were kept. Each job ran 2 rounds of each variant, `main` and the branch alternating, the order reversed in the second; the tables give the mean and the sample standard deviation of the 2 rounds. The jobs are not independent: they ran at the same time as one another, and the two kept TIP3P jobs on the EPYC 7543 overlapped in time.
- *The rule of leaving out.* A job's runs at one thread count are left out where `main`'s rate, averaged over the variants of that count, is more than 10% slower than in the fastest job of the same system and CPU model (the column "`main` vs the fastest job"). It leaves out two JAC jobs on the EPYC 7543 at 16 threads, whose `main` ran 16% slower (contention: 12.06 to 13.15 ms/step in `m16` against 10.31 and 10.41 in the kept jobs); their 1-thread runs are within 0.2% and 3.7% and are kept. Drift does not separate them: every JAC run at 16 threads drifts upward, the kept ones by +1.2% to +3.6% and the runs left out by +4.2% to +15%. A further job on that model, job a of its table, ran in their place at 16 threads only.

**EPYC 7543** (ms/step: the mean of the runs' rates, each the trimmed mean of its windows after the first 3, ± the sample standard deviation of the 2 rounds; ratio = `main` / lanes; within = the largest relative standard deviation of a run's windows; drift = the largest change from a run's first kept third of windows to its last)

| System | Job | Variant | Steps, windows | `main` | lanes | Ratio | within | drift | `main` vs the fastest job, by thread count | branch scalar |
|---|---|---|---|---|---|---|---|---|---|---|
| JAC | a | `m16` | 20000, 20 | 10.31 ± 0.0054 | 6.795 ± 0.0015 | 1.517 | 1.3% | +2.9% | +0.0% | 10.32 ± 0.012 |
| JAC | a | `d16` | 20000, 20 | 26.55 ± 0.054 | 20.21 ± 0.0091 | 1.314 | 0.6% | +1.4% | +0.0% | 26.54 ± 0.0022 |
| JAC | b | `m16` | 20000, 20 | 10.41 ± 0.0089 | 6.847 ± 0.012 | 1.520 | 1.5% | +3.6% | +0.5% | 10.42 ± 0.048 |
| JAC | b | `d16` | 20000, 20 | 26.58 ± 0.014 | 20.95 ± 0.94 | 1.269 | 0.7% | +1.5% | +0.5% | 26.58 ± 0.028 |
| JAC | b | `m1` | 2000, 20 | 129 ± 0.23 | 76.26 ± 0.32 | 1.691 | 1.0% | -0.5% | +0.2% | — |
| JAC | b | `d1` | 2000, 20 | 358.2 ± 0.55 | 267.4 ± 0.66 | 1.339 | 0.5% | +0.5% | +0.2% | — |
| JAC | c (left out) | `m16` | 20000, 20 | 12.61 ± 0.77 | 8.877 ± 0.29 | 1.420 | 2.9% | +5.8% | +15.8% | 12.28 ± 0.39 |
| JAC | c (left out) | `d16` | 20000, 20 | 29.03 ± 0.17 | 22.73 ± 0.28 | 1.278 | 2.3% | +5.3% | +15.8% | 29.05 ± 0.098 |
| JAC | c | `m1` | 2000, 20 | 128.5 ± 0.1 | 76.63 ± 0.039 | 1.677 | 1.3% | +0.5% | +0.2% | — |
| JAC | c | `d1` | 2000, 20 | 359.5 ± 0.41 | 267.9 ± 0.22 | 1.342 | 0.6% | +0.6% | +0.2% | — |
| JAC | d (left out) | `m16` | 20000, 20 | 12.61 ± 0.56 | 8.707 ± 0.89 | 1.448 | 6.5% | +15.2% | +16.7% | 12.32 ± 0.85 |
| JAC | d (left out) | `d16` | 20000, 20 | 29.51 ± 0.5 | 22.86 ± 0.72 | 1.291 | 1.9% | +4.2% | +16.7% | 29.38 ± 0.66 |
| JAC | d | `m1` | 2000, 20 | 132.8 ± 6.1 | 78.77 ± 4.3 | 1.686 | 0.9% | +0.3% | +3.7% | — |
| JAC | d | `d1` | 2000, 20 | 372.5 ± 13 | 281.2 ± 4.9 | 1.325 | 2.0% | +4.2% | +3.7% | — |
| TIP3P 103k | a | `m16` | 2000, 20 | 156.1 ± 0.023 | 98.98 ± 0.87 | 1.577 | 1.0% | +0.7% | +0.2% | — |
| TIP3P 103k | a | `d16` | 2000, 20 | 429.9 ± 1.5 | 323.7 ± 2.1 | 1.328 | 0.6% | -0.7% | +0.2% | — |
| TIP3P 103k | b | `m16` | 2000, 20 | 155.4 ± 1.3 | 98.15 ± 0.65 | 1.584 | 0.9% | -0.6% | +0.2% | — |
| TIP3P 103k | b | `d16` | 2000, 20 | 432 ± 4.7 | 325.6 ± 0.63 | 1.327 | 0.6% | -1.0% | +0.2% | — |
| Cellulose | a | `m16` | 2000, 20 | 199.7 ± 0.17 | 134.9 ± 0.38 | 1.480 | 0.9% | +0.3% | +0.3% | — |
| Cellulose | a | `d16` | 2000, 20 | 508 ± 0.44 | 385.8 ± 0.21 | 1.317 | 0.3% | +0.5% | +0.3% | — |
| Cellulose | b | `m16` | 2000, 20 | 198.6 ± 0.084 | 133.8 ± 0.86 | 1.484 | 0.8% | -0.2% | +0.0% | — |
| Cellulose | b | `d16` | 2000, 20 | 507.7 ± 0.82 | 386.5 ± 0.61 | 1.314 | 0.3% | +0.4% | +0.0% | — |

**EPYC 9354** (ms/step: the mean of the runs' rates, each the trimmed mean of its windows after the first 3, ± the sample standard deviation of the 2 rounds; ratio = `main` / lanes; within = the largest relative standard deviation of a run's windows; drift = the largest change from a run's first kept third of windows to its last)

| System | Job | Variant | Steps, windows | `main` | lanes | Ratio | within | drift | `main` vs the fastest job, by thread count | branch scalar |
|---|---|---|---|---|---|---|---|---|---|---|
| JAC | a | `m16` | 20000, 20 | 10.31 ± 0.0047 | 7.3 ± 0.039 | 1.412 | 1.2% | +2.2% | +6.0% | 10.31 ± 0.017 |
| JAC | a | `d16` | 20000, 20 | 25 ± 0.13 | 18.35 ± 0.034 | 1.363 | 0.9% | +2.0% | +6.0% | 24.93 ± 0.041 |
| JAC | a | `m1` | 2000, 20 | 124.9 ± 0.014 | 76.4 ± 0.47 | 1.635 | 1.7% | +1.5% | +0.3% | — |
| JAC | a | `d1` | 2000, 20 | 341.1 ± 0.38 | 239.9 ± 0.1 | 1.422 | 0.3% | +0.5% | +0.3% | — |
| JAC | b | `m16` | 20000, 20 | 9.569 ± 0.025 | 6.509 ± 0.0026 | 1.470 | 0.6% | +1.2% | +0.0% | 9.566 ± 0.014 |
| JAC | b | `d16` | 20000, 20 | 24 ± 0.026 | 17 ± 0.023 | 1.412 | 0.8% | +1.8% | +0.0% | 23.99 ± 0.0039 |
| JAC | b | `m1` | 2000, 20 | 124.1 ± 0.011 | 75.47 ± 0.024 | 1.645 | 0.9% | +1.0% | +0.0% | — |
| JAC | b | `d1` | 2000, 20 | 341.2 ± 0.11 | 239.6 ± 0.27 | 1.424 | 0.4% | +0.5% | +0.0% | — |
| TIP3P 103k | a | `m16` | 2000, 20 | 133.5 ± 0.53 | 86.36 ± 0.62 | 1.546 | 0.9% | +0.1% | +0.2% | — |
| TIP3P 103k | a | `d16` | 2000, 20 | 378.4 ± 0.099 | 263.1 ± 2.2 | 1.438 | 0.6% | -1.2% | +0.2% | — |
| TIP3P 103k | b | `m16` | 2000, 20 | 133 ± 0.29 | 85.85 ± 0.66 | 1.549 | 0.8% | +0.7% | +0.3% | — |
| TIP3P 103k | b | `d16` | 2000, 20 | 380.5 ± 1 | 262.1 ± 0.11 | 1.452 | 0.3% | -0.5% | +0.3% | — |
| Cellulose | a | `m16` | 2000, 20 | 186.4 ± 0.3 | 131.1 ± 0.47 | 1.422 | 1.3% | +1.0% | +0.8% | — |
| Cellulose | a | `d16` | 2000, 20 | 457 ± 0.32 | 335.9 ± 10 | 1.360 | 0.4% | +0.4% | +0.8% | — |
| Cellulose | b | `m16` | 2000, 20 | 183.4 ± 0.38 | 127.7 ± 0.7 | 1.436 | 1.6% | +0.9% | +0.4% | — |
| Cellulose | b | `d16` | 2000, 20 | 460.8 ± 8.1 | 327.1 ± 0.082 | 1.409 | 0.4% | +0.4% | +0.4% | — |
| Cellulose | c | `m16` | 2000, 20 | 187.1 ± 0.043 | 130.5 ± 0.047 | 1.433 | 1.3% | +0.9% | +1.3% | — |
| Cellulose | c | `d16` | 2000, 20 | 459.5 ± 0.022 | 330.5 ± 0.4 | 1.390 | 0.4% | +0.7% | +1.3% | — |

- *Ranges, the runs left out excluded.* On the EPYC 7543 the step takes 1.48 to 1.69 times less time in mixed precision and 1.27 to 1.34 times less in double; on the EPYC 9354, 1.41 to 1.65 and 1.36 to 1.45. The low end in double, 1.27, is JAC `d16` in job b on the EPYC 7543, whose two rounds of the lanes differ by 6.5% (20.28 and 21.61 ms/step) while `main` and the scalar row agree to 0.1%; the other kept 16-thread job there gives 1.31 in `d16`, as does the next lowest kept double ratio.
- *The branch's scalar row* runs at `main`'s rate: within 0.3% of `main` in each kept job and variant of JAC.
- *The pair loop's share* (the sampler of D267 on the tools of #836; the sampled runs of the first timing of this PR, JAC). On the EPYC 7543, `m16` goes from 57.0% to 33.5%. On the EPYC 9354, over two jobs: `m16` 54.6% to 32.9%; `d16` 79.3% and 82.4% to 75.2% and 75.6%; the one-thread proxy (`OMP_THREAD_LIMIT=1`, whose step is 1.5 to 6.0% longer than the true one-thread step) 65.2% and 65.4% to 44.2% in mixed precision and 88.3% to 83.1% in double.
- *Why double gains less.* In double the loop stays most of the step because `erfc` is called once a lane from libm (no vector library is linked). That is the next measurable step for the f64 loops.

**The default, observed.** A run of the branch with `MDIR_HOST_PAIRS` unset on each CPU model (JAC, 16 threads, 20 steps):

| CPU | CPU that LLVM detects | Pipeline | Lanes of the pair loops (`llvm.intr.powi`, 24 each) | `mdir doctor` | 20 steps |
|---|---|---|---|---|---|
| EPYC 7543 | `znver3` | `md-exec-choose-host-pairs{cpu=znver3 vector-bits=256}` | `vector<8xf32>` (mixed), `vector<4xf64>` (double) | `host code: cpu znver3, avx yes, avx512f no` | ran; the total energy changed by 2.2e-5 of its value |
| EPYC 9354 | `znver4` | `md-exec-choose-host-pairs{cpu=znver4 vector-bits=512}` | `vector<16xf32>` (mixed), `vector<8xf64>` (double) | `host code: cpu znver4, avx yes, avx512f yes` | ran; the total energy changed by 2.2e-5 of its value |

**Policy.** On both CPU models the lanes win in each variant measured at the head, beyond the spread of the runs. So `findPolicy` makes them the default on `znver3` (EPYC 7543) and `znver4` (EPYC 9354); the maintainer ruled on PR #823 to keep the entry of `znver4`.

**Peak memory.** The lanes allocate nothing. Their registers and the stack of a row are the only storage they add, and every buffer of the program is the same as before.

## Left open

- Nothing for the maintainer. The deterministic mode on an unmeasured host was asked on PR #823 and ruled above (option 2).

**Not in this slice.**

- `cluster(M, N)` (#549, after this is measured);
- the two-ended form, and `centers`, `center_set`, `ghost_free`, and `images(explicit)`;
- `vector.gather` and relaid tables;
- tables in f32 on the host in mixed precision (a separate decision, the maintainer's ruling 3);
- the builds of the list (D267 step 3).
