# D[host-pair-vector]

Recorded at its merge.

**On the CPU target, a loop over the pairs of the neighbor matrix with the policy `(directed, owner_only)` may run each row $W$ candidates at a time in registers of MLIR's `vector` dialect, the strategy `lanes(W)`, which `md-exec-choose-host-pairs` states on the op where one shared precondition holds and the host's measured policy selects it, and which the host lowering builds with one shared emitter; every other loop, and every loop on an unmeasured host, keeps the scalar row of today.** Issue #554, under #549 (step 4 of D267) and #444. The maintainer ruled on the design on PR #823: one constant $W$ in the deterministic mode, `MDIR_HOST_PAIRS=scalar` pinned in the corpus and the lit suite, f32 tables on the host a separate decision, the one-thread share by the proxy below, and Table 10.9 not measured again.

## The measurement before the choice (#554, the maintainer on PR #555)

The share of the step that the loop over pairs takes, in the NVE protocol of D333, on its systems and thread counts, on `main` (17704a2d). The sampler of D267 (`scripts/benchmarks/cpu-breakdown/sampler.c`, `prof.py`) attributes the samples of the main thread's wall clock to the parallel regions of the entry. Three changes to the tools were needed. They are local to the measurement and proposed as their own PR, after which the measurement can be reproduced from the repository: since #655 the executable exports no dynamic symbols, so the sampler finds the descriptor of the GDB JIT interface in the static symbol table instead of through `dlsym`; and the loops over pairs in f64 read no radial table, so `prof.py` classifies a region by its Lennard-Jones powers (`powi`) as well. A program of one thread has no parallel regions (the loops are inlined into the entry), so the one-thread share is measured on the program of two threads run by one OpenMP thread (`OMP_THREAD_LIMIT=1`; a variant `mp1`/`dp1` of `bench.py`), whose step is stated beside the true one-thread rate.

*Measured* on an EPYC 7543, 16 cores of a Slurm job (the scheduler placed every job on this model). There were two jobs per system, submitted together. The scheduler ran them one after another on the same node, so they are separated in time, not by device. Each ran four alternating rounds of `m1`, `m16`, `d1`, and `d16`, with the first round dropped, followed by one sampled run of each sampled variant. MDIR's rate is that of the second half of its steps. The pair share comes from the samples of the main thread's wall clock over the same half, and includes the waits of the loop's barrier. "Bound" is $1/(1-\text{share})$.

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

- `width`, where given (`MDIR_HOST_PAIRS=lanes:W`).
- In the deterministic mode, 8 in f32 and 4 in f64 on every host (the ruling).

**The deterministic mode takes the lanes on every host,** measured or not (the maintainer's ruling on PR #823, option 2): one order on every host comes first there, and the rule that an unmeasured host keeps the scalar row is a performance default of the ordinary modes. Its bits are then the same on every host, and they differ from those of the scalar row of before. `MDIR_HOST_PAIRS=scalar` still selects the scalar row.
- Otherwise the host's register width over the bits of the kernel's type: 512 bits with `avx512f`, 256 with `avx`, 128 otherwise. The CPU and its features are those of the machine that compiles the host code, read as `JITTargetMachineBuilder::detectHost` reads them, so that the Python front end, which builds its pipeline before it initializes the target, sees the same host.

**Policy.** `hostpairs::findPolicy` is the host column of #433's table, keyed by the CPU that LLVM names. Outside the deterministic mode, a host without an entry keeps the scalar row, and the remark says so. *The entry for `znver3` follows the timing below.*

**Switches.**

- `MDIR_HOST_PAIRS=scalar|lanes|lanes:W` overrides the policy for comparisons and tests. The lit suite sets `scalar` (`test/lit.cfg.py`), so that `mdir emit` of a CPU program does not depend on the host.
- `MDIR_HOST_CPU=<cpu>` compiles the host code for the CPU that LLVM names, with that CPU's own features (`x86-64`: SSE2 alone). This is how a narrower host is checked here. The compile cache keys a host object by the CPU and its features, and the module holds $W$.

No control-file key or file format changes.

## Validation

**Identity.** `scripts/builder-equivalence/check.py compare` of `main` and the branch over the whole corpus, with `MDIR_HOST_PAIRS=scalar`, at the stages `module` and `lowered`, printed and raw, for the CPU and GPU targets: all 1,062 cases (266 control files, the runs and the segments) are identical, of which 466 stop with the same error in both builds.

**Lane arithmetic, to the bit** (`test/Driver/host-pair-lanes.test`):

- `lanes(1)` gives the scalar row's state after 20 steps on the dipeptide in OPC with PME, in the mixed and double modes. The same holds on JAC after 2 steps.
- `lanes(8)` gives the scalar row's state on dimers of argon whose rows hold some twenty candidates with one within the cutoff, in both modes.

**Determinism.** In the deterministic mode with the lanes ($W$ = 8 in f32 and 4 in f64), the states after 20 steps are identical:

- at 1, 2, 4, and 7 threads;
- from run to run;
- with the host code compiled for SSE2 alone (`MDIR_HOST_CPU=x86-64`), where 8 lanes of f32 take two registers.

At step 0, the lanes give the same bits at 1 and 16 threads on all three systems in both modes, as the scalar row does.

**Step 0 against the scalar row.** Python API, `lanes(8)` in f32 and `lanes(4)` in f64, 16 threads. The 1-thread results are the same bits.

| System | Mode | $\lvert\Delta U\rvert$, kJ/mol (relative) | virial, relative difference | RMS $\lvert\Delta \mathbf F\rvert$ / RMS $\lvert\mathbf F\rvert$ | max $\lvert\Delta \mathbf F\rvert$, kJ/mol/nm |
|---|---|---|---|---|---|
| JAC | mixed | 9.8e-4 (3.2e-9) | 2.0e-8 | 2.6e-7 | 1.5e-3 |
| TIP3P 103k | mixed | 4.4e-2 (1.1e-8) | 5.1e-8 | 3.5e-7 | 1.7e-3 |
| Cellulose | mixed | 6.1e-3 (2.1e-9) | 3.0e-8 | 2.4e-7 | 2.2e-3 |
| JAC | double | 0 | 3.6e-16 | 5.2e-16 | 3.9e-12 |
| TIP3P 103k | double | 0 | 1.7e-16 | 7.3e-16 | 4.5e-12 |
| Cellulose | double | 0 | 3.7e-16 | 4.8e-16 | 4.1e-12 |

*The tolerances.* Reordering a sum of $n$ terms changes it by at most $2\gamma_{n-1}\sum_k \lvert t_k\rvert$, with $\gamma_m = m u/(1 - m u)$ [[Higham2002]](../references.md#higham2002). The terms of a row are not recorded, so the bound is applied through the scalar row's own rounding error, which the same sums carry.

- *Mixed mode.* The difference between the lanes and the scalar row must be smaller than the scalar row's own difference from the double result. It is 1.4% of it on JAC, 3.6% on TIP3P, and 0.3% on Cellulose (RMS forces). The energies differ by 9.8e-4 against 4.9e-2 kJ/mol, by 4.4e-2 against 3.4e-1, and by 6.1e-3 against 2.2e-2. The RMS force error against double is the same for both: 1.9570e-2 and 1.9570e-2 kJ/mol/nm on JAC, 9.1603e-3 and 9.1609e-3 on TIP3P, and 7.4986e-2 and 7.4986e-2 on Cellulose.
- *Double mode.* With rows of at most some 500 candidates, $2\gamma_{n}$ is 1.1e-13 relative to $\sum_k \lvert t_k\rvert$. The force differences are under 1e-15 of the RMS force. The double potentials agree to the bit, since the rows' differences lie below the total's last place.

**Against an independent oracle.** OpenMM 8.6.1, Reference platform, in double precision on JAC: the same Amber model, PME with MDIR's $\beta$ and the $64^3$ grid, and the dispersion correction. OpenMM's PME order is 5 and MDIR's is 4.

- RMS $\lvert\mathbf F_\text{MDIR} - \mathbf F_\text{OpenMM}\rvert$: 0.81193 kJ/mol/nm for both scalar and lanes in mixed precision, and 0.81182 for both in double. That is 7.9e-4 of the RMS force, 1030 kJ/mol/nm.
- Potential: −302545.45 (mixed scalar), −302545.45 (mixed lanes), and −302545.40 (double, both), against −302492.13 kJ/mol. The offset of 53 kJ/mol is the potential shift of MDIR's Coulomb cutoff and the order of the grid, the same for both rows.

The tolerance is that the lanes leave the agreement with the oracle unchanged: the RMS differences agree to 5 digits in each mode.

## Timing

FILLTIMING

## Left open

- Nothing for the maintainer. The deterministic mode on an unmeasured host was asked on PR #823 and ruled above (option 2).

**Not in this slice.**

- `cluster(M, N)` (#549, after this is measured);
- the two-ended form, and `centers`, `center_set`, `ghost_free`, and `images(explicit)`;
- `vector.gather` and relaid tables;
- tables in f32 on the host in mixed precision (a separate decision, the maintainer's ruling 3);
- the builds of the list (D267 step 3).
