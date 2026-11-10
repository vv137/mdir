# D[host-pair-vector]

Recorded at its merge.

**On the CPU target, a loop over the pairs of the neighbor matrix with the policy `(directed, owner_only)` may run each row $W$ candidates at a time in registers of MLIR's `vector` dialect, the strategy `lanes(W)`, which `md-exec-choose-host-pairs` states on the op where one shared precondition holds and the host's measured policy selects it, and which the host lowering builds with one shared emitter; every other loop, and every loop on an unmeasured host, keeps the scalar row of today.** Issue #554, under #549 (step 4 of D267) and #444. *This file is the design of the draft PR; the measurements and the validation are filled in as the implementation lands.*

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

## Design of the first slice

**Strategy attribute.** An optional attribute of `md_exec.pair_for`, `host(lanes(W))`, printed only where it is set, so that every program without it, and every GPU program, prints as today. `md-exec-choose-host-pairs` sets it on the CPU target only, after `md-exec-assign-algebra` and before `md-exec-assign-storage`, which carries it (as `md-exec-choose-scatter` precedes the storage on a device). Absent, the row is scalar.

**One precondition** (`md_exec::hostpairs::checkLanes(PairForOp, int64_t W) -> Refusal`, `include/mdir/Dialect/MDExec/HostPairs.h`), called by the pass and again by `convert-md-exec-to-loops`, which fails on an attribute that it refuses (principle 2):

1. the policy is `(directed, owner_only)`, with no `centers`, `center_set`, `ghost_free`, or `images(explicit)` (the two-ended form adds at neighbors; the others are outside the survey and come later);
2. the kernel is one block and every op in it is in a row of the table above (refused: regions, calls, writes, an op without a mapping, `fpowi` whose exponent is not a constant);
3. its computed type is f32 or f64 and its values are scalars or vectors of one dimension;
4. every reduced value is `any_order`, `any_order_exact`, or `fixed_order`: the lanes order of the row is a function of $W$ alone, which the attribute states (below);
5. $W$ is a power of two from 1 to 64.

With `explain`, the pass remarks the choice, or the first precondition that fails, or the policy's reason (the host is not measured; the switch).

**One emitter** (`kernels::emitPairRowLanes`, `lib/Conversion/MDExecKernels/HostPairLanes.cpp`, header `include/mdir/Conversion/HostPairEmitter.h`), which `lowerPairFor` of the host calls in place of `emitPairKernel` where the attribute is set. `emitPairKernel` is not changed, so the device lowering, which also calls it, prints the same bytes, as does the scalar CPU row. No intrinsic of a host is written: the code is `vector`, `arith`, `math`, and `scf`, vectorized by MLIR's lowering to LLVM.

- *Groups of a row.* The row's $n$ candidates in $\lfloor n/W \rfloor$ full groups read by one `vector.load` of the matrix row, then one tail read by `vector.maskedload`. A padding lane takes the row's last candidate, so every lane evaluates the kernel at a pair that the scalar row also evaluates; its contribution is masked.
- *Gathers.* The positions, the `ins` fields, the rows of the radial tables, and the entries of `md.lookup`, one load a lane, then transposed into components. `vector.gather` and a relaid table are later strategies, chosen by measurement (#419's host counterpart).
- *The pair.* The difference of the positions, the minimum image (rectangular and triclinic, the same formula lane by lane), $r^2$ as $((0 + d_x^2) + d_y^2) + d_z^2$ in the order of the scalar `vector.reduction`, the kernel's ops by the table, and the mask $(\text{lane} < n) \wedge (r^2 < r_c^2) \wedge (j \ne i)$, with the scalar row's $r_c^2$ (`createCutoff2`).
- *Reduction order.* Lane $l$ adds the candidates $k \equiv l \pmod W$ in increasing $k$, in the type of today's row sum (a global sum the kernel widens is summed narrow and widened once per row, as today); then the $W$ lanes are combined by halving, lane $l$ with lane $l + W/2$, then $W/4$, down to one, a host lane reduction beside the device's in `ReductionEmitter.h`. The destination is then added or stored as today, and the global sums go to the fixed-order reductions of D171. A row's result is thus a function of its list, $W$, and the kernel alone: it differs from the scalar row by the rounding of the sum (a bound of Higham's form, Section 5.1 of device-primitives.md), and it does not depend on the number of threads or on the run. On a device the lanes of a row already add lane-strided and combine by D301's butterfly, under `fixed_order` as well.
- *Precision and accumulator.* No accumulator changes type (D301's guard): only the order changes.

**$W$ from the target.** The capability is the width of the host's vector registers, from the CPU features that LLVM detects for the JIT (`avx512f`: 512 bits; `avx`/`avx2`: 256; otherwise the baseline 128, SSE2 or NEON); $W$ is that width over the bits of the kernel's computed type (8 or 16 in f32, 4 or 8 in f64). The policy is a host column of #433's table, keyed by the CPU that LLVM names (`znver3` for the EPYC 7543, `znver4` for the EPYC 9354 here), with the measured $W$ and default; a host without an entry takes the scalar row and the remark says so (the documented conservative default). The host object of the compile cache is already keyed by the CPU and its features (docs/compile-cache.md), and the module, which holds $W$.

**Switches.** `MDIR_HOST_PAIRS=scalar|lanes|lanes:W` (`docs/environment.md`) overrides the policy, for the comparisons; the pass's options `mode` and `width` for tests. No control-file key, format, or default output changes; the default on a measured host changes only where the measurement wins beyond the spread of the runs (#549's rule).

**Validation plan.**

1. Identity: `check.py compare` of `main` and the branch, `--stages lowered`, on the whole corpus for the GPU target (byte-identical), and for the CPU target with `MDIR_HOST_PAIRS=scalar` (byte-identical).
2. Lane arithmetic: `lanes(1)` against scalar, bitwise, in mixed and double, deterministic mode, on JAC; and `lanes(W)` against scalar, bitwise, on a dilute system whose rows hold one pair within the cutoff (a sum of one term and zeros), so that any lane-wise difference of an op is found.
3. Lit tests of the IR: the attribute and each refusal through `explain`; the attribute survives `md-exec-assign-storage`; the emitted groups, tail, gathers, and halving tree; the lowering's failure on a refused attribute.
4. Numbers: `lanes(W)` against scalar on JAC, TIP3P 103k, and Cellulose at step 0 (energy, virial, forces), mixed and double, 1 and 16 threads, with tolerances from the summation bound; against an independent oracle, OpenMM 8.6.1 (CUDA, double) on JAC's energy and forces, both modes.
5. Determinism: the deterministic mode with `lanes(W)` gives the same bits at 1, 2, 4, 7, and 16 threads and on two runs (as `pme-threads.test`).
6. Timing: the pair loop's share and the whole step, `main` against the branch interleaved in each job (`bench.py --builds`), at least two jobs per system, per CPU model; the Amber suite's CPU rates; peak memory.

**Not in this slice.** `cluster(M, N)` (#549, after this is measured); the two-ended form and `centers`, `center_set`, `ghost_free`, `images(explicit)`; `vector.gather` and relaid tables; tables in f32 on the host in mixed precision (a separate decision, bitwise neutral if the f32 table is the rounded f64 one); the builds of the list (D267 step 3).
