# D[towers-graph-npt]

Recorded 2026-11-28.

**Graph replay of ordinary TOWERS steps (D439, `step_graph = "GRAPH"`) under stochastic cell rescaling. The kernels of a chunk read the arguments that the cell sets from a device mirror, which the host fills before each chunk.** Issue #1017.

## 1. What changes

`[execution] step_graph = "GRAPH"` no longer refuses a barostat. Graph replay covers mixed-precision, constrained TOWERS dynamics with Gaussian PME, velocity Verlet, V-rescale and, now, `[barostat] method = "C-RESCALE"` in each coupling (`ISOTROPIC`, `SEMI_ISOTROPIC`, `ANISOTROPIC`), with each `work`, in a rectangular or a triclinic cell. `AUTO` runs its setup trial under a barostat as well. No key is added. The other refusals of D439 stay (deterministic execution, which TOWERS itself refuses; double precision; other integrators, thermostats and electrostatics).

## 2. Why a chunk needs the cell from the device

1. [derived] The barostat acts at the end of the step that completes a coupling period, the step at which the thermostat acts (D72). The ordinary steps between two coupling steps run at one cell. The chunk of D439 replaces that loop of ordinary steps with a graph captured once and replayed while its banks and buffers are the same.
2. [derived] Before the loop the host loads the cell from its own memory (the edges, and the tilts of a triclinic cell). Inside the loop it derives launch arguments from them with pure arithmetic, for example the reciprocal edges of the PME wrappers and the edges of the TOWERS moving rows, and passes them to the kernels by value. A kernel in a captured graph keeps the argument bytes of its capture. With a fixed cell these are the right bytes at every replay; under a barostat a replay would use the cell of the capture.
3. [derived] The lowering of a chunk (`lib/Conversion/GPUToBinary/TowersChunk.cpp`) therefore marks as *varying* every value that a load of such host memory reaches through pure arithmetic. The memory must be a small array of static shape, defined before the chunk and not written by it. Each varying value that a captured kernel takes from above gets slots in one device mirror of 256 words. Each kernel of the chunk loads its slots instead. A PME wrapper that takes a varying argument gets a variant whose kernels do the same. Before each chunk, the host recomputes every slot from its memory at the chunk site, with the same operations the step's launches use, and writes the words into the mirror. The write is one launch with the words by value on the stream of the graph, ahead of the graph's launch. Words that did not change since the last chunk issue nothing.
4. [derived] So each kernel of a replayed chunk reads the bytes that its launch would have received from the host at that cell. This is the argument of D448 Section 2 applied to the varying arguments, with the mirror write placed before the graph on the same stream.
5. [derived] The lowering refuses, when it builds the program, every other use of a varying value inside a chunk: a launch dimension, a call of the host, an argument of an outlined TOWERS kernel, or a value it cannot recompute at the chunk site. Each such use would replay its value from the capture. The TOWERS kernels take only device memory; their cell enters through the moving-row kernels, which now read the mirror.

## 3. The questions of the issue

### 3.1 Is the barostat step a host boundary?

Yes, as before, in every submission path. The coupling step runs on the host's launch path, between chunks. Counts from Nsight Systems over the steady part of 4,000-step JAC runs (RTX 4090, TOWERS, coupling every 25 steps), per coupling period:

| Run | Copies to the host (bytes: count) | Waits of the stream |
|---|---|---:|
| NVT, GRAPH | 8,192: 1.0; 8: 1.0; 48: 1.0 | 1.0 (copy) + 2.0 (chunk progress and step check) |
| NPT, GRAPH | 8,192: 2.0; 24: 1.0; 56: 1.0; 48: 1.0; 8: 0.5 | 4.0 (copies) + 1.6 (chunk progress and step check) |
| NPT, SPANS | 8,192: 2.0; 24: 1.0; 56: 1.0 | 4.0 (copies) |

1. [measured] The barostat adds three copies to the host, each with a wait of the stream, to the one of a V-rescale step. The second 8 kB copy is the reduction of the step after the scaling. The 24 B copy is the TOWERS virial of the step before the scaling, which the dispatch copies to the host apart from the reduction.
2. [derived] The factor of V-rescale needs the kinetic energy of that step on the host (`mdrtBussiFactor`), and the barostat acts at the same step. Moving only the strain of the barostat to the device therefore leaves the step a host boundary. It also leaves the step's own launches, which take the new cell by value from the host, and the Trotter bath of the conserved energy, which the host sums. This decision leaves the coupling step as it is (Section 6).
3. [derived] What the chunks needed from the host was the cell's arguments by value. After this decision the chunk kernels read them from device memory. A coupling step on the device could therefore write the mirror itself.

### 3.2 Does the TOWERS grid change with the cell?

No, by construction. The bins per axis, the tower height, the capacity and the launch shape are fixed when the program is built (`mdir.towers_grid`, `mdir.towers_launch`). With a barostat, setup reserves one bin per axis where the periodic stencil allows (`TowersCell::bins`). The force kernel checks at every evaluation that the faces of the current cell still cover the cutoff (`validWidths`). A cell that fails stops the run. Each evaluation rebuilds the cells from the current cell, so no list outlives a scaling.

Measured on the suite's NPT inputs with LAUNCH, a row at every scaling. JAC and TIP3P 10k ran 100,000 steps, Factor IX 50,000. The bins a setup would choose were recomputed offline for the cell of each row (`scratch` analysis of the volume rows; isotropic coupling):

| Input | Scalings | Bins of the run (x, y, z) | Bins a fresh setup would choose, without the reserve | Changes, with / without the reserve | Volume change (min, max) | Face width / (cutoff / reach), min–max | Linear shrink to an invalid face |
|---|---:|---|---|---|---|---|---:|
| JAC (HALF) | 4,000 | 6, 6, 14 | 7, 7, 15 | 0 / 0 | −3.5%, 0 | 1.089–1.286 | 9.3% |
| Factor IX (FULL) | 2,000 | 16, 9, 8 | 17, 10, 9 | 0 / 0 | −2.5%, 0 | 1.092–1.220 | 9.2% |
| TIP3P 10k (HALF) | 4,000 | 4, 4, 9 | 5, 5, 10 | 0 / 0 | −1.8%, +1.2% | 1.135–1.301 | 12.4% |

1. [measured] In 10,000 scalings no grid that setup would choose changed, with or without the reserve. The grid of the run is fixed and stays valid with at least 8.9% of face width to spare.
2. [computed] A grid sized for the largest cell with masked cells, or an update of the graph's kernel nodes when the counts change, is not needed. This decision keeps the fixed grid.
3. [measured] The largest cell population, binned offline from frames every 5,000 steps, grew from 60 to 64 (JAC), 96 to 103 (Factor IX) and 82 to 90 (TIP3P 10k). The capacities of the runs, rounded up to a multiple of 8, were 64, 96 and 88. Members beyond the capacity go to the overflow list of the TOWERS build (`TowersBuild.cpp`) in every submission path. This concerns TOWERS under a barostat, not graph replay.

### 3.3 The screen and the fixed-point certificate (D428–D432)

[derived] The coordinate scale of the screen, the filter radius and the limit of the fixed-point words are not constants of the program. `towers_prepare_force` computes them at every evaluation from the current face widths and steps (`screenConstants` in `TowersForce.cpp`). An invalid geometry raises the fatal word, and the guarded kick of the chunk does not run. Under a barostat the moving rows that feed that kernel read the current cell from the mirror. The certificate is therefore recomputed on the device for each scaled cell already. No re-certification is added. The test of Section 5 compresses a cell until a face no longer covers the cutoff. The run stops at the step of the host's launches that follows the scaling, before a chunk takes the cell, in GRAPH as in LAUNCH.

### 3.4 Bank phases, the accepted-step counter and a retry

1. [derived] The scaling happens outside the chunks, on the host's launch path. A chunk starts from the banks that the host selected after the previous chunk, and the accepted-step counter advances at each step of the launch path (`mdrtStepBoundary`), the steps before and after a scaling included. The test checks the step and time of the checkpoint byte for byte, and the steps of every row.
2. [measured] A retry after a rejected proposal replays one step from the accepted banks with the mirror of that chunk. The transaction harness (`scripts/benchmarks/step-graph/transaction.cu`, mode 5) drifts by the mirrored cell. It changes the cell between two chunks and rejects the first proposal after the change once. The chunk ends at value 38 = 2 · 1 + 18 · 2 with 21 force calls and 20 kicks. The replayed step therefore read the new cell.

## 4. Number of graphs

The chunk graphs are keyed by their banks and buffers (D439). The barostat does not add keys: the mirror is one buffer per chunk site.

1. [measured] On 100,000 steps of JAC with coupling every 25 steps and a row every 1,000, NPT made 30 chunk graphs in 4,000 chunks and 7 proposal banks (2.7 MiB). With the coupling every 10 steps of the NVT input of the same system, GRAPH made 84 graphs in 10,000 chunks and 12 proposal banks (5.4 MiB).
2. [computed] The proposal banks are allocated per input bank address (`partner` in `runtime/mdrt_step_graph.cu`). Their fixed count shows that the keys are permutations of a fixed set of banks, as in D439, and that the number of graphs is bounded by the number of these permutations.

## 5. Validation

`test/Driver/step-graph-npt-gpu.test` (`Inputs/check_step_graph_npt.py`) runs the dipeptide in water (1,168 particles, PME, SHAKE and SETTLE, V-rescale and C-rescale every 10 steps, 1 fs). The barostat targets 500 bar with a time constant of 0.1 ps, so the volume falls by about 6% in 200 steps. TOWERS has no deterministic mode. The oracle is LAUNCH, with the bound of D448 for GROUPS: the relative L2 difference of the positions and velocities of GRAPH from LAUNCH is at most ten times that of two LAUNCH runs, with a floor of $10^{-5}$. The edges differ by at most ten times the LAUNCH spread, with a floor of $10^{-6}$ nm. Every row has the same step and time, and its values differ by at most ten times the LAUNCH spread plus $0.05 + 10^{-4}\lvert x\rvert$.

| Case | Quantity | GRAPH − LAUNCH | LAUNCH − LAUNCH | Bound |
|---|---|---:|---:|---:|
| Isotropic, 200 steps | positions / velocities | (filled at the final head) | | |

- **A stale cell.** With `MDRT_STEP_MIRROR_FAULT=1` the runtime writes the mirror once, as a graph that took the cell by value would behave. The test requires that run to differ from LAUNCH by more than 100 times the bound. It does, by four orders of magnitude.
- **A continuation** from the GRAPH checkpoint at step 200, 100 steps more with a row every 20 steps: the same bound against two LAUNCH continuations, and the rows at steps 220 to 300. (A row interval that does not divide the restart step is the subject of #1014.)
- **Semi-isotropic and anisotropic coupling, and a triclinic cell** (the dipeptide's cell tilted to γ = 88°), 100 steps each, with the same bound.
- **A crushed cell** (50,000 bar): LAUNCH and GRAPH stop at the same kernel after the same rows.
- **Refusals.** No barostat case remains refused. `test/Driver/step-graph.test` (`check_step_graph_scope.py`) checks that `mdir check` keeps GRAPH, and leaves AUTO to its trial, in the three couplings.
- **The suite's inputs** (one-off, not in the suite): JAC, Factor IX and TIP3P 10k NPT with the launch shapes of the timing below cover tower heights FULL and HALF, split 2 and two blocks per SM. In 100 steps with a row at every scaling, GRAPH stayed within the LAUNCH spread in every row (filled at the final head).

## 6. What remains

- The coupling step stays a host boundary (Section 3.1). A coupling step on the device would compute the V-rescale factor, the strain, the scaled cell and the bath there and write the mirror. It would read back to the host only at outputs. That is a decision of its own; the counter-based draws of both factors are already functions of the step.
- The TOWERS virial of the step before a scaling is copied to the host apart from the step's reduction (the 24 B copy of Section 3.1).
