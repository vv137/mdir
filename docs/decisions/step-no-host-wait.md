# D[step-no-host-wait]

Recorded 2026-11-27.

**Recorded spans: the work that the host issues between two host boundaries is recorded and issued together, and a sequence that repeats is issued as a CUDA graph. `[execution] step_graph = "SPANS"`, the default on the GPU.** Issue #60 (CUDA graphs for the step program), after D439 (graph replay of ordinary TOWERS steps) and the parked PR #929.

## 1. What a span is

The device runtime (`runtime/mdrt_cuda.c`) issues every kernel of a program, every transform of cuFFT and every set of memory on one stream. A *host boundary* is a call of the runtime that the host must not pass before the device has reached it, or that issues work outside that stream:

- a wait: the wait of a flag (`mdrtFlagFinish`, D113, D118), a copy to or from the host (`mgpuMemcpy`, which waits), a wait of the stream (`finish`), a synchronization of the context (the growths of the matrix, the groups and the edges);
- a change of stream: the side stream of the reciprocal sum (`mdrtSideBegin`, D81), a borrowed stream (`mdrtUseStream`), the stream of a plugin (`mdrtDeviceExternalStream`), a hand-off (`mdrtDeviceHandOff`);
- work issued by other code on the stream: the staging of frames (D400), the TOWERS graphs of D439 (`runtime/mdrt_step_graph.cu`), the structures of the tuples, cuBLAS.

The ops between two boundaries form a *span*. With spans on, the runtime does not issue them as they come. It records each launch (function, grid, block, dynamic shared memory, cooperative attribute, cluster, and the bytes of its arguments, whose sizes it takes from `cuFuncGetParamInfo`), each transform (plan, input, output, kind), each set (address, value, count) and the event of each mapped flag, and issues the span at the next boundary, which calls `mdrtSpansFlush` before anything else.

The runtime keeps the graphs of spans by their *shape*, the ops without their argument bytes and grids:

- a span whose ops and argument bytes equal those of a graph it holds is issued as that graph (`cuGraphLaunch`);
- a span of a shape recorded before, of which fewer than 16 graphs are held, is captured on the stream (`cuStreamBeginCapture`) and instantiated;
- a span of a shape of which 16 graphs are held takes the graph of that shape used least recently: the kernel nodes whose arguments or grid differ are updated (`cuGraphExecKernelNodeSetParams`) and the graph is issued;
- a span seen for the first time, or with fewer than two kernels and transforms, is issued as recorded.

The updates are what makes the graphs repeat. Two steps of one run rarely issue the same argument bytes: the banks of the positions and forces alternate, the outer loops of the schedule permute them with other periods, the temporaries of a step move in the pool after an output, and a build changes the counts of a list. On TIP3P 10k a cache of exact matches made a new graph every five steps, and the instantiations cost more than the graphs saved (Section 5). With updates, a run of 40,000 steps made 239 graphs (Section 5). The cache holds at most 1,024 graphs; the one used least recently is released, after a wait for the stream, when it is full (`MDRT_SPANS_CACHE` sets a smaller room for the tests). A graph whose update the driver refuses is marked dead and never matched again; the log counts the refusals. When a program is unloaded, its graphs and the layouts of its kernels' arguments are released, since another program may later have kernels at the same handles.

The read of a mapped flag ends a span too (`mdrtFlagStart`), so that the device runs the work before the flag while the host records the work that it issues before the wait, the reciprocal sum of D179 among it. Its event is the place's own event, not one of a ring, so that the span repeats from step to step; a place has one read in flight (D118). In a capture the event is recorded as an external event node, which the host can wait for while the rest of the graph runs.

## 2. Why the results are those of the launches one by one

1. [derived] A span is issued in full before any boundary that follows it, and a boundary is a call that waits for the device or issues work on another stream. So every wait of the host still finds the device past the same work, and every other stream still waits for the same work, as without spans.
2. [derived] The ops of a span run on the one stream in the order in which they were recorded, as a graph made by capture of that stream or as launches. Kernels take their arguments by value; the recorded bytes are those the launch would have copied. So each kernel runs with the same arguments after the same kernels.
3. [derived] What the host writes between two ops of a span cannot reach a kernel of it differently than before: memory of the host that a kernel reads is written by the host only at a boundary (the clearing of a flag in `mdrtFlagFinish`, the copies), and a launch already issued would have read it at its execution, not at its issue, in either case.
4. [derived] A graph refers to memory only through the arguments of its kernel nodes, which equal the recorded ones byte for byte before it is issued (compared, or updated where they differ), and to the plans of cuFFT, which it holds (`graphUsers`) so that a trim or the budget of the plans (#622) does not destroy their workspaces under it.

So in the deterministic mode the checkpoint and every row of the log are bitwise those of `LAUNCH` (Section 4).

## 3. The key

| Value | Meaning |
|---|---|
| `"SPANS"` (default) | Recorded spans. Without the key on the CPU target, `LAUNCH`, a rule of the target recorded in `resolved_choices.step_graph`; given on the CPU target, refused (`SPANS requires the GPU target`). |
| `"LAUNCH"` | Each op issued as it comes. |
| `"GRAPH"` | The TOWERS chunks of D439, and spans for the other steps. |
| `"AUTO"` | As in D439; the setup trial runs both candidates without spans. |

`MDRT_SPANS=0` or `1` overrides the key, and `MDRT_WAIT`, `MDRT_PROFILE` and `MDRT_TRACE` turn spans off (`docs/environment.md`). A simulation of the Python interface refuses `SPANS`. The log ends with the counts of the spans: those issued as graphs and as recorded, the graphs made and released, the captures refused, the kernel nodes updated and the updates refused, and the kernels and transforms recorded.

## 4. Validation

`test/Driver/step-spans-gpu.test` (`Inputs/check_step_spans.py`) runs the dipeptide in water (1,168 particles, PME, SHAKE and SETTLE, V-rescale every 10 steps, 300 steps of 2 fs) on the GPU. The oracle is the same program submitted with `LAUNCH`:

- **Deterministic mode, MATRIX: bitwise.** The checkpoint (positions, velocities, forces, cell) and every row of the log are compared byte for byte, in mixed and double precision, NVT and NPT (stochastic cell rescaling every 10 steps). Each SPANS run made graphs and updated kernel nodes, with 1,006 (NVT) and 1,087 (NPT) graph launches.
- **The other configurations: bitwise.**
  - the default without the key (category `rule`);
  - spans that end only at the waits (`MDRT_SPANS_SPLIT=0`);
  - a cache of four graphs, which released 498 graphs;
  - `MDRT_SPANS=0`, which records nothing.
- **GROUPS** (dual list, not deterministic), 20 steps: the relative difference of positions and velocities from LAUNCH is at most ten times that of two LAUNCH runs (with a floor of $10^{-5}$).
- **Refusals.** `test/Driver/step-graph.test` checks the default on the CPU (`LAUNCH`, category `rule`, reason `SPANS requires the GPU target`) and the refusal of a given `SPANS` there by `mdir check` and `mdir run`. `test/Driver/step-graph-python.test` checks the refusal by a Python simulation.
- **D439 and the frames.** `step-graph-gpu.test`, `step-graph-certificate-gpu.test`, `frame-writer-gpu.test` and `frame-writer-fault-gpu.test` pass, with spans in the boundary steps of GRAPH and around the staging of frames.

**Mutations.** Each defect below was built into the runtime once, and the test failed for each.

| Defect | How the test failed |
|---|---|
| No flush in `mdrtFlagFinish` | The no-split case is not bitwise. The wait finds no record of its event, so the flag is read stale and a rebuild comes late. |
| No flush in `mgpuMemcpy` | The copy runs ahead of the recorded kernels: the run stops, with 140 positions not numbers at a build. |
| The first changed kernel node not updated | The run stops at step 4 with a failure of SETTLE. |

In the default configuration, without the flush in `mdrtFlagFinish` the results stay bitwise, because the span that holds the flag's event was issued at `mdrtFlagStart`. The flush in `finish` has the same redundancy: every copy flushes before it issues. Both flushes are kept as the rule for every boundary.

**The event of a flag in a graph.** A host wait for an event recorded by an external event node of a graph must wait for the launch just issued, and must not wait for the work after the node. A probe on an RTX 4090 tested this (`scripts/benchmarks/step-graph/event-node.cu`). The graph was a kernel that spins 2 million clock cycles (about 0.7 ms) and then stores the launch number into mapped memory of the host, then the event node, then a kernel of 4 million cycles. Over 200 launches, every wait read the number of its own launch, and no wait found the tail done.

## 5. Measurements

**What a graph saves** (RTX 4090, CUDA 13.2, `segbench`, kernels of 1 to 3 µs with the host ahead):
- a graph of 8 to 24 dependent kernels runs each 0.42 to 0.48 µs faster than stream launches, and a graph of 4 kernels 0.31 µs faster;
- `cudaGraphLaunch` costs the host 1.6 µs, against 1.9 µs for a stream launch;
- `cudaGraphExecKernelNodeSetParams` costs 0.27 µs.

**Conditional nodes.** An IF node (`scripts/benchmarks/step-graph/conditional.cu`) adds 2.5 µs to its site when its body is skipped and 5.1 µs when it runs, against static nodes with the same kernels. A rebuild behind an IF node in every step would cost that every step. That is a fifth of the 10 to 13 µs a step that graphs remove at these sizes (#60). Spans place the rebuild outside the graphs instead.

**Three designs that were measured and dropped.**
1. *Exact matches only, 256 graphs.* On TIP3P 10k the cache thrashed: the run made 1,220 graphs and released 964 in 4,000 steps, and gained about 1%.
2. *Exact matches only, 4,096 graphs.* The temporaries of a step move in the pool after each output, so a new set of spans appeared every 1,000 steps. The run made 2,101 graphs in 10,000 steps, and their instantiations made it 10 to 25% slower than LAUNCH.
3. *Spans that end only at waits* (`MDRT_SPANS_SPLIT=0`). The event of a flag is then issued at the wait, so the device idles while the host decides. TIP3P 10k ran at 780 ns/day against 1,741 with LAUNCH.

The design of Section 1, with 16 graphs a shape, made 239 graphs in the 40,000 steps of TIP3P 10k below and updated 2.3 kernel nodes a step. It issued 76% of the spans as graphs; the others were spans of one kernel or seen once. JAC (GROUPS) made 134 graphs and issued 60% of its spans as graphs. A build of the dual list changes a count that one refresh kernel takes, so the spans around it repeat less.

**Throughput.** RTX 4090, 2 fs, mixed precision, the settings of the Amber suite (`mdir-suite.toml`: NVE, or NPT with V-rescale and stochastic cell rescaling every 25 steps), and TIP3P boxes with V-rescale every 10 steps. main is 03506508. The branch was measured at fc006c8b on the same main; the later commits add the count of refused updates and the release of the graphs when a program is unloaded. Each system ran main, the branch, the branch with `MDRT_SPANS=0` twice, the branch, and main. Two jobs reversed the order of main and the branch. The rate is the 10%-trimmed mean of the intervals between rows after the stationary cut of D417's protocol (`MDIR_CHECKPOINT_TIMES=1`). Every run was stationary but one (main, TIP3P 10k, forward job), and the spread of the intervals was 0.1 to 0.7 µs a step. The median SM clocks under load were 2,775 and 2,715 MHz (range 2,520 to 2,790), at most 65 °C. STMV is in the suite only at 4 fs and is left out.

| System | main ns/day (forward / reverse) | SPANS ns/day | branch LAUNCH ns/day | change (forward / reverse) |
|---|---:|---:|---:|---:|
| TIP3P 10k (MATRIX) | 1703 / 1677 | 1899 / 1840 | 1706 / 1672 | +11.5% / +9.8% |
| TIP3P 20k (MATRIX) | 1303 / 1272 | 1400 / 1377 | 1304 / 1269 | +7.5% / +8.2% |
| JAC NVE (GROUPS) | 1287 / 1252 | 1393 / 1360 | 1282 / 1258 | +8.2% / +8.6% |
| JAC NPT (GROUPS) | 1175 / 1145 | 1239 / 1214 | 1174 / 1146 | +5.4% / +6.0% |
| Factor IX NVE (GROUPS) | 582 / 570 | 600 / 591 | 580 / 570 | +3.1% / +3.6% |
| Factor IX NPT (GROUPS) | 542 / 531 | 557 / 547 | 541 / 533 | +2.8% / +3.0% |
| Cellulose NVE (GROUPS) | 153 / 150 | 154 / 151 | 152 / 149 | +0.9% / +0.8% |
| Cellulose NPT (GROUPS) | 146 / 143 | 147 / 144 | 146 / 143 | +0.8% / +0.4% |
| TIP3P 10k, TOWERS | 2118 / 2066 | 2412 / 2364 | 2112 / 2081 | +13.9% / +14.4% |
| TIP3P 20k, TOWERS | 1553 / 1521 | 1727 / 1691 | 1550 / 1525 | +11.2% / +11.2% |
| JAC NVT, TOWERS | 1575 / 1545 | 1750 / 1705 | 1577 / 1538 | +11.1% / +10.4% |
| TIP3P 10k, TOWERS, `GRAPH` | 2195 / 2160 | 2221 / 2192 | 2195 / 2168 | +1.2% / +1.5% |
| TIP3P 20k, TOWERS, `GRAPH` | 1593 / 1565 | 1612 / 1579 | 1591 / 1565 | +1.2% / +0.9% |
| JAC NVT, TOWERS, `GRAPH` | 1590 / 1561 | 1605 / 1573 | 1593 / 1563 | +0.9% / +0.8% |

1. [measured] No system is slower. The branch with LAUNCH runs at the rate of main, within the spread between the two jobs.
2. [measured] The gain falls with the size of the system, from 10 to 14% at 10,000 particles to under 1% for Cellulose. In the TOWERS rows, `GRAPH` replays ordinary chunks of D439, and SPANS outside them adds 0.8 to 1.5%.
3. [measured] On TOWERS, SPANS without D439's chunks is faster than `GRAPH` by 7 to 9%: 2,412 against 2,221 ns/day on TIP3P 10k, and 1,750 against 1,605 on JAC.
4. [hypothesis] The chunks of D439 drain the stream to read their progress after each chunk and before each boundary. On main's traces of JAC with node tracing, `GRAPH` waits for the stream 0.31 times a step (0.21 of them in `mdrtStepRun` and `mdrtStepRead`), and TOWERS without it 0.10 times. Each such wait leaves the device idle until the host issues again. The traces do not separate this from the cost of the chunks' guards.
5. [hypothesis] The spans' graphs remove the gaps between kernels within a span, and the host time of the launches. Under nsys with node tracing (inflated), TIP3P 10k went from 21.4 stream launches and 29.5 µs of device idle a step to 0.12 stream launches, 3.25 graph launches and 15.8 µs.

## 6. What remains on the host

Spans overlap the host's decisions but do not remove them: the host still reads what the device computed and decides. Counts per step on main, from nsys over 1,500 steady steps (NVT, V-rescale every 10 steps, energies every 1,000):

| Path | Waits of a flag (D113, D118) | Waits of the stream | Reads to the host |
|---|---:|---:|---:|
| TIP3P 10k, MATRIX | 2.00 | 0.25 | 8 B at each build (0.14); 8 kB of the reductions at each coupling step (0.10) |
| JAC, GROUPS dual list | 3.00 | 0.21 | 20 B at each build (0.11); 8 kB at each coupling step (0.10) |
| TIP3P 10k and JAC, TOWERS | 1.10 | 0.10 | 8 kB at each coupling step (0.10) |

The flags are the tests of validity, pruning and the constraints. Each ends a span, and the event of each is in a graph that the device runs while the host records the next work. A wait of the stream drains the device: a build reads the sizes that set the grids of the pair kernels, and a coupling step reads the kinetic energy for the factor of V-rescale. The next decisions to move to the device are these two: grids from the room of the structure, with the growth read late, as #929 began for the matrix; and the factor of V-rescale from the kinetic energy on the device, with the draws of the host.
