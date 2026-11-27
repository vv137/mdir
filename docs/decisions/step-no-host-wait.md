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

The updates are what makes the graphs repeat. Two steps of one run rarely issue the same argument bytes: the banks of the positions and forces alternate, the outer loops of the schedule permute them with other periods, the temporaries of a step move in the pool after an output, and a build changes the counts of a list. On TIP3P 10k a cache of exact matches made a new graph every five steps, and the instantiations cost more than the graphs saved (Section 5). With updates, a run of 30,000 steps made 72 graphs. The cache holds at most 1,024 graphs; the one used least recently is released, after a wait for the stream, when it is full (`MDRT_SPANS_CACHE` sets a smaller room for the tests). A graph whose update the driver refuses is never matched again.

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

`MDRT_SPANS=0` or `1` overrides the key, and `MDRT_WAIT`, `MDRT_PROFILE` and `MDRT_TRACE` turn spans off (`docs/environment.md`). A simulation of the Python interface refuses `SPANS`. The log ends with the counts of the spans: those issued as graphs and as recorded, the graphs made and released, the captures refused, and the kernels and transforms recorded.

## 4. Validation

PENDING.

## 5. Measurements

PENDING.

## 6. What remains on the host

PENDING.
