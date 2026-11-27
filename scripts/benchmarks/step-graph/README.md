# Ordinary-step graph experiments

`transaction.cu` exercises accepted-bank publication across a chunk boundary
and injects constraint, position and force-certificate failures. The GPU
lit test builds it against the production runtime.

`conditional.cu` measures the isolated cost of CUDA conditional nodes from
the prototype comparison. `gaps.py` analyzes ordinary-step idle time in an
exported trace. The benchmark modes in historical traces predate
`[execution] step_graph`. See D439 for their measured costs.

`event-node.cu` checks that a host wait for an event recorded by an
external event node of a graph waits for the launch just issued and not
for the work after the node, which the recorded spans of
D[step-no-host-wait] rely on for the flags of the tests. Build it with
`nvcc -arch=sm_89 event-node.cu -lcuda`; it prints the number of stale
reads and exits with 1 if there is one.
