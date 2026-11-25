# Ordinary-step graph experiments

`transaction.cu` exercises accepted-bank publication across a chunk boundary
and injects constraint, position and force-certificate failures. The GPU
lit test builds it against the production runtime.

`conditional.cu` measures the isolated cost of CUDA conditional nodes from
the prototype comparison. `gaps.py` analyzes ordinary-step idle time in an
exported trace. The benchmark modes in historical traces predate
`[execution] step_graph`. See D439 for their measured costs.
