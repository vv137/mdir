# D[gpu-target-from-device]

Recorded at its merge.

Forced PTX uses the detected Volta architecture when no architecture override
is present. The upstream portable PTX target requires compute capability 7.5.
The cubin path already compiles for the selected device. GPU execution tests
that call the upstream lowering directly query the first visible CUDA device
and pass its architecture to that lowering. Tests that execute TOWERS MMA
require compute capability 7.5 or later. The setup diagnostic for an explicit
TOWERS request also has a test on older devices.
