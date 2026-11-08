# D[reciprocal-primitives-design]

Recorded at its merge.

**Propose a reciprocal chain whose structured particle-grid spread and gather declare their transposes, and whose window and influence kernels retain the position, charge, and cell derivatives needed by force matching.**

Status: proposed; awaiting a ruling on #452.

[The design](../reciprocal-primitives.md) composes those operations with
the existing semantic FFT and a spectral influence map and reduction.
It states the Hermitian pairing, unnormalized transforms, nonlinear
operand rules, charge-field force product, position Hessian-vector
product, site curvature boundary, execution effects, quantization
contract, and staged validation gates.

This is a documentation-only proposal. It adds no operation, lowering,
control-file key, Python API, output format, default, or numerical result.
The existing reciprocal route remains in use. Implementation and the
default-route decision wait for the maintainer's ruling and their gates.
Part of #452; C(ii) of #726 and HVP-3 of #771 depend on the declared
spread/gather rules.
