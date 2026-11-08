# D[reciprocal-primitives-design]

Recorded at its merge.

**The reciprocal chain uses structured particle-grid spread and gather with declared transposes, while its window and influence kernels retain the position, charge, and cell derivatives needed by force matching.**

Design approved by the [maintainer's ruling](https://github.com/vv137/mdir/pull/781#issuecomment-6716738603)
on 2026-11-08; implementation is staged under #452.

[The design](../reciprocal-primitives.md) composes those operations with
the existing semantic FFT and a spectral influence map and reduction.
It states the Hermitian pairing, unnormalized transforms, nonlinear
operand rules, charge-field force product, position Hessian-vector
product, site curvature boundary, execution effects, quantization
contract, and staged validation gates.

Every implementation slice gives a new-versus-legacy precision table
before any performance conclusion: positions as loaded, window weights,
assigned grid, FFT, influence, gathered potential, forces, and each
accumulator. Differences are stated and timings compare matching types.
#778's forward tangent retains its two reciprocal sums along the direction;
whether to migrate it to the JVP route is decided in slice 3 by measurement.

This is a documentation-only design. It adds no operation, lowering,
control-file key, Python API, output format, default, or numerical result.
The existing reciprocal route remains in use. Implementation follows the
approved slices. A default-route switch requires a separate ruling after
its gates.
Part of #452; C(ii) of #726 and HVP-3 of #771 depend on the declared
spread/gather rules.
