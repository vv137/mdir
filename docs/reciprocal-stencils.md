# Particle-grid stencil primitives: slice 1

Status: implementation draft, part of #452. The approved contracts and
the two additional requirements are recorded in
[the ruling on #781](https://github.com/vv137/mdir/pull/781#issuecomment-6716738603).

## Current checkpoint

The first checkpoint introduces the structural relation type and
**md.grid_stencil**, with positions and cell as explicit operands. It
verifies the particle-set identity, three grid extents, checked point
count, B-spline window, and support order. The relation contains indices,
not active window values. A consumer must retain its continuous operands
and differentiate the weight kernel separately.

The initial relation supports B-splines of order 2 through 8, with each
extent at least the support order and at most the signed 32-bit limit.
This bound keeps a particle's wrapped support indices distinct; the
product must fit signed i64. Curvature starts with order four as ruled.
Unknown windows and incompatible shapes are diagnostics.

This checkpoint does not yet execute the relation. Structured
**md.scatter_stencil** and **md.gather_stencil**, their transposes and JVPs,
and the opt-in execution lowering remain to be implemented in this PR.
No ordinary PME builder emits the new operation. The reciprocal sum,
FFT path, default strategies, and #778's two-sum forward tangent are
unchanged. Migration of that tangent is a measured decision in slice 3.

## Precision record

This is a semantic-IR checkpoint, with no new-route execution or timing.
The table separates known semantic types from unimplemented operations;
it is not an execution-precision or performance claim. Before a performance
conclusion, replace pending entries with the audited new-versus-legacy
execution types, for both Mixed and Double, as the ruling requires.

| Operation | Legacy semantic contract | New route at this checkpoint |
|---|---|---|
| Positions | Three-component f64 reference field; execution assigns its policy later | Three-component f64 operand, verified; no device load yet |
| Relation indices | Internal grid/support indices | Structural metadata only; extents checked against i32 and product against i64 |
| Window weights | Reference arithmetic, with execution precision assigned later | Not implemented |
| Assigned grid | Existing spreading strategies | Not implemented |
| FFT | Existing f32/complex-f32 or f64/complex-f64 transform contract | No new FFT route |
| Influence | Existing convolution strategy | Not implemented |
| Gathered potential | Existing gather strategy | Not implemented |
| Forces | Existing reciprocal result and derivative rules | Not implemented |
| Accumulators | Existing floating or deterministic fixed-point grid sums, and energy/derivative reductions | No new accumulator |

## Remaining work and gates

1. Define the structured scatter/gather kernel arguments and exact
   coverage. Generate both operations from one window body, keeping
   positions and cell dependencies explicit beside structural indices.
2. Register their amplitude/grid transposes and continuous-operand JVPs
   and VJPs. Unsupported active operands receive named diagnostics;
   structural indices never prove position or cell independence.
3. Add explicit, opt-in execution and effects/algebra declarations before
   any fusion or placement. Ordinary reciprocal execution stays unchanged.
4. Check independent stencil values, transpose identities, JVP/VJP
   duality, signed and zero amplitudes, wrapping and support boundaries,
   shapes and particle sets, and the required refusals. Differentiate the
   same scalar window body rather than maintaining a second formula.
5. Complete the precision audit and the approved CPU/GPU,
   Mixed/Double, deterministic/ordinary, corpus, order, checkpoint, suite,
   and device-code sanitizer gates appropriate to the completed slice.
   Numerical thresholds are recorded before first comparison. The draft
   is not ready until these gates and the remaining implementation pass.

The decision to move #778 onto a JVP route is not work of this slice.
