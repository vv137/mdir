# D[reciprocal-stencils]

Recorded at its merge.

**Introduce structured particle-grid stencils with explicit continuous operands as the first slice of the approved reciprocal-primitives design.**

Status: implementation draft, part of #452. The current checkpoint
provides the structural relation type and its constructor/verifier;
structured spread/gather, transposes, JVPs, and execution remain pending.
[The implementation record](../reciprocal-stencils.md) distinguishes those
parts and gives the preliminary precision table and remaining gates.

Positions and cell remain explicit. Integer support metadata does not
prove continuous operands inactive. No ordinary PME route or #778 forward
tangent changes. The latter's possible migration is decided by measurement
in slice 3. No performance conclusion is made at this checkpoint.
