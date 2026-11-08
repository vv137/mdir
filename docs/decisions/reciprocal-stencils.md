# D[reciprocal-stencils]

Recorded at its merge.

**Introduce structured particle-grid stencils with explicit continuous operands as the first slice of the approved reciprocal-primitives design.**

Part of #452, slice 1 of the approved contracts. The structural relation
and structured spread/gather keep continuous inputs explicit. Linear
transposes and JVPs clone the same scalar window body; active positions
and cell receive named refusals until their composed rules are admitted.
The order-four scalar window has forward and reverse derivatives through
the existing scalar rules. [The implementation record](../reciprocal-stencils.md)
also records the ordinary evaluator's cell-activity guard: an unsupported
virial composition must not become zero when another position field is
used. The guard is limited to compositions containing the new primitives.
The record
states exact coverage, effects, bounds, the precision table against legacy
and the slice-specific gates. It makes no new-route GPU or performance claim.

Positions and cell remain explicit. Integer support metadata does not
prove continuous operands inactive. No ordinary PME route or #778 forward
tangent changes. The latter's possible migration is decided by measurement
in slice 3. No performance conclusion is made at this checkpoint.
