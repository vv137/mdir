# D[builder-native-program]

Recorded at its merge.

**The builder constructs a program from owned native IR; only user input is parsed at its boundary.**

This is stage C of #568, after the complete term adapter registry and D363's
typed term fragments. The construction contract and current status belong in
[driver Section 2.12](../driver.md#212-the-files-of-the-builder).

## Ownership and construction

Stage C (#718, D[builder-native-program]) constructs the complete program in
one owned native module. `TermIRPlan` owns its context and module; adapters and
parsed model input share that context and remain alive for the full build.
Factories insert declarations, potentials, dynamics programs and entry functions
directly in their original order. `serializeNativeProgram` prints the completed
module once for the existing `Program.module` consumer handoff. Construction
never parses generated operation or type syntax, serializes temporary
connections, rewrites SSA tokens, or repairs loop operands after construction.
The default printer supplies SSA names without builder name hints.

Only user-written custom expressions and staged `ModelTerm` input are parsed
at their boundaries. `StagedModelIR` checks signatures, region shape, block
arguments and returns before typed access, then clones declarations and bodies
with `IRMapping`. Owned buffers preserve resource lifetimes. User string
attributes and diagnostic locations are retained, and duplicate declarations
are not silently renamed. Consumer parsing for display, verification and
lowering remains at its existing boundary. Construction and consumers share
`insertProgramDialects`, including tensor/linalg stage dialects.

All term kinds are native: externals, pulls, restraints, Born with surface,
custom tuples/centroids/compound terms, custom pairs, triplets, staged/opaque
models, every bonded family, and nonbonded LJ/Coulomb including reciprocal,
excluded and within-selection contributions. Shared potential sites, explicit
contribution ranks, field/potential/output order, inlining and fusion remain
unchanged. Narrow hooks receive actual typed operands and return actual typed
results; adapters do not retain names of intermediate SSA values.

Native entry factories cover minimization, dynamics, frames and programs of
parts, including orthorhombic/triclinic cells, supported integrators, virtual
sites, SETTLE/SHAKE, restraint and hosted-model completion, timed terms,
checkpoint restart, observables, free energy, gradients and pull output.
`EntryLoopIR` carries actual block arguments and yielded values; the host
boundary remains in place and loads the original part counts. Minimization
retains accepted/rejected controller state and first-call projection. Frame
checks inspect actual evaluation operations and operand provenance.

COM and thermostat coupling, FIRST_ORDER/EXACT barostat coupling and both
Trotter work modes share typed physical cores. Multistep Trotter periods retain
both closing steps. Leapfrog coupling uses full-time velocities and restores
the stored half step. Period-one Trotter keeps nine pressure-state values and
separate fresh, stored-state and missing-state restart paths. Missing-state
initialization retains atomic kinetic energy without COM removal; fresh
initialization retains group kinetic energy. Existing unsupported period-one
programs of parts retain their diagnostic.

Exact trial virtual-site placement and force spreading deliberately retain the
archived entry cell (#722); trial energy evaluation and other gathers retain
their original cells. Plain energy rows construct no unused virial diagonal;
construction tests check their result-producing operations directly, without a
cleanup pass. Native tests also check operation identity, actual arguments,
serialization ownership and staged-input boundaries.

Static checkpoint restart controls for programs of parts are rejected before
IR construction (ruling 10, #721):
`a program of segments takes restart state through its activation, not a static restart control`.
This is the sole approved diagnostic change: eight exact old/new output pairs
are checked separately, with no exception matching the original corpus or
matrix. Continuation through activation between parts is unchanged.

Every implementation tree compares printed module/lowered outputs against one
archived base, including the combined CPU/GPU, mixed/double order matrix.
Final validation additionally requires five bitwise checkpoint comparisons and
two clean D309 shards. No numerical method, control key, precision mode, file
format or overwrite behavior changes.

## Validation status

The implementation and construction tests are complete. The PR records each
comparison tree and its counts. Rebase validation, five bitwise checkpoints
and both final D309 shards gate readiness.
