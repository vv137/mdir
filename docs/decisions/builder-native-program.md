# D[builder-native-program]

Recorded at its merge.

**The builder constructs a program from owned native IR; only user input is parsed at its boundary.**

This is stage C of #568, after the complete term adapter registry and D363's
typed term fragments. The construction contract and current status belong in
[driver Section 2.12](../driver.md#212-the-files-of-the-builder).

## Ownership and construction

One build owns its MLIR context, module, parsed model modules, and construction
captures. Input IR and destination IR share that context. A term borrows typed
operands and an `OpBuilder` insertion point through a bounded capability;
neither a mutable `Program` nor the complete `Builder` crosses the adapter
boundary. Borrowed state does not escape a hook.

Custom expressions retain their owned `ExpressionIR` snapshots. Staged
`ModelTerm` input is parsed once, with its potential signature and return
checked at that boundary. Declarations and the potential body are cloned with
`IRMapping`, preserving declaration and operation order. Symbols supplied by
the complete program, such as `@atoms`, are checked at the ordinary combined
verification boundary. Duplicate user declarations are not silently renamed. Generic operation
syntax is checked for the container and potential region shape, nullable
function-type properties, and matching block arguments before typed
accessors or cloning are used.
There is no substring search or SSA-token rewriting of model input.

Program declarations, shared potential wrappers, constraints, dynamics,
entry functions, schedules, segments, and output calls are constructed with
`OpBuilder`. Their interfaces take and return typed values and operations.
Generated operation or type syntax is not formatted and reparsed, and an
SSA-name dictionary is not a substitute for those typed interfaces. Once the
last caller moves, connection serialization and carry-marker fixups disappear.

The completed native module is printed for the existing serialized
`Program.module` handoff. Ordinary consumer verification stays at its existing
boundary. Contribution and field order, shared potential and loop placement,
inlining, fusion, and MLIR's default SSA numbering remain unchanged; builder
name hints and new opaque call boundaries are not introduced.

## Validation and status

Control keys, defaults, support restrictions, numerical methods, precision
modes, formats, and overwrite behavior are unchanged. Every implementation
commit compares the full printed module/lowered corpus and combined order
matrix against one archived PR base. The final head compares five checkpoint
states bitwise and passes both clean D309 shards.

Staged-model input parsing and cloning are implemented in `StagedModelIR.cpp`.
The build context outlives the adapters and their captured input modules;
recipe construction errors propagate to the producer. Input buffers remain
owned for parsed resource lifetimes. Native tests cover the frozen exported
model, user string literals and nondefault SSA names, signature rejection,
foreign program operands, and context ownership. The remaining program
construction follows. Particle sets, tuple sets, and the constraint disjoint
union are already native operations in the owned destination module. Their
operation identity and order survive the join with remaining assembly.
Runtime declarations are native `func.func` operations at their original
positions around the model declarations. Their memref element types follow
the state/force precision, and C interface and stop-effect attributes retain
their original conditions. Construction captures a scalar configuration
snapshot rather than borrowing the builder or program.
Virtual-site placement and its work potential share the typed geometry in
`SiteIR.cpp`; site and constraint position changes use native particle maps.
`SiteSpreadIR.cpp` constructs both force-transfer paths and their virials
from typed site inputs, including owner translation and the placement
derivative. The old site-specific text kernels and body assembly are removed.
The existing minimum-image coordinates and arithmetic order are preserved.
Constraint length checks and failure reductions construct their scalar and
Boolean kernels directly. Native conditional stop calls preserve the
existing stream ordering and declared effects. Bond-length inputs are
indexed typed arguments rather than rewritten SSA tokens. The initial
descent map retains its zero-mass guard and existing projection order.
Constraint drift virials and initial pressure traces are constructed in
`ConstraintIR.cpp`. Typed records name their positions, displacement, masses,
time step, projected descent, forces, and velocities without borrowing
builder state. Coordinate/gather order and scalar/vector reductions retain
their original arithmetic and accumulation order. RATTLE velocity projection
uses the same small Gaussian elimination and reads its stored changes for
the half-step virial. Existing unused tuple parameters remain in the kernel
interface, preserving the complete generated program. The constraint-force
virial uses the same bond acceleration matrix, centripetal term, and small
solve for both water and hydrogen groups. SETTLE velocity projection keeps
its original three-bond determinant expressions and reads stored changes
for its virial, preserving the energy/plain-step equivalence.
SETTLE position projection retains its triangle rotation, while M-SHAKE
carries small displacements through native `scf.for` block arguments. Its
single-bond analytic branch keeps the residual and direction checks and
Newton fallback. Bond lengths use typed parameter slots rather than SSA
labels. Constraint kernels no longer write operation text. Molecular
pressure, group kinetic reductions, and rigid-center scaling use typed
records in `CouplingIR.cpp`, retaining the particle/group order, massless
guard, center-of-mass correction, and half-step force term. The otherwise
unused coordinate of the internal group reduction is preserved. Only fields
used by the selected group path cross its capability, and the shared legacy
arithmetic text helper is removed.
Philox rounds, word order, Box-Muller transforms, damping/spread constants,
and the massless-particle guard are native in `NoiseIR.cpp`. Brownian drift
retains its guarded mass/friction division, and the host step counter uses
native load/add/store operations. Seed and noise constants are owned
captures; no generated random-operation text or SSA-name construction
remains in these kernels.
Kinetic observables share typed reductions in `KineticIR.cpp`. Scalar energy
retains its within-particle vector reduction, while axis energies retain their
separate vector accumulation. Momentum broadcasts mass before multiplying;
the force-square reduction retains the zero-mass safe division. Center-of-mass
correction borrows the existing half-vector constant rather than creating one.
Measured half-step excess subtracts the full-step axis energy; unmeasured,
unconstrained excess retains the force-square expression of [Jung2018].
Half-step measurements construct their predictive kick/drift and constrained
velocity correction directly. Particle and rigid-water energy kernels retain
the quarter-mass weighting, scalar summation order, and unused tuple coordinate.
Their predictive state remains separate from the state the step returns.
D363's term recipes still join remaining assembly at this commit.
