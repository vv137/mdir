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
Output construction uses typed physical state in `OutputIR.cpp`. Owned lambda
tables preserve derivative/state order and the single-state shortcut. Observed
constants preserve runtime tunable operands and user output slots. Gradient
requests retain the one-iteration host-controlled loop, scalar/field request
indices, energy shift, field host calls, and scalar stores. Solvent output
extracts its two energies before the host call. Native captures borrow neither
the builder nor the mutable program; metadata updates remain at construction.
Stochastic cell strain in `BarostatIR.cpp` retains isotropic, semi-isotropic,
and anisotropic host-call order, pressure units, constant virial correction,
and stored cell-column scaling. Cell and reference geometry use native values;
barostat state stores retain extraction/index/store order and the nine-value
host checkpoint copy. Virial diagonals/traces and three-axis reductions retain
their original extraction and arithmetic order. Trotter work and new-volume
state are typed results. Unused coupling parameters do not cross the typed input.
The diagonal interface names and exports both its vector and original scalar
components. Virial reporting consumes those scalar outputs directly, without
extracting them again. `checkDiagonalExports` in `test/Driver/TermIRTest.cpp`
rejects a consumer of unexported components at construction and verifies that
the valid host call uses the original three extracts exactly once.
Minimization's scaled 16-norm, guarded mean square, and normalized motion use
typed kernels in `DynamicsIR.cpp`. A constrained trial returns its chosen
positions and acceptance predicate explicitly; the energy consumer borrows
that predicate and preserves the positive-infinity rejected value. Scalar
roots, mass-metric direction, projection order, and vector broadcasts remain
at their original points in the graph.
Particle reordering uses typed state in `ReorderIR.cpp`, preserving the
constraint-anchor renumber/gather, spatial ordering, field/ID permutations,
and final topology renumbering. Derived listed-particle fields are regenerated
from the preceding semantic source field slot rather than permuted. Native
inputs omit unused listed-field values and absent force state; results name
all state and parameter fields required by the schedule. The unused anchor
coordinate and zero contribution remain in the original gather kernel.
Shared potential bodies construct neighborhoods, ordered scalar accumulation,
lambda constants, and returns with native operations. Exclusion selection
retains registry order, and the scalar configuration snapshot owns the cutoff
and lambda values. Empty topology energies retain their zero constant. The
remaining potential signatures and surrounding program structure still join
assembly until their typed scopes are constructed directly.
The initial term-energy buffer is an explicit native output shared with the
stored-correction recipes. Definitions are evaluated and stored in their
existing order before the stored correction energies; the final dynamic
memref cast and host write remain last. The physical output state and owned
potential-symbol list cross the native capability, without builder captures.
Center-of-mass coupling exports momentum, center velocity, and removed energy
explicitly. Thermostat construction preserves optimal-temperature control
without center motion, host factor calls, squared factors, and heat from the
actual kinetic energy. Only parameters used by the selected thermostat cross
its typed input. The final velocity map retains center subtraction, thermostat
scaling, and non-Trotter barostat scaling in order, after the bath host call.
Barostat work preserves scaled kinetic energy, stored group state, first-order
or mean virial work, and kinetic scale changes as typed scalar/vector results.
Trotter work retains its subtraction order; exact work retains the constant
energy change from inverse volumes and the energy difference of the newly
evaluated positions. Projection/evaluation phases remain in their original
places between these computations, with no algebraic reassociation.
Potential evaluation shares `PotentialStateIR` between dynamics and output.
Requests explicitly return energy, forces, and virial in their original order;
plain steps, minimization, and exact cell work use that typed boundary. Step
time retains signed integer conversion and multiplication order, with the
same memory read where needed. Private connection packing remains transitional
and captures no builder/program in native recipes; it is removed with the
remaining native function and schedule scopes.
D363's term recipes still join remaining assembly at this commit.

Step kicks, drifts, scale inverses, velocity corrections, minimization constants,
and dynamic returns are constructed with native operations. Their original
operand order and control-flow positions are preserved. Program signatures
and schedule scopes remain transitional.

Schedule energy evaluations, half-kicks, optimal-temperature arithmetic,
Trotter state reads, energy rows, spread/frame/checkpoint calls, and
minimization trial acceptance use native operations in their original order.
The surrounding loop and function scopes remain transitional.

Frame evaluation presence is checked on actual operations after native
recipes materialize, rather than by searching the generated assembly for
an evaluation token. The existing frame-source guard remains transitional.

Entry indices, start/time/noise state, loop counts, cell memory and edge
indices, reference edges, buffer imports, total mass and the backward half
step use native operations. Logical field, table and relation types are
constructed from their semantic metadata, preserving storage precision.

Barostat and thermostat parameters are owned snapshots with native constants
and explicit typed results. Frame boundary buffers and request masks, force
and virial output, initial energy rows and leapfrog initialization preserve
their original order. Part counts use semantic slots; the frame-if-due loop
and nonfinite-state reduction at part boundaries are constructed natively.

Every dynamics and descent call is constructed as a native `dyn.step`,
using its actual program definition for result types. Physical state,
optional scales, parameter fields and noise inputs retain their argument
order. Restart Trotter vectors and velocity reconstruction are native.

`ScheduleIR.cpp` constructs step numbers from typed loop state and preserves
nested-period arithmetic. Minimization convergence memory, polling bounds,
row/frame counters, loop counts and boundary calls are native. Cell reloads
export every scalar edge explicitly at their original positions.

Segment step/start counters, part-count memory, first-call selection and
plain-segment energy/time/spread outputs use native operations. The initial
correction virial zero is native as well. Remaining generated schedule text
is function/loop structure and its temporary carry/connection machinery.

Complete native minimization signatures and bodies cover SETTLE, M-SHAKE,
virtual-site placement and force spreading. Semantic parameter slots and actual
block arguments preserve the owned operation and argument identity after transient
metadata is cleared. Restraint and hosted-model evaluations use synchronous
narrow roles with actual physical values and semantic particle-field bindings;
shared kernels preserve correction order and effects. The old minimization
signature/body assembly is removed.

All dynamics functions construct complete native signatures and bodies for
Verlet, leapfrog, Langevin and Brownian motion. Semantic constraint groups retain
SETTLE/M-SHAKE position corrections, RATTLE projection, integrated or separate
failure tests, ordered stops, molecular/constraint virials and future half-step
kinetic and solvent measurements. Time-dependent potentials, virtual sites,
restraint/hosted evaluation and Trotter scaling preserve their existing order.
Explicit result configuration retains velocity staggering and every measurement.
Half-step eligibility uses one predicate in the runtime owner, native constructor
and construction matrix. Native return counts and types are checked before a
terminator is created; construction errors name the integrator, constraints,
energy, measurement and site combination.
Bare pair and triplet potentials construct native signatures and bodies through
synchronous typed contribution hooks. Semantic particle-field/table bindings
supply actual operands, and a shared scalar-potential factory owns the complete
body before its temporary placement join. Construction tests verify operation,
argument and scalar-capture lifetime after transient metadata is cleared.
Built-in bond, angle, Urey-Bradley, improper, dihedral and CMAP potentials use native signatures and semantic tuple
bindings, with shared kernels and unchanged accumulation order.
External potentials also use native bodies with actual particle fields, tables,
frame/time values and semantic expression bindings. Lambda and observed arguments
retain their declaration and binding order; their native expression kernel is
shared with the remaining topology path.
Other topology-potential and entry/schedule scopes remain in progress; their temporary assembly
join and loop-carry connections remain until those scopes move.
