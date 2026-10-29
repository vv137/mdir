# D[builder-term-ir]

Recorded at its merge.

**Term fragments are constructed with `OpBuilder` at the same shared potential insertion sites, with typed operands and owned per-build construction state.**

This is step 4 of #568, after the complete adapter registry of #683.
A fragment borrows an `OpBuilder`, its insertion point, and typed operands.
It returns IR values rather than inventing SSA names or exposing the
complete builder or mutable program. The build owns every deferred recipe
and its captures; borrowed hook-local references never escape their calls.
The narrow hosted-model role borrows a synchronous `TermIRWriter` instead
of an output stream. It keeps read-only `TermInputs`; the writer submits
owned recipes and never escapes the hook call.

During the transition, a bounded join with the remaining assembly may use
internal typed placeholders. They are materialized into their original
regions before a valid program is printed or lowered. They introduce no
runtime callback or public, opaque, or otherwise non-inlined call boundary.
The final operations occupy the original shared potential and loop bodies,
so `md-inline`, loop fusion, and deterministic contribution order retain
the same input. MLIR's default SSA numbering is used, without builder-name
`NameLoc` or `printNameLocAsPrefix`.

Materialization validates each connection exactly once and rejects result
count/type mismatches. A final walk rejects every remaining `mdir.term_ir`
attribute, including attributes on newly created operations. The equivalence
harness rejects the same marker in every successfully parsed module and successful lowered emission
before comparison. Native negative tests cover these boundaries. Malformed
builder assembly now fails at this construction boundary with the internal
error `the builder produced invalid assembly`; it cannot pass through to a
reading printer or downstream consumer. The #706/#707 fixes leave no builder
parse failure in the audited corpus and combined-kind matrix.

This decision amends D352's construction rule. A construction parse failure,
even without typed recipes, now reports the internal error above rather than
silently emitting raw generated text. The fallback hid #706 and #714; the
construction parser now shares the full accepted program dialect set with
lowering, including tensor/linalg stage functions. Valid generated programs
remain unchanged.

The shared Lennard-Jones scalar emitter moves first, followed by the other
term emitters in their change-frequency order. One commit moves one
emitter. Custom expression and `ModelTerm` input remain text at their
boundaries; step 5 removes the remaining generated-text assembly boundary.

Custom-expression scalar operations are built from an immutable owned
`ExpressionIR` snapshot. It keeps the AST and grid geometry, with coefficient
data remaining in the program tables. Caller bindings retain precedence over
named definitions, and definitions are computed lazily once. The existing
user-term diagnostic location remains independent of SSA naming. Deferred
recipes receive typed values and never retain borrowed binding maps.

No control key, default, precision support, format, numerical method,
tolerance, or overwrite behavior changes. The printed module and lowered
corpus and combined-kind order matrix run per implementation commit against
one archived PR base. Five bitwise checkpoints and both D309 shards run
once at the final PR head. The PR records the counts and any required
rebase onto the merged adapter stage before ready.

For failed printed-form comparisons, generated-buffer diagnostic coordinates
are ignored. A source excerpt is omitted only when the same diagnostic also
supplies its complete generic operation dump. Diagnostic reasons, severity,
operation dumps, parser excerpts (including missing SSA names), user-file
locations, exit status, and output remain compared. Successful program output
and raw-form failures remain byte-exact. Recorded hash files use version 2
for this diagnostic presentation policy.

## Emitter construction invariants

The prolate near/far splitting emitter also constructs its clamped table
lookup and Horner polynomial with typed operations at the original site. Direct
Coulomb kernels now use typed operations too: cutoff and reaction field,
Gaussian Ewald, prolate series/cells, alchemical charge scaling, and the
clamped distance retain their original operation order. The direct LJPME
dispersion subtraction and its optional cutoff shift are typed fragments
at that same pair-kernel site. Soft-core Lennard-Jones distance,
root-free or switched potential, and alchemical scaling also use typed
fragments, preserving the original intermediate operations and branch order. Shielded
Coulomb kernels for excluded and within-selection pairs use typed Gaussian
or prolate operations, sharing the same typed prolate polynomial builder. The
excluded and within-selection Coulomb terms, including reaction-field
exclusions, also build their complete tuple loops, typed block arguments,
charge scaling, cutoff selection, and yields with `OpBuilder`. The LJPME
excluded-pair loop is typed as well, including its cancellation-free
small-distance Horner series and closed-form selection. Reciprocal dispersion
and Coulomb emission share a typed constructor with owned grid/configuration
data; charge scaling is a typed particle map before the same reciprocal op. The complete
primary nonbonded relation loop is also typed, including alchemical flags,
tunable-table seeds, scalar kernels, and the ordered final contribution.
`TermNonbonded.cpp` no longer writes operation text. The shared custom-expression
scalar emitter also builds typed operations from an owned `ExpressionIR`
snapshot of the parsed AST and grid geometry. Lazy definitions and caller
binding precedence are preserved without retaining borrowed hook state.

The complete topology custom-pair relation loop also uses typed operations:
overlapping group masks, particle-field arguments, all six LJ parameter
lookups, tunable constants, time/lambda/observed bindings, truncation, energy
conversion, and the yield retain their original order. Each loop owns its
configuration and expression snapshot; no hook-local state escapes.

Bare custom-pair loops also construct their regions and yields with typed
operations. Table overrides, uniform and nonuniform arithmetic/geometric/product
mixing, constants, function tables, cutoff modifiers, and ordered total-energy
addition retain the original operation order. Both topology and bare pair
kernels use owned expression snapshots and bounded configuration.

Regular and compound custom-tuple loops also use typed coordinates, regions,
arguments, and yields. Distance conversion, angle/dihedral coordinates,
per-tuple parameters, gradient-field selectors and seeds, function tables,
time/lambda/observed bindings, and energy conversion keep their operation order.
Owned loop configuration replaces borrowed term and gradient references.

Shared centroid geometry also builds typed IR for the tuple and pull adapters.
The minimum-image weighted component sums, distance, angle, and dihedral
retain their explicit arithmetic and loop order. Deferred construction owns
only group count and arity, with typed relation/position/cell/weight operands;
it does not retain center membership arrays or hook-local references.

Centroid scalar terms also construct their parameter constants, expression,
and energy conversion with typed operations. Owned scalar metadata preserves
constant order and geometry/time/lambda/observed binding precedence.
`TermTuple.cpp` no longer writes operation text.

Custom triplet terms also construct their neighborhood relation, coordinate and
particle-place arguments, scalar body, yield, and ordered energy additions with
typed operations. Owned configuration preserves coordinate order, unit
conversions, uniform and gathered parameters, time and table bindings, and the
unused-distance fallback. Triplet execution remains CPU-only; `TermTriplet.cpp`
no longer writes operation text.

External terms also construct cell-frame scaling, particle gathers, scalar
coordinates and bindings, expression results, membership selection, and yields
with typed operations. Owned configuration preserves field and constant order,
charge and parameter arguments, and time/lambda/observed binding precedence.
`TermExternal.cpp` no longer writes operation text.

Auxiliary pull potentials now construct their complete function and scalar
body with typed operations. They borrow the common argument types from the
shared energy potential at materialization, then append coordinate arguments
in their existing order. Owned expression metadata and parameter/lambda values
preserve constant emission and binding precedence; no hook-local state escapes.
Pull output also constructs its time conversion, coordinate/term buffers,
derivative requests, ordered force arithmetic, column stores, casts, and runtime
call with typed operations. Captures own only counts, indices, and names used by
their recipes. `TermPull.cpp` no longer writes operation text.

Born and surface terms construct their neighborhoods, descreening integral,
HCT/OBC radii, screened pair/self energies, and nonpolar surface energy with
typed operations. Their owned configuration contains scalar method data only;
Born radii remain private to the adapter. The Born energy is added to the outer
sum before surface construction, preserving accumulation order as well as
loop order. `TermBorn.cpp` no longer writes operation text.

Positional restraints construct force maps, energy/virial sums, reference
scaling, reductions, diagonal tensors, and their stored energy column with
typed operations. The per-build energy name and missing-state diagnostic from
#706 remain; force-only requests preserve the recorded energy. Loop and scalar
arithmetic order remain unchanged, including the separate barostat virial sum.
`TermRestraint.cpp` no longer writes operation text.

Opaque hosted models construct their private force/energy ABI declarations,
cell projections, typed neighborhood/edge lists, host calls, and ordered
energy/virial accumulations with typed operations. The host role keeps read-only
`TermInputs` and borrows a synchronous `TermIRWriter`; callers submit owned
recipes without exposing mutable parameter access. Staged user `ModelTerm`
text remains at its existing boundary for step 5.

Built-in harmonic bonds construct the complete typed tuple reduction,
including its distance coordinate, tuple-field order, scalar kernel, and
yield. The half constant, displacement, squared displacement, half-force
constant, and final multiplication retain their original order.
`TermBond.cpp` no longer writes operation text.

Built-in harmonic angles construct the complete typed tuple reduction with its
angle coordinate and original tuple-field/scalar operation order.
`TermAngle.cpp` no longer writes operation text.

Urey-Bradley terms construct the complete typed distance reduction over their
existing endpoint tuple. Tuple fields and half/displacement/square/multiply
order remain unchanged.
`TermUreyBradley.cpp` no longer writes operation text.

Periodic dihedrals construct the complete typed tuple reduction and its
period-times-angle, phase subtraction, cosine, offset, and force multiplication
in their original order. Multiple Fourier terms retain their tuple order.
`TermDihedral.cpp` no longer writes operation text.

Harmonic impropers construct their complete typed dihedral reduction. The
three wrapping constants, floor-based reduction into the original interval,
and harmonic scalar arithmetic retain their order.
`TermImproper.cpp` no longer writes operation text.

CMAP constructs its complete typed two-dihedral reduction, wrapped cell
indices, 16 table lookups, and bicubic Horner evaluation. The build owns the
resolution; lookup and row/column accumulation order remain unchanged.
`TermCMap.cpp` no longer writes operation text.

Special 1-4 pairs construct their complete typed distance reduction, optional
charge-gradient gather/products, selected Lennard-Jones and Coulomb terms,
and final sum. The shared typed Lennard-Jones kernel keeps the original
switch choices and scalar order; its final transitional scalar writer is removed.
`TermPairs14.cpp` no longer writes operation text.
