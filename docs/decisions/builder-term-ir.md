# D[builder-term-ir]

Recorded at its merge.

**Term fragments are constructed with `OpBuilder` at the same shared potential insertion sites, with typed operands and owned per-build construction state.**

This is step 4 of #568, after the complete adapter registry of #683.
A fragment borrows an `OpBuilder`, its insertion point, and typed operands.
It returns IR values rather than inventing SSA names or exposing the
complete builder or mutable program. The build owns every deferred recipe
and its captures; borrowed hook-local references never escape their calls.

During the transition, a bounded join with the remaining assembly may use
internal typed placeholders. They are materialized into their original
regions before a valid program is printed or lowered. They introduce no
runtime callback or public, opaque, or otherwise non-inlined call boundary.
The final operations occupy the original shared potential and loop bodies,
so `md-inline`, loop fusion, and deterministic contribution order retain
the same input. MLIR's default SSA numbering is used, without builder-name
`NameLoc` or `printNameLocAsPrefix`.

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
