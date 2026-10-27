# D[term-registry]

Recorded at its merge.

**An explicit registry of term builders owns the dispatch for each migrated kind of energy term, beginning with external terms.**

Step 3 of #568 first split the builder into files by responsibility
(#679). This part adds `TermBuilder`, private to `lib/Driver`, and the
explicit list `registerTermBuilders` in `TermBuilders.cpp`. Each migrated
kind has one translation unit with its adapter and emission code. The
first is `TermExternal.cpp` (#680).

The hooks collect parameter fields, emit the selected contribution to a
topology potential, emit standalone term potentials, append the names of
energy-output columns, and emit entry declarations. Two output hooks
resolve observed term names and observed constant values. An unused hook
does nothing. The potential selection carries the existing term mask,
tuple/pair/external indices, and alchemical-only flag; it does not
reinterpret them.

The registry contains direct references to stateless adapters created on
first use. There are no objects that register themselves during global
initialization. The explicit references keep the adapters reachable when
the static libraries are linked with section garbage collection (#655).
List order fixes the order of migrated contributions at each hook site.
The first migration preserves each external-term site relative to the
remaining, unmigrated code: fields after restraints, potential terms after
Born, and energy-output columns after the existing topology, tuple, pair,
and Born columns and before restraints. A later migration whose sites
need different orders must make those orders explicit rather than sort
by name or depend on initialization order.

External entry arguments, buffers, and reordering already follow the
shared field list; their adapter needs no entry declaration. The
observable grouping still follows first-column order. The existing
external expression emitter moves without changing its generated text.
The energy, its derivatives, the integration scheme, control-file keys,
defaults, formats, checkpoints, and overwrite behavior are unchanged.

The alternative, adding cases to the central builder for every new kind,
keeps the overlapping edits that #568 is removing. Automatically
discovered registration also makes linking and emission order depend on
global initialization. Neither is used.

Validation compares the printed modules and lowered programs with an
archived build of main over the builder-equivalence corpus, with zero
tolerance for a difference. Deterministic checkpoints on the five systems
of the builder gate and the full suite are required. Existing external
tests independently check analytic energies, forces, and virials, cell
scaling, time-dependent traps, and error paths on CPU and GPU. Other term
kinds remain for later PRs; this first migration does not complete #568.
