# Builder equivalence

`check.py` compares the printed module and lowered program of two builders.
The reference for a staged PR is built once from a git archive of its base;
each implementation commit is compared with that same reference.

Before comparison, every successfully parsed module and successful lowered emission is rejected if
it contains a `mdir.term_ir` attribute. Two outputs with equally surviving
internal connections therefore cannot pass the equivalence gate together.
Native connection tests also cover missing, duplicate, malformed, count/type
mismatch, and surviving markers at the materialization boundary.

`order.py BASE_MDIR COMMIT_MDIR --printer MDIR_OPT --scratch OWN_DIRECTORY`
compares combined terms in CPU/GPU, double/mixed, run/segment forms. It
covers tuple/centroid, pair, Born/surface, restraint, external, and triplet
ordering, then adds single/multiple staged models and hosted-model NVE,
NPT, and minimization sites. The hosted cases run with exact, padded, and
plugin-search pair-list modes. At most eight comparison workers run.

The script emits programs; it never runs a model or uses a GPU. Its small
C plugin supplies cutoff metadata and refuses evaluation. This checks the
host-call and pair-list emission paths without an optional model framework,
and makes no numerical claim about a model. CPU models and GPU triplets
retain their expected support errors; unexpected fixture errors are
reported separately from generated-program differences. Hosted controls with
and without restraints are successful module/lowered cases after #706 and
#707, alongside the staged-model controls. No parser or verifier failure is
accepted as a baseline in this matrix. `--host-raw` additionally compares
hosted module text; this supplementary check is for stage A, where text is
preserved, rather than the later typed-IR stage.

The order matrix also generates the synthetic CHARMM chain from the reader
fixture, combined with custom tuple, centroid, pair, external, and restraint
terms in both precisions. It exercises all bonded families, their independent
energy columns, special 1–4 switching, and the CMAP table in both targets and
both program forms, alongside the existing Born/model cases.

The nonbonded extension combines the same kinds with direct cutoff, reaction
field, P3M, prolate PME, LJPME, and alchemical selection. Both precision modes,
targets, and program forms are compared in printed module/lowered output,
including the order of the loops that lower to fused kernels.

The staged-dialect controls add a minimal tensor helper and a small frozen
export from `scripts/mlip/painn-stages.py` (one feature, no message layers,
64 rows per tile). They exercise tensor/linalg functions in both precisions,
targets, and run forms without needing JAX or a device. Ordinary comparisons
remain strict; the #714 repair separately enumerates the segmented GPU
transitions against the old main that skipped the carry rewrite.

For failed printed-form comparisons, generated-buffer diagnostic coordinates
are ignored. A source excerpt is omitted only when the same diagnostic also
supplies its complete generic operation dump. Diagnostic reasons, severity,
operation dumps, parser excerpts (including missing SSA names), user-file
locations, exit status, and output remain compared. Successful program output
and raw-form failures remain byte-exact. Recorded hash files use version 2
for this diagnostic presentation policy.

The nonbonded matrix includes the supported power-two soft-core paths with
a plain cutoff and potential shift, in both precision modes and targets.

Expression cases cover all scalar primitives, lazy definitions, and a
constant with no bound inputs, in both precisions and targets.

Grouped topology-pair cases combine every LJ parameter lookup with overlapping
groups, particle fields, a tabulated function, and time dependence in both
precisions and targets. Their modules and lowered programs must verify.

Bare-pair cases additionally cover arithmetic/geometric/product mixing without
overrides and an ungathered uniform-parameter function term, including the
ordered addition of several pair contributions in both precisions and targets.

Tuple cases combine regular angles/dihedrals and compound distance/angle/dihedral
coordinates with the other kinds, particle fields, time, and alchemical bindings.

Centroid geometry cases combine all three arities with overlapping weighted
groups, regular/compound tuples, the other term kinds, and pull output in both
precisions and targets.

Centroid expression cases combine lambda, time, polynomial parameter dependence,
weighted groups, and pull output in both precisions and targets, retaining the
reader's existing restriction on tabulated functions in centroid terms.

Triplet cases cover all distance/angle/cosine coordinates, particle-place fields,
uniform parameters, time, tabulated functions, and a constant-zero expression
with its required fallback coordinate. Both precisions verify on the CPU; GPU
targets retain the existing support error.

External-term cases combine charges, particle parameter lists, coordinates, time,
lambda, and a function table in cell and fixed frames, in both precisions and
targets. Existing combined controls retain the ungathered and uncharged paths.

Pull-potential fixtures combine radial and Cartesian distance terms with angle
and dihedral terms, lambda and time, retaining their complete common argument
signature and coordinate derivative order in both precisions and targets.

Pull-output fixtures add NPT time dependence and time-independent output to the
radial/Cartesian, angle, and dihedral cases, in both precisions and targets.
