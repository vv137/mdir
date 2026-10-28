# Builder equivalence

`check.py` compares the printed module and lowered program of two builders.
The reference for a staged PR is built once from a git archive of its base;
each implementation commit is compared with that same reference.

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
reported separately from generated-program differences. The known hosted
model/restraint stored-energy SSA error (#706) is retained as a baseline
error case. Hosted controls without restraints are successful module/lowered
cases after the default-resource effect declaration fix (#707), alongside
the staged-model controls. `--host-raw` additionally compares hosted module
text before the known baseline failures; this supplementary check is for stage A,
where text is preserved, rather than the later typed-IR stage.

The order matrix also generates the synthetic CHARMM chain from the reader
fixture, combined with custom tuple, centroid, pair, external, and restraint
terms in both precisions. It exercises all bonded families, their independent
energy columns, special 1–4 switching, and the CMAP table in both targets and
both program forms, alongside the existing Born/model cases.

The nonbonded extension combines the same kinds with direct cutoff, reaction
field, P3M, prolate PME, LJPME, and alchemical selection. Both precision modes,
targets, and program forms are compared in printed module/lowered output,
including the order of the loops that lower to fused kernels.
