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
error case, alongside hosted controls without restraints. Those controls
expose the separate default-resource effect declaration error (#707), also
retained explicitly. Staged-model controls successfully reach printed and
lowered IR. `--host-raw` additionally compares hosted module text before
these baseline verifier failures; this supplementary check is for stage A,
where text is preserved, rather than the later typed-IR stage.
