# How MDIR is developed

This document describes how work on MDIR is planned, carried out,
reviewed, and merged. The same process applies to human contributors and to
software agents. [CONTRIBUTING.md](../CONTRIBUTING.md) says what a change
must carry (tests, a decision, documents, references), and
[principles.md](principles.md) lists the questions a review asks. This
document covers the steps around them.

## Roles

- **The maintainer** owns the repository. The maintainer chooses the work,
  rules on every user-visible design, and approves every merge and every
  push to `main`.
- **The coordinator** turns the maintainer's goals into pieces of work and
  assigns them. It relays designs and rulings, reviews pull requests, runs
  the full suite, and merges what the maintainer approves. The coordinator
  may be the maintainer, a person the maintainer names, or an agent.
- **Contributors** carry out one piece of work each, on a branch of their
  own, and open a pull request for it.

## A piece of work

A piece of work is one decision, or one fix, and becomes one pull request.
Unrelated refactoring stays out of it. A defect found outside the piece's
scope is reported to the coordinator and recorded as its own piece of work;
it is not fixed in passing.

Pieces that change the same code run one after another. Pieces in disjoint
code may run at the same time, in waves. Overlap of code, not priority
alone, decides the order. For example, two features that both rewrite
`Differentiate.cpp` are sequential, while a new reciprocal sum and a new
thermostat are not.

## The life of a change

1. **Branch.** Start from a GitHub issue; open one if the work has none.
   Work in a separate git worktree, on a branch named `feature/<task>`,
   with a build tree of its own. Never commit to `main`.
2. **Design first, as a draft.** As soon as the user-visible design is
   decided, push the branch and open a draft pull request
   (`gh pr create --draft`). User-visible design means the keys of the
   control file, file formats, defaults, and overwrite behavior. The
   description states the design, the decision label, the plan of
   validation, and the issue it closes (`Closes #N`). The maintainer can then redirect it before the work is
   done. When a user-visible choice is unclear, ask in a PR comment instead
   of guessing, add the label `needs-decision`, and continue with the parts
   that do not depend on the answer; remove the label once it is answered.
3. **Implement and validate.** Push commits to the same branch. Read the
   PR's comments between parts of the work, and answer each one with a
   commit or a reply. After addressing a review labeled
   `changes-requested`, remove that label and say so in a comment; the
   coordinator then reviews the new head.
4. **Ready.** When the implementation, its documents, and its validation
   are complete, rebase on the current `main`, resolve every conflict
   there, and run the full suite at that head, unless an earlier passing
   suite carries over (see [Reusing a suite](#reusing-a-suite)); then mark
   the PR ready (`gh pr ready`). The comment that marks it ready names the
   head commit, the commit of `main` it is based on, and either the suite's
   counts or what the reuse requires. `main` may move after that without
   the PR being rebased again (see [Staying current](#staying-current)).
5. **Review.** The coordinator builds the PR's head commit and runs the
   full suite on a GPU, or verifies the author's suite at that same head
   (the build recorded that commit with no uncommitted changes and was not
   rebuilt after the run, and the log shows no failure), or checks that an
   earlier suite carries over (see [Reusing a suite](#reusing-a-suite)). The
   review comment names the job that tested the head. There is no GPU
   continuous integration, so this check is part of the review. The
   coordinator checks the rules below, posts the
   result as a PR comment naming the commit tested, and labels the PR
   `merge-recommended` or `changes-requested`. Before labeling, the
   coordinator checks the other open PRs: files both change (conflicts to
   expect), interactions of behavior (for example a new key that a
   checkpoint fingerprint must cover), and the order in which they should
   merge. The review comment states what it found.
6. **Decision.** The maintainer decides on GitHub: the label `approved`
   approves the merge; a comment asks for changes.
7. **Merge.** The coordinator merges the PRs labeled `approved`; see
   below.

## Labels and where decisions are made

Reviews, questions, and approvals live on GitHub, so that each PR keeps its
own record. All contributors may push under one account, which cannot
request a review of or approve its own PRs; labels carry the state instead.

| Label | Set by | Meaning |
|---|---|---|
| `needs-decision` | contributor or coordinator | A question for the maintainer, asked in a PR comment; removed when answered |
| `changes-requested` | coordinator | The review asks for changes; removed when they are pushed |
| `merge-recommended` | coordinator | Reviewed, the suite passed at the commit named in the review comment |
| `approved` | maintainer | The maintainer approves the merge |

To approve, the maintainer only adds `approved` and leaves
`merge-recommended` in place: the coordinator merges a PR that carries
both, at the commit named in the review. A PR whose head moves after
`merge-recommended` is reviewed again. Work
that is not yet a PR is tracked as GitHub issues.

## The pull request

The description states:

- the decision label (see below), and the roadmap item;
- what changed;
- the user-visible keys;
- the validation, with numbers: the quantity, the independent reference
  and its value, the difference, and the tolerance, on the CPU and on the
  GPU, in the `mixed` and `double` modes;
- the performance against `main` on the same system, if the change can
  affect it;
- what is left open.

Every PR adds its entry to the Unreleased section of
[CHANGELOG.md](../CHANGELOG.md), under Added, Changed, Fixed, or Removed,
with its decision label. A change that breaks a control file, a checkpoint,
or an output of an earlier release says so there, with what a user has to
do. The entry is part of what the review checks before it labels the PR
`merge-recommended`. Whether a change counts as breaking, when that is in
doubt, is a design question for the maintainer: ask it in a PR comment and
label the PR `needs-decision`.

PR titles, descriptions, comments, and commit messages are in American
English. Commits carry no trailers (no `Co-Authored-By`, no
`Signed-off-by`). A contributor force-pushes only its own branch, and only
with `--force-with-lease`. A contributor never merges its own PR.

Issues, PRs, and their comments name no paths of a contributor's machine:
no home directories, build trees, logs, scratch directories, or private
notes. Others cannot open them, and they leak the layout of a machine.
Describe the measurement or the input instead, and put anything others need
in the repository; paths in the repository are fine.

## Decision numbers

A branch does not take a decision number. It writes its decision as a
label, `D[<label>]` (lowercase letters, digits, and dashes), wherever the
number would go: in the row it adds at the end of the table in
[decisions.md](decisions.md), in comments, in the documents, and in the
paper. Several decisions of one branch take several labels. At merge time
the coordinator runs `scripts/decision-number.sh assign <label>`, which
gives the next free number and rewrites Markdown links to headings whose
anchors change with the number, including duplicate-heading suffixes.
`scripts/decision-number.sh check` also reports stale decision-label anchors
that no heading defines. `scripts/decision-number.sh check --pending`
runs on every push to `main`: it refuses a number given twice and a label
without its row in decisions.md, and lists the labels still waiting for
their numbers, since a merge reaches `main` before the commit that numbers
it. A tag runs the strict `scripts/decision-number.sh check`, which also
refuses any label left without a number. Numbers therefore follow the
order of the merges, and no number is lost to abandoned work.

## Validation

Every new method is compared with an independent oracle. The oracle may be
another program (OpenMM, GROMACS, sander or pmemd, CHARMM) or a reference
written for the purpose, such as a NumPy sum or an analytic result for the
discrete scheme. A value that a test pins after such a comparison names the
program and the value in the test's comment. A statistical check states its
tolerance in standard errors. A test that fails intermittently is reported
with its number of repeats and failures, not rerun until it passes.

A derivative is checked against finite differences. An analysis that may
conclude that a result is zero, empty, or unchanged has to prove it; a case
it cannot decide is an error that names the reason, so that a proven zero
and a failure are never confused.

## Performance

Timings are taken on a GPU that runs nothing else, after checking that no
other job uses it. During development one long run is enough; a figure for
publication is the mean and standard deviation of repeated runs. A PR that
can affect speed reports ms/step or ns/day against `main` on the same
system, and states any slowdown on the Amber suite.

## Shared machines

GPUs are shared among contributors and other users. Each device runs one
job at a time: every GPU command takes a lock for its device and checks the
device's use inside the lock. A timing run has its device to itself; tests
may use either device between timings, and no contributor uses more than
two at once. Full suites queue
behind the lock instead of running side by side. Within one suite, lit runs
at most sixteen tests that need the device at once and the others in
parallel (`test/mdir_lit.py`, `-Dgpu_workers=N`), so the suite needs no
`-j 1`. Most of these tests are bound by compilation on the host, not by the
device; a smaller device or a host with fewer cores takes a smaller N.

On a Slurm machine that permits two MDIR GPU jobs at once, one full-suite
validation may be run as two lit shards (D309). Configure and build
two separate trees from the same source commit, then submit each through the
machine's `suite-run --batch` helper with `--num-shards=2` and respectively
`--run-shard=1` or `--run-shard=2`. Give both jobs `--skip-test-time-recording`:
the recorded times of only one shard would be incomplete for a later full
suite. Keep both trees and their source untouched until both jobs end. Record
both job exit statuses and add their passed, unsupported, and failed counts;
the two shards together, with no skipped selection, are one full-suite gate.
This uses two GPU and CPU allocations at once, so the one-job suite remains
appropriate when the scheduler or other users need the capacity. The
one-suite lock on a workstation still applies.

## Permissions

A contributor that is refused an action, such as a push or a run on a GPU,
reports it. No other contributor, and not the coordinator, performs the
refused action for it or reaches the same outcome another way. The
maintainer decides whether to allow it.

## Staying current

A PR is reviewed and merged at the head that its suite tested, on the
`main` it was based on when it was marked ready. The merge train (see
[Merging](#merging)) tests the combination with the `main` of its time, so
an author does not rebase each time `main` moves:

- The author rebases only when the train cannot merge the PR without
  resolving a conflict in prose or code. Those conflicts are always
  resolved by the author, who knows what both sides mean; the coordinator
  asks with the label `changes-requested`. A conflict limited to the rows
  that every PR adds at the end of `docs/decisions.md` and `CHANGELOG.md`
  is not a reason to rebase: the train keeps both rows, in the order of the
  merges.
- The author also rebases when the train removes the PR because the
  combination failed (see [Merging](#merging)), and when the PR depends on
  something that `main` gained.
- A rebased head is reviewed again, as any head that moves after
  `merge-recommended`, and runs its suite unless the earlier one carries
  over (see [Reusing a suite](#reusing-a-suite)).

## Reusing a suite

A passing full suite at an earlier commit carries over to a new head when
the two trees differ only in documentation, or in documentation and
scripts that nothing else runs:

- **The diff.** The comparison is `git diff --name-only <tested commit>
  <new head>`, tree against tree. After a rebase it includes everything
  `main` gained in between. A head whose own new commits change only
  documentation does not qualify when `main` gained anything else.
- **Documentation.** Documentation means files under `docs/` (the white
  paper included), `README.md`, and `CHANGELOG.md`. Code comments and
  TableGen descriptions do not count, because they are compiled; nor do
  tests and their inputs and comments, which FileCheck and the tests read;
  nor `examples/`, some of which tests run; nor generated templates and
  their generator scripts. `CONTRIBUTING.md` counts, since nothing reads
  it. Any other path in the diff calls for a full suite, except the scripts
  of the next item.
- **Scripts that nothing else runs.** A file under `scripts/` qualifies
  when no build file, code, example, or other script reads or runs it, and
  the only tests that run it are its own tests on the CPU (under
  `test/Lit/` or `test/Docs/`). The release scripts are an example.
  - **The check.** `git grep -l <file name> -- ':!docs' ':!*.md'` lists
    the files that name it. Each match must be the script itself or one of
    its own CPU tests. A match only in a comment does not count; the record
    says so.
  - **What runs again.** Its own tests may change in the same diff, with
    their inputs under `Inputs/`. They, and all of `test/Lit` and
    `test/Docs`, run again at the new head.
  - **What does not qualify:**
    - a script that writes files the repository checks in, such as the
      generators of templates (`test/Runtime/generated-templates.test`
      compares them);
    - `scripts/environment.sh`, which the other scripts source.
- **Documentation that tests read.** Some documents are inputs of tests on
  the CPU, among them `docs/environment.md` (the checks of the environment
  and script paths), `docs/decisions.md` (`scripts/decision-number.sh
  check`), and every Markdown file (`test/Docs`, where it exists). The CPU
  tests that read a changed document run again at the new head:
  `test/Docs`, `test/Lit`, and every test that names a changed file.
- **The record.** The comment names the tested commit, its suite job and
  counts, the new head, the files of the diff, and the CPU tests run again
  with their results; for a script, also the matches of the check above.
  The reviewer checks them before relying on the
  earlier suite.

## Merging

The coordinator merges the PRs labeled `approved` by the maintainer, at the
commit named in each review, in a **merge train**: once an hour, when at
least one approved PR waits. A train that is still running when the next
hour comes is not joined by a second one; the PRs approved meanwhile wait
for the next.

1. Branch `train-<date>-<hour>` from `origin/main`.
2. Merge the approved PRs one after another with `--no-ff`, in the order of
   their approval. A conflict limited to the rows at the end of
   `docs/decisions.md` and `CHANGELOG.md` is resolved by keeping both rows,
   in the order of the merges. Any other conflict leaves that PR out of the
   train: the coordinator labels it `changes-requested` and asks its author
   to rebase.
3. Assign the decision numbers and run `scripts/decision-number.sh check`.
   The numbering commit changes labels into numbers, in the text and in the
   anchors of links to headings that held them, and nothing else; the
   coordinator checks that its diff is limited to that.
4. Build the combined head and run the full suite on a GPU (two D309 shards
   where the scheduler allows), and the compute-sanitizer tier when a PR of
   the train changes device code, after the suite has rebuilt the tree. A
   train never pushes on a failing run. When a run fails, the coordinator
   finds the PR that causes it (from the failing tests, or by running again
   without the suspect), removes it from the train with `changes-requested`
   and the failure, and runs the remaining train again. A train of one PR
   whose head contains `main` needs no run of its own: its merge has the tree
   of the reviewed head, whose suite is the suite of the merge.
5. Fast-forward `main` and push it. The PRs then show as merged; the label
   `approved` is the maintainer's approval of this push. The comment on
   each PR names the train, its head, and the runs that gated it.
6. Rebuild the white paper (`scripts/paper/build-pdf.sh`).
7. Remove the merged worktrees, branches, and build trees.

## Releases

A release is two steps, scripted so that nothing is done by hand
(`scripts/release/`):

1. `scripts/release/prepare.sh VERSION` opens the release PR: it sets
   `project(mdir VERSION ...)` in `CMakeLists.txt` and moves the
   changelog's [Unreleased] under [VERSION] and the date. The release notes
   (`docs/release-notes/vVERSION.md`) and the Known limitations are edited
   on that branch. The PR goes through the labels like any other.
2. After its merge, `scripts/release/publish.sh VERSION` builds the merged
   `main` in a fresh tree, runs the full suite on a GPU, builds the white
   paper, writes the assets (the source archive, the installed tree for
   Linux x86-64 with CUDA unless `--no-binary`, the PDF, the notes, and
   SHA256SUMS), and makes
   the annotated tag. Nothing is tagged unless the suite passes. With
   `--publish` it pushes the tag and creates the GitHub Release, whose
   notes are the changelog's section; the maintainer runs it, or approves
   it. `--container` also builds the Docker image.

`publish.sh` names no path of the machine it runs on. The machine provides,
in the environment or in the file that `MDIR_ENV_FILE` names
([environment.md](environment.md#scripts)): `MDIR_RELEASE_DIR`, where the
release writes its worktree, build tree, assets, and logs; `MDIR_LLVM`,
`HDF5_PREFIX`, and `CUDA_ROOT`, the prefixes of the build; `MDIR_PYTHON`
and `PYBIND11_DIR`, the interpreter and pybind11's CMake directory of the
Python interface; and `MDIR_RELEASE_GPU`, the device of the suite. `cmake`, `ninja`, `lit`,
`pandoc`, `tectonic`, and `docker` are found on the path or named by
`CMAKE`, `NINJA`, `LIT`, `PANDOC`, `TECTONIC`, and `DOCKER`. On a shared
machine, `MDIR_RELEASE_SUITE_LOCK` and `MDIR_RELEASE_GPU_LOCK` name its
locks, which the script takes in that order around the suite. On a
machine whose GPUs belong to a batch scheduler, `MDIR_RELEASE_SUITE` names
instead the machine's command of the suite (#593): the script builds the
tree and calls it as `<command> <build tree> <lit option>...` with the
options of a release (below); the command runs `check-mdir` of the tree on
a GPU of the scheduler, waits for it, prints the log, and returns lit's
status. The suite then needs neither `MDIR_RELEASE_GPU` nor the locks; the
run of the manylinux tree still uses them. The script
stops at its start and names everything that is missing.

`--no-binary`, for a machine without Docker, makes a release without the
manylinux tree: it is neither built nor run, and the assets that the
GitHub Release attaches are the source archive, the PDF, the release notes
(`vVERSION.md`), and SHA256SUMS, which lists the other three. The notes of the
GitHub Release (the file `notes.md` of the assets) then start by saying
that the release has no binary tarball and link
[building.md](building.md) at the tag, which describes the build from the
source.
The changelog's Known limitations of that release should say so too. With
`MDIR_RELEASE_SUITE` and `--no-binary` no GPU is used outside the command.

`publish.sh VERSION --dry-run` rehearses a release, also of a version that
is tagged already: it does everything up to the tag in trees of its own
(their names end in `-dry-run`) and tags, pushes, and publishes nothing;
`--publish` is refused with it; `--no-binary` works with it as with a
release. A run, dry or not, removes
its worktree when it ends and leaves its build tree
(`build-vVERSION[-dry-run]`), its assets (`vVERSION[-dry-run]`), and its
logs under `MDIR_RELEASE_DIR`, to be removed by hand.

The build of a release has the Python interface (`MDIR_ENABLE_PYTHON`),
so that its suite holds the Python tests as the suite of a pull request
does; before #297 it had not, and a release was tagged on a suite in which
they were unsupported. A machine that has interpreters with PyTorch and
with h5py names them in `MDIR_TORCH_PYTHON` and `MDIR_H5MD_PYTHON`, and the
suite gives them to lit (`-Dtorch_python`, `-Dh5md_python`); a release
should set both and `MDIR_BENCH_DIR`. The log of the suite names every
test that is unsupported, and the script prints their count: read it
before `--publish`.

The suite of a release runs more than the suite of a pull request.
`publish.sh` gives lit `-Drelease=1`, which enables the tests that require
`release-tests` (`*-release.test` and `*-release-gpu.test`): where a test
runs only a part of its cases by default, to keep the suite of every pull
request short, its release twin runs the whole matrix. Today these are
the tests of the leapfrog step that leaves the state of the plain step
(`test/Driver/leapfrog-energy-step*.test`) and the Python `observe` tests
(`test/Driver/python-observe*.test`). For leapfrog, by default all 17 cases run on
the CPU in double precision, and the CPU in mixed precision and a GPU in
double and in mixed precision run three of them, one for each form of the
step with constraints; a release adds the 17 cases on the CPU in mixed
precision and on a GPU in double and in mixed precision. Without the
parameter the release tests are reported as unsupported. To run them by
hand, give lit the parameter: `LIT_OPTS="-Drelease=1"` with `check-mdir`,
or `lit -Drelease=1 test`.

The default Python `observe` tests run Double with cutoff and Mixed with
shift for the dipeptide, and Double with shift and Mixed with cutoff for
the mixture, on the CPU and GPU.
They keep both precisions for NPT, external terms, and tunables. The
release twins run all four precision and modifier combinations for the
dipeptide and mixture, including their independent CLI file comparisons
and closed-form dispersion checks.

A test is a candidate for a release twin when it is among the slowest of
a suite because it repeats its cases over targets, precisions, or
parameters, and when its default form still runs every path of the code
that the whole matrix checks, with a case that failed before the fix it
guards for each. Its twin then holds what the default form leaves out, so
that a case leaves the default suite without leaving the release. The
tier is not a place for a case that fails or fails now and then: that is
a defect, and an issue.

GitHub has no GPU, so the build and the suite run on the maintainer's
machine; the workflow `release-check` checks on the pushed tag that the
version, the changelog, the notes, and the decision numbers agree.
