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
   are complete and the full suite passes locally, rebase on `main` and
   mark the PR ready (`gh pr ready`).
5. **Review.** The coordinator builds the PR's head commit and runs the
   full suite on a GPU. There is no GPU continuous integration, so this run
   is part of the review. The coordinator checks the rules below, posts the
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

## Decision numbers

A branch does not take a decision number. It writes its decision as a
label, `D[<label>]` (lowercase letters, digits, and dashes), wherever the
number would go: in the row it adds at the end of the table in
[decisions.md](decisions.md), in comments, in the documents, and in the
paper. Several decisions of one branch take several labels. At merge time
the coordinator runs `scripts/decision-number.sh assign <label>`, which
gives the next free number. `scripts/decision-number.sh check` runs on
every push to `main` and refuses a label left without a number or a number
given twice. Numbers therefore follow the order of the merges, and no
number is lost to abandoned work.

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
behind the lock instead of running side by side.

## Permissions

A contributor that is refused an action, such as a push or a run on a GPU,
reports it. No other contributor, and not the coordinator, performs the
refused action for it or reaches the same outcome another way. The
maintainer decides whether to allow it.

## Merging

The coordinator merges only PRs labeled `approved` by the maintainer:

1. Branch `review-<name>` from `origin/main`.
2. Merge the approved PRs one at a time with `--no-ff`, resolving
   conflicts: rows of `decisions.md` stay in the order of the merges, lists
   of keywords take the union, and rows of the control-file reference keep
   both changes.
3. Assign the decision numbers and run `scripts/decision-number.sh check`.
4. Build and run the full suite, on a GPU. A merge never proceeds on a
   failing suite.
5. Fast-forward `main` and push it. The PRs then show as merged; the
   label `approved` is the maintainer's approval of this push.
6. Rebuild the white paper (`scripts/paper/build-pdf.sh`).
7. Remove the merged worktrees, branches, and build trees, and ask the
   authors of the open PRs to rebase on the new `main`.

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
   Linux x86-64 with CUDA, the PDF, the notes, and SHA256SUMS), and makes
   the annotated tag. Nothing is tagged unless the suite passes. With
   `--publish` it pushes the tag and creates the GitHub Release, whose
   notes are the changelog's section; the maintainer runs it, or approves
   it. `--container` also builds the Docker image.

GitHub has no GPU, so the build and the suite run on the maintainer's
machine; the workflow `release-check` checks on the pushed tag that the
version, the changelog, the notes, and the decision numbers agree.
