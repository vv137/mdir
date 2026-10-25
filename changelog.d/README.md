# Changelog fragments

A pull request adds its entries of [CHANGELOG.md](../CHANGELOG.md) here, in
a file of its own, and does not edit `CHANGELOG.md` (#602). The file is
named after the decision's label, `changelog.d/<label>.md`, or after the
issue, `changelog.d/<issue>-<word>.md`, so that two PRs never write the
same file and the merge train has nothing to join. It holds one or more
subsections, each with its entries, as they would stand in [Unreleased]:

```md
### Added

- **Checkpoints are written on a thread beside the steps** (D[<label>], #564).
```

The subsections are those of `scripts/release/changelog.py` (Added,
Changed, Deprecated, Removed, Fixed, Security, Validated, Performance,
Known limitations), and an entry follows the rule of
[docs/workflow.md](../docs/workflow.md#the-pull-request). Its links are
written as from `CHANGELOG.md`, where it goes: `docs/topic.md`, not
`../docs/topic.md`. `changelog.py unreleased` prints [Unreleased] with the
fragments; `changelog.py fragments` checks them. At a release,
`changelog.py release` takes the entries of [Unreleased] and then those of
the fragments, in the order of their names, and removes the fragments.
This file is not a fragment.
