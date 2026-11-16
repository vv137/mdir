### Fixed

- **Lists of units: refusals instead of a silent rewiring or an assertion** (D419, #898).
  - `convert-md-exec-to-gpu` now refuses a loop that yields a list of units other than its argument through refreshes of it. Before, the loop was rewired to its initial list.
  - It also refuses a list that passes through the results or block arguments of `scf.if` or `scf.while`, and the message names the op.
  - The coverage query `verifySegmentCoverage` returns a diagnostic for a list whose refreshes begin at a function argument or at a region result. Before, it asserted, and a build without assertions read a null op.
  - No IR that `md-exec-reuse-neighbors` produces reaches these paths.

### Validated

- **Exclusions of the rest with two taken kinds** (D419, #898).
  - The $K = 2$ mixture of the coverage and forces tests now holds a molecule of the rest with an anchor and three excluded pairs.
  - The exclusion stop of the build has a case for a pair between two copies of a kind.
  - The coverage checker derives the slots of the members from the kind's frame.
