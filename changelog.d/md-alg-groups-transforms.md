### Added

- A device contract test of `md_alg.bin_members` (D[md-alg-groups-transforms], #869), whose SIMT lowering had not run on a device: the SIMT lowering, a thread a bin, against the serial one, to the bit.

### Changed

- The first experiment of collection-level transformations on the groups build (#869) is recorded with its measurements and not merged. Key fusion, in which the scatter of a counting sort recomputes its keys, and the removal of the key buffer gave the same output, but the step was 0.3 to 0.6% slower on an RTX 5090. Sort elision is blocked on both of the groups build's sorts, which take at most 1.5% of a step. `md-alg-transform-collections` stays empty.
