# D[towers-split-interleave]

Recorded 2026-11-28.

**The parts of a split TOWERS tower take the column's sources in turn: part $p$ of `split` takes the logical sources $p, p + \texttt{split}, \dots$. They took contiguous halves, and only the first half meets the home targets.** Issue #1035, Section 4 of D[towers-triclinic-filter].

## 1. What changes

`Force::traversal` (`lib/Conversion/MDExecKernels/TowersForce.cpp`) gave part $p$ of a tower the logical sources $[\lfloor n_sp/k\rfloor, \lfloor n_s(p+1)/k\rfloor)$ of the column, with $n_s$ the column's sources and $k$ the split. Now part $p$ takes the logical sources $p + k\ell$ for the part-local indices $\ell = 0, 1, \dots$, fewer than $\lceil (n_s - p)/k\rceil$. `Force::stage` maps a part-local source index to its logical index before it resolves the field. The home-target range of `Force::visitTile` is converted to part-local indices. With `split = 1` the code is unchanged. No key, format or launch shape changes; `towers_shape` keeps `split` 1 or 2.

## 2. Why

1. [derived] The logical sources of a column run through its fields in order: home, the lower layers (fields 1 to the reach), then the upper layers. A home target takes the home sources above its own slot and the sources of the lower layers (`emitOwnedTowersSpans`); a lateral target takes the whole column. So every source before the end of the lower layers meets the home targets and every later source does not.
2. [measured] With contiguous halves the first part holds the home and lower layers. On the Trp-cage cell (ff19SB+OPC, 8,602 particles, 3 x 3 x 5 bins at HALF height, split 2, 90 tower blocks) the first parts took 9.2 times the screen passes of the second; on the orthorhombic control of the same particles and volume 4.4 times; on ff14SB+TIP3P 9.5 and 4.5 times. The tilt of the column along $\mathbf c$ moves the upper layers away from the half plate on the negative side, which is why the cell is worse than its control.
3. [measured] A block's time follows its screen passes (correlation 0.99 to 1.00 over the 90 tower blocks, against 0.02 to 0.14 for the screened source and target slots). The tower and exclusion blocks fit in one wave of the 128 SMs, so the launch lasts as long as its slowest block, a first part.
4. [derived] With interleaved parts each part holds every $k$-th source of every field. The sources of a field are sorted by their quarter-cell key, so each part receives the same mix of layers and of positions within them, and its screen passes are close to $1/k$ of the tower's.
5. [derived] Coverage. The map $\ell \mapsto p + k\ell$ over the parts $p$ is a bijection onto $[0, n_s)$, so each source of the column lies in exactly one part. A home target at logical index $o$ takes the logical sources $L$ with $o < L < n_\text{lower}$, where $n_\text{lower}$ counts the home and lower layers. In part $p$ these are the part-local $\ell$ with $\lceil (o + 1 - p)/k\rceil \le \ell < \lceil (n_\text{lower} - p)/k\rceil$ (each bound 0 when its logical bound does not exceed $p$), a contiguous range, so the screen's range masks keep their form.

## 3. Measured

Instrumented launches (phase counters on, same frame, same pinned shape, RTX 4090), 261 force launches each:

| Input | Part 0 / part 1 screen passes, before | After | Slowest block, before (µs) | After | Launch span, before (µs) | After |
|---|---:|---:|---:|---:|---:|---:|
| Trp-cage ff19SB+OPC | 9.18 | 1.00 | 97.4 | 68.9 | 97.3 | 69.6 |
| Orthorhombic control, ff19SB+OPC | 4.39 | 1.00 | 89.5 | 66.0 | 88.1 | 65.5 |
| Trp-cage ff14SB+TIP3P | 9.49 | 1.00 | 62.8 | 44.0 | 62.5 | 44.0 |
| Orthorhombic control, ff14SB+TIP3P | 4.49 | 1.00 | 56.7 | 41.8 | 58.4 | 43.0 |

The mean block grew by 2 µs (58.6 to 60.7 µs on the cell): the staging of a strided set of sources reads more cache lines. Uninstrumented timings are in Section 6 of D[towers-triclinic-filter].

## 4. Tests

The TOWERS runtime tests run split 2 with one and two warps (`towers-triclinic-gpu`, `towers-outputs-gpu`, `towers-exclusions-gpu`, `towers-triclinic-filter-gpu`) and compare the words with exhaustive references. A mutation that rounds the part-local bounds down instead of up fails `towers-triclinic-gpu` and `towers-triclinic-filter-gpu`.
