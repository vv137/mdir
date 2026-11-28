# D[towers-split-interleave]

Recorded 2026-11-28.

**The parts of a split TOWERS tower take the column's sources in turn, by chunks of 32 consecutive sources: part $p$ of `split` takes the logical chunks $p, p + \texttt{split}, \dots$. They took contiguous halves, and only the first half meets the home targets.** Issue #1035, Section 4 of D[towers-triclinic-filter].

## 1. What changes

`Force::traversal` (`lib/Conversion/MDExecKernels/TowersForce.cpp`) gave part $p$ of a tower the logical sources $[\lfloor n_sp/k\rfloor, \lfloor n_s(p+1)/k\rfloor)$ of the column, with $n_s$ the column's sources and $k$ the split. Now part $p$ takes the part-local indices $\ell = 32c + r$ ($0 \le r < 32$) as the logical sources $32(kc + p) + r$: chunks of 32 consecutive sources, in turn. A warp stages one chunk, so the staging stays contiguous. A first version took single sources in turn ($p + k\ell$); it balanced the parts as well but cost 2.8% on TIP3P 10k NPT, whose parts were nearly balanced already (Section 3). `Force::stage` maps a part-local source index to its logical index before it resolves the field. The home-target range of `Force::visitTile` is converted to part-local indices. With `split = 1` the code is unchanged. No key, format or launch shape changes; `towers_shape` keeps `split` 1 or 2.

## 2. Why

1. [derived] The logical sources of a column run through its fields in order: home, the lower layers (fields 1 to the reach), then the upper layers. A home target takes the home sources above its own slot and the sources of the lower layers (`emitOwnedTowersSpans`); a lateral target takes the whole column. So every source before the end of the lower layers meets the home targets and every later source does not.
2. [measured] With contiguous halves the first part holds the home and lower layers. On the Trp-cage cell (ff19SB+OPC, 8,602 particles, 3 x 3 x 5 bins at HALF height, split 2, 90 tower blocks) the first parts took 9.2 times the screen passes of the second; on the orthorhombic control of the same particles and volume 4.4 times; on ff14SB+TIP3P 9.5 and 4.5 times. The tilt of the column along $\mathbf c$ moves the upper layers away from the half plate on the negative side, which is why the cell is worse than its control.
3. [measured] A block's time follows its screen passes (correlation 0.99 to 1.00 over the 90 tower blocks, against 0.02 to 0.14 for the screened source and target slots). The tower and exclusion blocks fit in one wave of the 128 SMs, so the launch lasts as long as its slowest block, a first part.
4. [derived] With interleaved parts each part holds every $k$-th chunk of the column, and a column of $n_s$ sources has about $n_s/32$ chunks (about 23 and 30 on the two Trp-cage cells). The sources of a field are sorted by their quarter-cell key, so each part receives the same mix of layers and of positions within them, and its screen passes are close to $1/k$ of the tower's.
5. [derived] Coverage. The map $(p, c, r) \mapsto 32(kc + p) + r$ is a bijection onto the logical indices, so each source of the column lies in exactly one part. It is increasing in $\ell$ within a part. A home target at logical index $o$ takes the logical sources $L$ with $o < L < n_\text{lower}$, where $n_\text{lower}$ counts the home and lower layers; in part $p$ these are the part-local $\ell$ with $B_p(o + 1) \le \ell < B_p(n_\text{lower})$, a contiguous range, where $B_p(x)$, the number of part-local indices below logical $x$, is 32 times the chunks of $p$ below chunk $\lfloor x/32\rfloor$ plus $x \bmod 32$ when that chunk is $p$'s. The screen's range masks keep their form.

## 3. Measured

Instrumented launches (phase counters on, same frame, same pinned shape, RTX 4090), 261 force launches each. "Single" is the first version, single sources in turn; "chunks" is this decision.

| Input | Part 0 / part 1 passes: contiguous | Single | Chunks | Slowest block (µs): contiguous | Single | Chunks | Launch span (µs): contiguous | Single | Chunks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Trp-cage ff19SB+OPC | 9.18 | 1.00 | 1.08 | 97.4 | 68.9 | 70.0 | 97.3 | 69.6 | 68.6 |
| Orthorhombic control, ff19SB+OPC | 4.39 | 1.00 | 1.09 | 89.5 | 66.0 | 66.2 | 88.1 | 65.5 | 67.6 |
| Trp-cage ff14SB+TIP3P | 9.49 | 1.00 | 1.11 | 62.8 | 44.0 | 43.9 | 62.5 | 44.0 | 44.0 |
| Orthorhombic control, ff14SB+TIP3P | 4.49 | 1.00 | 1.10 | 56.7 | 41.8 | 42.3 | 58.4 | 43.0 | 43.0 |

Single sources made the staging of a block 0.5 µs longer (3.5 to 4.0 µs on the cell); chunks keep it at 3.5.

Uninstrumented, RTX 4090, 2 fs, two jobs in opposite orders, n = 6, ns/day and TOWERS force + finish (µs per step):

| Input | `main` | Filter only (D[towers-triclinic-filter]) | Single sources | Chunks |
|---|---:|---:|---:|---:|
| Trp-cage ff19SB+OPC NVT | 1057 ± 20 | 1071 ± 18 | 1264 ± 28 | 1268 ± 25 |
| Trp-cage ff19SB+OPC NPT | 792 ± 17 | 789 ± 12 | 952 ± 16 | 947 ± 19 |
| TIP3P 10k NPT (split 2) | 1893 ± 41 | 1895 ± 35 | 1846 ± 37 | 1865 ± 36 |
| force + finish, ff19SB NVT | | 89.4 ± 2.2 | 63.9 ± 1.3 | 64.0 ± 1.4 |
| force + finish, TIP3P 10k NPT | | 32.1 ± 0.8 | 34.6 ± 0.5 | 34.0 ± 0.5 |

TIP3P 10k NPT is the one input where the interleave costs: in each job the chunks are 1.6% and 1.3% below `main` (1.5% over both, within `main`'s standard deviation of 2.2%), and its force + finish is 1.9 µs longer than with contiguous halves. Its shape takes split 2 with a panel of 192 and 128 threads; its parts were nearly balanced already. JAC (NVT, NPT) and TIP3P 10k NVT take split 1 and are unchanged. A choice between contiguous and interleaved parts by the setup trial, which already times both splits, would remove that cost; it needs a field of `towers_shape` and its entry under the automatic-choices contract, and is left to a follow-up.

## 4. Tests

The TOWERS runtime tests run split 2 with one and two warps (`towers-triclinic-gpu`, `towers-outputs-gpu`, `towers-exclusions-gpu`, `towers-triclinic-filter-gpu`) and compare the words with exhaustive references. `towers-triclinic-filter-gpu` also runs 1,200 particles in 64 and 128 cells with split 2, so that columns hold more than one chunk. A mutation that drops the part of the chunk below a bound fails `towers-triclinic-gpu` and `towers-triclinic-filter-gpu`, as did, for the first version, one that rounded the part-local bounds down.
