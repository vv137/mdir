# D[towers-triclinic-filter]

Recorded 2026-11-28.

**The lateral filter of the TOWERS tile visit on a triclinic cell keeps a target when the largest of four projected gaps, onto the two face normals and the two diagonal axes of the column's cross-section, is within the padded filter radius. The supports of the four axes come from the vertices of the cross-section. The geometry kernel computes the axes and the supports once per geometry. A traversal converts them, with the field offsets and the face normals, once into workgroup memory.** Issue #1035.

## 0. Two changes and their effects

This decision and D[towers-split-interleave] come from one measurement (Section 4) and land together. Their effects are separate:

1. **The interleaved source split (D[towers-split-interleave]) is the main gain on every cell with split 2.** A block's time follows its screen passes (correlation 0.99 to 1.00), not its screened slots (0.02 to 0.14), because a pass costs a pair evaluation (about 70% of the slowest block) while a screened slot costs a share of one tensor-core instruction. With contiguous halves the first part of a tower held 9.2 times the passes of the second on the Trp-cage cell and 4.4 times on its orthorhombic control; interleaved, 1.00 on both. TOWERS force + finish on the ff19SB+OPC cell: 90.5 → 65.0 µs per step (Section 6).
2. **The four-axis filter (this decision) is the small triclinic gain**, and it stays on for triclinic cells beside the interleaved split (the maintainer, 2026-11-28):
   - Never more work: $L$ is the two-axis bound with two more axes, so up to the outward rounding of the supports the targets it keeps are a subset of those the two-axis filter keeps; the pairs within the cutoff are the same (Section 5).
   - Measured gain: with the interleaved split on both arms, force + finish 67.1 ± 0.4 → 64.9 ± 0.2 µs on the ff19SB+OPC cell and 47.4 ± 0.7 → 46.6 ± 1.0 on ff14SB+TIP3P (Section 6, item 3); no arm was slower beyond its spread.
   - Cost: two more dot products and maxima per filtered target. Orthorhombic cells keep the exact sum.

## 1. What changes

- `Force::visitTile` (`lib/Conversion/MDExecKernels/TowersForce.cpp`), triclinic cells: the filter took the larger of the two squared face gaps. It now takes the square of the four-axis bound of Section 2. Orthorhombic cells keep the sum of the two squared face gaps, which is the exact distance to a rectangular cross-section.
- `TowersBuild::geometry` (`TowersBuild.cpp`): rows 0 and 1 of the triclinic geometry gain five columns, the diagonal axis of the row (three components) and the supports of the row's face normal and of its diagonal. The geometry has 18 columns instead of 13. The packed binary32 geometry of `towers_prepare_force` gains ten words at its end: 53 floats at FULL height and 59 at HALF instead of 43 and 49.
- A triclinic force entry that reads the binary64 geometry, as a run does, converted the offset of a field for each staged particle (nine binary64 loads and conversions) and the face normals for each filtered target. A traversal now converts the field offsets, the two face normals and the ten lateral words once into workgroup memory (172 bytes at HALF height). An entry that traverses no tower, such as a finish without a replay, converts nothing.
- No key of the control file changes. `neighbor_structure = "AUTO"` still takes the matrix on a triclinic cell (Section 7).

## 2. The bound

1. [derived] Let $P$ be the cross-section of the column of the owning bin in the plane normal to the tower axis $\mathbf c$, measured from the bin center. $P$ is the parallelogram with the vertices $\pm\tfrac12\mathbf s_0 \pm \tfrac12\mathbf s_1$ projected onto that plane, where $\mathbf s_a$ is the bin step along cell vector $a$ (a row of $H$ divided by the bin count). It is centrally symmetric.
2. [derived] Let $\mathbf u$ be a unit vector normal to $\mathbf c$, and $s(\mathbf u) = \max_{\mathbf x\in P}\mathbf u\cdot\mathbf x$ its support. For a target at $\mathbf p$ and any $\mathbf x\in P$, $\lvert\mathbf u\cdot\mathbf p\rvert - s(\mathbf u) \le \lvert\mathbf u\cdot(\mathbf p - \mathbf x)\rvert \le \lVert\mathbf p - \mathbf x\rVert$, since $\lvert\mathbf u\cdot\mathbf x\rvert \le s(\mathbf u)$ by central symmetry. So $\max(0, \lvert\mathbf u\cdot\mathbf p\rvert - s(\mathbf u)) \le d(\mathbf p, P)$, the distance of the target to the column. The projection of $\mathbf p$ onto the plane does not change $\mathbf u\cdot\mathbf p$, so the bound holds for the column as a whole.
3. [derived] The four axes are $U = \{\mathbf n_0, \mathbf n_1, \widehat{\mathbf n_0+\mathbf n_1}, \widehat{\mathbf n_0-\mathbf n_1}\}$, with $\mathbf n_a$ the unit normal of the faces of cell vector $a$ (normal to the other two vectors, hence to $\mathbf c$). The bound is $L(\mathbf p) = \max_{\mathbf u\in U}\max(0, \lvert\mathbf u\cdot\mathbf p\rvert - s(\mathbf u))$, so $L(\mathbf p)\le d(\mathbf p, P)$, and a target with $d(\mathbf p, P) \le r_c$ passes a filter that keeps $L(\mathbf p) \le r_c$.
4. [derived] Since $\mathbf n_0\cdot\mathbf s_1 = 0$ and $\mathbf n_0\cdot\mathbf s_0 = h_0$, the slab height of a bin, $s(\mathbf n_0) = h_0/2$: the face terms are the face gaps of the former filter. The diagonal $\widehat{\mathbf n_0+\mathbf n_1}$ lies in the normal cone of the vertices where the faces of $\mathbf n_0$ and $\mathbf n_1$ meet, so it bounds the distance in the corner region that the two face gaps leave open.
5. [derived] Why not the sum of the squared face gaps on a triclinic cell. For a target outside a vertex along $\widehat{\mathbf n_0+\mathbf n_1}$ at the distance $t$, each face gap is $t(1 + \mathbf n_0\cdot\mathbf n_1)/\lVert\mathbf n_0+\mathbf n_1\rVert$, and the sum of their squares is $t^2(1 + \mathbf n_0\cdot\mathbf n_1)$. With $\mathbf n_0\cdot\mathbf n_1 > 0$ it exceeds $t^2$, and a filter by that sum drops targets within the cutoff. The test of Section 5 holds such a pair.
6. [derived] The exact distance $d(\mathbf p, P)$ is 0 inside $P$ and otherwise the least distance to the four edges of $P$. It bounds what any filter of this kind can remove, and Section 4 reports the share of that removal that $L$ reaches. Two positive face gaps do not make the nearest point a vertex: on a parallelogram that is not a rectangle, the foot of the perpendicular can lie inside an edge.

## 3. Rounding

1. [derived] The geometry kernel computes the diagonal axes and the four supports in binary64, each support as the largest projection of the four vertices of item 2.1 in the Cartesian frame of the staged positions. The force entry rounds each axis component to the nearest binary32 and each support up to the next binary32 at or above the binary64 value. The comparison keeps ties: a target with $\mathrm{fl}(L^2)$ equal to the squared filter radius passes.
2. [derived] Let $B$ be the coordinate bound that chooses the screen's scale (`screenConstants`), so each staged coordinate has magnitude at most $B$. Rounding an axis to binary32 moves $\mathbf u\cdot\mathbf p$ by at most $2^{-24}\sum_k\lvert u_k\rvert\lvert p_k\rvert \le 2^{-24}\sqrt3\,B$. The dot product by two fused multiply-adds and one product adds at most $\gamma_3\sqrt3\,B$, with $\gamma_3 = 3u/(1-3u)$ and $u = 2^{-24}$. The subtraction of the support adds at most $2^{-24}(\sqrt3\,B + s) \le 2^{-23}\sqrt3\,B$, since a support is a projection of a point within the bound. The computed bound therefore exceeds $L(\mathbf p)$ by at most about $6\sqrt3 \cdot 2^{-24}B \approx 2^{-20.6}B$.
3. [derived] The filter radius is $(r_c + 2^{-15}B)$, squared and widened by $1 + 2^{-19}$ (`screenConstants`). The widening covers the rounding of the square of the bound and of the radius. The padding $2^{-15}B$, the common-image staging allowance of the triclinic screen (Section 4.8 of the paper), is 48 times the filter's own rounding. It also covers the staging error of the two endpoints, a few units of $2^{-24}B$ each.

## 4. Where the 2.2 times comes from

Measured on RTX 4090 with the same frames and the same pinned TOWERS shape (512 threads, one block per SM, split 2, HALF height: 3 x 3 x 5 bins, 90 tower blocks and 28 exclusion blocks per launch for ff19SB+OPC) on the Trp-cage cell (60°, 60°, 90°) and on an orthorhombic control of the same particles and volume, $L \times L \times L/\sqrt2$. The control's residues were wrapped by their centers into that box and relaxed by minimization and 10 ps of NVT (OpenMM); its grid is again 3 x 3 x 5 bins with the same populations.

1. [measured] Kernel time per step, uninstrumented `main`, Nsight Systems over steps 1,200 to 3,800 of 4,000, NVT, 2 fs:

| Input | TOWERS force + finish (µs) | TOWERS pair kernels (µs) | Matrix pair loop (µs) | Matrix pair kernels (µs) | Ratio of the pair kernels |
|---|---:|---:|---:|---:|---:|
| Trp-cage ff19SB+OPC, cell | 92.4 | 101.9 | 40.3 | 46.4 | 2.20 |
| Orthorhombic control | 78.2 | 81.1 | 40.1 | 46.3 | 1.75 |
| Trp-cage ff14SB+TIP3P, cell | 63.4 | 73.3 | 33.5 | 38.8 | 1.89 |
| Orthorhombic control | 52.7 | 55.6 | 31.9 | 37.0 | 1.50 |

   The pair kernels are every kernel launched under a pair loop's name: under TOWERS they add the per-atom input kernel of the rebuild (`TowersMovingRow` in `TowersBuild.cpp`: it clears the words, converts to fractional coordinates and bins), under the matrix the pair kernel of the thermostat steps, amortized over all steps. The exclusion corrections run inside the fused tuple kernel. The matrix does not depend on the cell's shape. So of the 2.20 times, 1.75 are present on the orthorhombic control, and 1.26 come with the cell.
2. [measured] The counters of an instrumented build, per force launch averaged over 261 launches, two-axis filter (`main`'s) on the same frames:

| Quantity | Cell | Control |
|---|---:|---:|
| Input targets of the tiles | 124,034 | 118,102 |
| Kept by the lateral filter, per input target | 0.974 | 0.720 |
| Screened slot pairs (sources times kept targets) | 40.0 M | 29.6 M |
| Screen passes | 4.21 M | 4.21 M |
| Pass rate of the screen | 0.105 | 0.142 |
| Records | 341,500 | 323,300 |
| Sources per tile / panel (480) | 0.69 | 0.73 |
| Targets per tile / stripe (512) | 0.93 | 0.93 |
| Mean tower block (µs) | 59.3 | 57.4 |
| Slowest tower block (µs) | 97.7 | 89.4 |
| Exclusion block, the closest-image replay (µs) | 9.4 | 2.8 |

3. [measured] The tower and exclusion blocks fit in one wave of 128 SMs, so a launch lasts as long as its slowest block. A block's time follows its screen passes (correlation 0.99 to 1.00 over the 90 tower blocks), not its screened slot pairs (0.02 to 0.14). The slowest blocks are the first parts of the towers: with contiguous halves of the column, only the first part meets the home targets, and it holds 9.2 times the passes of the second on the cell, 4.4 times on the control (D[towers-split-interleave]). The tilt of the column moves the upper layers away from the half plate, which is why the cell is worse.
4. [measured] The phases of the slowest block on the cell: staging 4.4 µs, filter 3.1, screen 13.5, pair evaluation 69.2, the rest 8.0. The closest-image replay of the exclusion blocks (9.4 µs) and the conversion of the image shifts (now once per traversal, Section 1) are not on the critical path.
5. [measured] So the 2.2 times is mostly the split: the first parts of the towers carry the home pairs, the launch waits for them, and the cell's tilt makes it worse. The filter is a small term: the cell keeps 0.974 of its lateral targets against 0.720 on the control. That excess is the cell's geometry: its bins are 12.3 Å across their faces against 15.0 Å on the control, for a cutoff of 12 Å. No lateral filter recovers it (item 6).
6. [measured] The filters on the same frames. Kept per input target: two-axis 0.974, four-axis 0.893 (in the kernel, 261 launches); offline on the input frame, two-axis 0.968, four-axis 0.866, exact distance 0.843. The four-axis bound removes 8.3% of the targets that pass the two-axis filter in the kernel (10.5% offline); the exact distance would remove 12.9% offline, so the bound reaches 82% of what any lateral filter can remove (81% on the control, where the sum is exact). The screened slot pairs fall from 40.0 M to 36.7 M per launch and the pass rate rises from 0.105 to 0.115; the passes are unchanged. The slowest block's screen phase is 13.49 µs with either filter: the filter does not move the critical path.

## 5. Correctness

1. [measured] The instrumented build appends every unordered pair that the direct force kernel takes, within the cutoff, excluded pairs included, from the first launch of a module, and the runs write the frame of that launch. Brute force over the closest image in binary64 gives the reference set under the same cutoff and exclusion convention. On the input frames after 300 steps and on four frames of each Trp-cage input taken from a matrix run every 500 steps (then 10 steps): zero pairs missing, zero extra and zero duplicated outside the pairs within $10^{-4}$ Å of the cutoff, where binary32 and binary64 disagree (0 to 9 such pairs per frame of about 4.14 M and 2.34 M pairs). The same holds on the orthorhombic controls, with the two-axis and the four-axis filter, and after D[towers-split-interleave].
2. [measured] The matrix's pair set is not dumped. It is compared here through brute force, which its own tests check, and through the run's energies (Section 6).
3. [measured] `Runtime/towers-triclinic-filter-gpu.test`: two sheared lattices, ((8, 0, 0), (0, 8, 0), (4, 4, 6)) and ((8, 0, 0), (-3, 8, 0), (-2, -3, 8)), with $\hat{\mathbf n}_0\cdot\hat{\mathbf n}_1$ = 0.31 and 0.43, 150 random dyadic particles each, and a corner pair at 0.981 and 0.990 of the cutoff whose summed squared face gaps are 1.12 and 1.18 times the cutoff squared. The force words of a spring within the cutoff match an exhaustive loop over images exactly, for the packed and the binary64 geometry entries, reach 1 and 2, one and two warps per tower, in both insertion orders and through the replay of the certificate. Mutations that fail it: the sum of the squared gaps (the corner pair is missing), supports from one vertex, and part-local bounds rounded down (D[towers-split-interleave]).

## 6. Timing

RTX 4090, 2 fs, mixed precision, two jobs with the arms interleaved in opposite orders, three repeats per arm in each (n = 6), the rate of steps 3,000 to 6,000 of 6,000 as `mdir run` reports it; mean ± standard deviation and the mean without the lowest and highest sample. SM clock under load 2,573 and 2,598 MHz in the two jobs. Arms: `main` with the matrix; `main` with TOWERS; this decision alone (the four-axis filter and the conversion once per traversal, "filter"); this decision with D[towers-split-interleave] ("branch"). TOWERS takes the shape pinned in Section 4 in all three TOWERS arms; "branch, AUTO shape" takes the shape of the setup trial.

1. [measured] Whole step, ns/day:

| Input | Matrix | TOWERS, `main` | Filter | Branch | Branch, AUTO shape |
|---|---:|---:|---:|---:|---:|
| Trp-cage ff19SB+OPC, NVT | 1274 ± 26 | 1042 ± 22 | 1052 ± 17 | 1202 ± 65 (trimmed 1210) | 1246 ± 21 |
| Trp-cage ff19SB+OPC, NPT | 1038 ± 21 | 779 ± 17 | 782 ± 12 | 933 ± 17 | 928 ± 23 |
| Trp-cage ff14SB+TIP3P, NVT | 1546 ± 40 | 1307 ± 86 (trimmed 1326) | 1344 ± 27 | 1544 ± 32 | 1531 ± 69 |
| Trp-cage ff14SB+TIP3P, NPT | 1239 ± 27 | 990 ± 24 | 985 ± 19 | 1123 ± 29 | 1137 ± 25 |
| JAC, NVT (orthorhombic) | | 1639 ± 42 | 1642 ± 42 | 1641 ± 39 | |
| JAC, NPT | | 1405 ± 29 | 1405 ± 33 | 1401 ± 29 | |
| TIP3P 10k, NVT | | 2386 ± 46 | 2380 ± 32 | 2351 ± 73 (trimmed 2364) | |
| TIP3P 10k, NPT | | 1841 ± 65 (trimmed 1856) | 1868 ± 29 | 1814 ± 32 | |

2. [measured] TOWERS force + finish, µs per step (Nsight Systems, NVT, n = 6):

| Input | `main` | Filter | Branch |
|---|---:|---:|---:|
| Trp-cage ff19SB+OPC | 94.9 ± 1.4 | 90.5 ± 1.5 | 65.0 ± 1.3 |
| Orthorhombic control | 81.5 ± 1.6 | 81.6 ± 1.3 | 62.1 ± 1.0 |
| Trp-cage ff14SB+TIP3P | 64.6 ± 2.3 | 64.4 ± 1.9 | 45.8 ± 1.0 |
| Orthorhombic control | 56.3 ± 1.5 | 55.9 ± 1.8 | 42.6 ± 1.0 |

3. [measured] The filter alone, with the interleaved split on both arms (the instrumented build run without counters and with the two-axis filter, against the branch; n = 3): force + finish 67.1 ± 0.4 against 64.9 ± 0.2 µs on the ff19SB cell, 47.4 ± 0.7 against 46.6 ± 1.0 on ff14SB.
4. [measured] The orthorhombic controls JAC NVT and NPT and TIP3P 10k NVT are unchanged within one standard deviation: their pinned shapes, those the setup trial chose on `main`, take split 1, where the interleaved split is the identity and the filter does not apply. TIP3P 10k NPT takes split 2: 1,814 ± 32 against 1,841 ± 65 ns/day on `main`, 1.5% lower, within the spread of `main`; against the filter arm (1,868 ± 29) 2.9% lower. Its tower parts were nearly balanced before; the strided staging of D[towers-split-interleave] costs there.

5. [measured] The branch arm above is the first version of D[towers-split-interleave], single sources in turn. Its final form, chunks of 32 sources, measured against it in two more jobs (n = 6): ff19SB+OPC NVT 1,268 ± 25 against 1,264 ± 28, NPT 947 ± 19 against 952 ± 16, TIP3P 10k NPT 1,865 ± 36 against 1,846 ± 37 (`main` 1,893 ± 41); see Section 3 of that decision.

## 7. AUTO

[measured] With the branch the whole step under TOWERS is within 2% of the matrix in NVT on both Trp-cage inputs (1,246 against 1,274 and 1,531 against 1,546 ns/day, AUTO shape) and 8 to 11% below it in NPT. It does not beat the matrix, so this decision does not propose that AUTO time TOWERS on triclinic cells. The rule that keeps AUTO on the matrix for a triclinic cell belongs to #1007, not to `main`; it stays as it is. What remains on these grids is granularity: 90 tower blocks on 128 SMs at one block per SM, and a launch of about 65 µs against 41 µs for the sum of the block times spread over all SMs.

## 8. The whole step outside the pairs

[measured] ff19SB+OPC cell, NVT, Nsight Systems, two traces per arm, branch with its AUTO shape against the matrix of `main`, one stream each (µs per step): untraced step 139.4 against 138.0; TOWERS force + finish 66.9 against the matrix pair loop 43.3; the other kernels under a pair loop's name 11.2 against 4.8 (under TOWERS the per-atom input kernel of the rebuild, `TowersMovingRow` in `TowersBuild.cpp`, which clears the words, converts to fractional coordinates and bins; under the matrix the pair kernel of the thermostat steps, amortized over all steps); list upkeep 6.9 against 27.9; PME 23.2 against 23.4; bonded 7.5 against 6.8 (the exclusion corrections run inside the fused tuple kernel); integration 21.8 against 23.4; idle and host 11.7 against 15.0; the thermostat-step extra 8.8 against 12.9. The one term that grows outside the TOWERS force launch is the input kernel: 11.2 µs on the cell against 2.6 on the orthorhombic control, about 10 µs fixed per launch from the triclinic conversion and binning. It is #1050.
