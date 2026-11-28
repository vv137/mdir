# D[towers-input-geometry]

Recorded 2026-11-28.

**The per-atom input kernel of the TOWERS moving dispatch recomputes the cell geometry only when the lattice words of the cell change. It keeps the words of the last geometry in the box buffer, which starts as NaN. The divisions of the fractional conversion stay.** Issue #1050, after D428 (TOWERS), D450 (the device mirror of the cell) and D452 (the coupling step on the device).

## 1. What changes

The input kernel (`TowersMovingRow::emit`, `lib/Conversion/MDExecKernels/TowersBuild.cpp`, launched by `Dispatch::inputs` in `lib/Conversion/MDExecToGPU/TowersDispatch.cpp`) clears the force words, converts each position to fractional coordinates, and bins it. Its thread 0 also stored the lattice in the box buffer and computed the geometry of the bins (`TowersBuild::geometry`) at every evaluation. On a triclinic cell that is, for each axis, a cross product, a square root, nine binary64 divisions (by the norm, by the bins, and for the bin width and its reciprocal) and the stores of the lattice row, the bin step, the unit face normal, the slab height and the bin width: 27 divisions and 3 square roots in one thread, before the compiler folds the constant ones.

Thread 0 now loads the nine (triclinic) or three (rectangular) words of the box buffer and compares them bit for bit with the lattice words of the current cell. If all are equal and none is NaN, it only clears the rebuild status, as the geometry did. Otherwise it stores the lattice and computes the geometry as before. No key is added; the trajectory is unchanged.

1. [derived] The geometry is a function of the lattice words and of the grid (the bins per axis). The arithmetic is the same at every evaluation, so equal words give equal geometry words. A comparison of bit patterns, not of values, also separates $+0$ from $-0$.
2. [derived] The box buffer starts as NaN, written once by a kernel after its allocation. A NaN word fails the comparison, so the first evaluation computes the geometry, and a cell with a NaN word recomputes it at every evaluation, as before.
3. [derived] Sites of one entry share scratch buffers by name (`Dispatch::buffer`). The box and geometry buffers are now named by the grid (`box 6x6x14`, `geometry 6x6x14`), so two sites share them only when their geometries agree.
4. [derived] Only thread 0 of the input kernel writes the box and geometry buffers. The force kernels and `towers_setup` read them. With more cells than atoms, `towers_setup` still computes the geometry from the box at every evaluation.
5. [derived] Under a barostat the cell changes at each scaling, and the evaluation after it recomputes the geometry. In graph replay (D450) the kernel reads the cell from the mirror and compares it in the same way. The setup trial (`tools/mdir/TowersSpecialization.cpp`) allocates its own buffers and keeps the computation at every evaluation.

## 2. The probe: which thread costs the time

Two variants of the main program, traced with Nsight Systems on RTX 4090 over 4,000 steps of NVT under TOWERS (the input kernel's duration over the 3,564 launches of graph replay; the controls of the gap table, `towers_shape` from setup):

- `recip`: the three reciprocals of the diagonal of the lattice computed once by the host per launch and multiplied in the fractional conversion of every thread, in place of the divisions. This changes the rounding of the fractional coordinates, so it was a probe only.
- `skip`: thread 0 skips the geometry when the cell is unchanged (this decision).

| Input | Cell | Atoms | main (µs, median) | `recip` | `skip` | `recip` + `skip` |
|---|---|---:|---:|---:|---:|---:|
| Trp-cage ff19SB+OPC | triclinic | 8,602 | 9.82 | 9.09 | 5.09 | 4.67 |
| Trp-cage ff14SB+TIP3P | triclinic | 6,545 | 10.46 | 9.09 | 4.93 | 4.51 |
| TIP3P 10k | rectangular | 9,989 | 3.36 | — | 3.36 | — |

1. [measured] Thread 0's geometry is the fixed cost of the issue: skipping it removes 4.7 µs (ff19SB) and 5.5 µs (ff14SB) of the median. Hypothesis (b).
2. [measured] The divisions of the conversion cost at most 0.4 µs on top (`recip` + `skip` against `skip`). Hypothesis (a) is the smaller part.
3. [measured] On the rectangular cell the geometry is three pairs of divisions and the kernel does not change.

## 3. Why the divisions stay

1. [derived] The quotient of the fractional conversion is stored in the coordinates that the force kernel reads (`raw`) and enters the binary32 local coordinates through the lattice product. A product by a rounded reciprocal differs from the correctly rounded quotient in the last bit for some positions. Such a difference changes the binary32 coordinates only rarely, so a few frames could agree by chance; the force words would not be bitwise those of main by construction.
2. [derived] A correctly rounded quotient from a stored reciprocal takes a product and a residual correction (two fused multiply-adds per correction). It saves less than the 0.4 µs that the plain product saved.
3. This decision keeps the divisions. The 0.4 µs is recorded as measured and not taken.

## 4. What remains

[measured] With the geometry skipped, the input kernel on the triclinic cells is bimodal: about 15% of the launches take 3.1–3.6 µs, the rest 4.9–5.1 µs. The pack kernel that follows (`towers_pack`) is bimodal in the same steps (4.7 against 7.2 µs median, ff19SB) on main as well, and not on the rectangular cell (2.6 µs). Both kernels take a longer path on the same steps, which points to the population of the cells (members beyond the capacity go to the overflow lists of the build), not to the geometry. It is left to a separate issue.

## 5. Validation

See the pull request for the numbers: the trajectories of main and of this change in a configuration where main repeats itself bit for bit, NVT and NPT, on both Trp-cage inputs, a Trp-cage substitute in a truncated octahedron and the rectangular controls; and the test below.

`test/Driver/towers-input-geometry-gpu.test` runs the dipeptide under a barostat that compresses the cell by several percent, in a rectangular and a triclinic cell, with LAUNCH and with GRAPH. A run of 200 steps is compared with 100 steps continued from the checkpoint of the same run at step 100, within ten times the spread of two uninterrupted runs. The continued run starts without a geometry, so a geometry kept from an earlier cell shows as a difference.
