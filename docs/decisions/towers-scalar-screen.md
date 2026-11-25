# D[towers-scalar-screen]

Recorded at its merge.

The optional `[execution] towers_shape.screen` selects `"AUTO"`, `"MMA"`,
or `"SCALAR"`. AUTO uses the matrix instruction at compute capability 7.5
and above. Older devices use the scalar screen. Preflight refuses an
explicit MMA request on an older device. The setup log and manifest record
the effective choice and reason. Both the setup trial and production kernels
use that choice for every launch at the selected height.

Both screens import the same lattice features and apply the same padded
threshold. A scalar lane evaluates the four-term dot product for its target
and packs the signs into the source mask. Section 4 of the paper gives the
no-false-negative bound.

The bounded signed 32-bit certification vote uses a subgroup sum.
Compilation for sm_80 and newer enables the NVVM redux instruction. Older
targets use five XOR shuffles and additions. The wide vote retains its
original overflow bound and accumulation. This also fixes #978 on sm_75.

## Tensor-core ablation

For a full warp of 32 targets and a panel of 128 sources on sm_89, the
isolated screen compiled to 385 MMA or 1,461 SCALAR warp instructions
after the shared-input barrier. This interval includes owner masking
and population counts and excludes five harness output stores. Dividing
by 4,096 candidate pairs gives 0.093994 and 0.356689 instructions per
candidate. The MMA path used 32 HMMA instructions. The scalar path
used 512 FFMA instructions. These are full-panel static counts.

All 4,905,552 certified force words on the eight fixed frames agreed
bit for bit. The occupied atom IDs and ownership spans agreed exactly
on each frame under the common single-split traversal.

The table gives means over the two reversed timing orders on one RTX 4090.
Screen time estimates scale the Nsight force duration by the fixed-frame
CTA screen-time fraction. The CTA column uses the device global timer directly. Whole-step times use the final unprofiled third.
The owner comparison covers span descriptors and atom IDs in occupied slots.

| Input | Force words | Unequal words / spans / atom IDs | CTA screen µs MMA / SCALAR | Attributed grid screen µs MMA / SCALAR | Force µs MMA / SCALAR | Step µs MMA / SCALAR |
|---|---:|---|---|---|---|---|
| TIP3P-10k | 29967 | 0 / 0 / 0 | 2.574 / 7.703 | 3.038 / 8.911 | 34.697 / 40.510 | 104.938 / 110.289 |
| TIP3P-20k | 59640 | 0 / 0 / 0 | 3.467 / 10.650 | 3.949 / 12.022 | 44.458 / 53.808 | 123.438 / 132.459 |
| JAC | 70674 | 0 / 0 / 0 | 3.147 / 9.360 | 5.264 / 15.422 | 51.675 / 60.938 | 139.773 / 148.409 |
| Factor-IX | 272718 | 0 / 0 / 0 | 4.995 / 14.271 | 13.628 / 39.283 | 136.542 / 166.037 | 310.540 / 338.525 |
| Cellulose | 1225827 | 0 / 0 / 0 | 4.855 / 13.229 | 57.350 / 158.163 | 576.806 / 693.030 | 1121.350 / 1231.209 |
| STMV | 3201285 | 0 / 0 / 0 | 3.777 / 10.325 | 164.658 / 435.401 | 1695.196 / 1977.370 | 3459.463 / 3764.706 |
| Trp-cage-CHARMM36m | 19635 | 0 / 0 / 0 | 22.436 / 91.460 | 23.264 / 93.115 | 259.247 / 312.242 | 328.395 / 385.733 |
| Trp-cage-ff19SB | 25806 | 0 / 0 / 0 | 39.483 / 159.471 | 41.045 / 156.782 | 417.907 / 514.817 | 495.562 / 585.885 |
