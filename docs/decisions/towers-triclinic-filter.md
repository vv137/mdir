# D[towers-triclinic-filter]

Recorded 2026-11-28.

**The lateral filter of the TOWERS tile visit on a triclinic cell keeps a target when the largest of four projected gaps, onto the two face normals and the two diagonal axes of the column's cross-section, is within the padded filter radius. The supports of the four axes come from the vertices of the cross-section. The geometry kernel computes the axes and the supports once per geometry. A traversal converts them, with the field offsets and the face normals, once into workgroup memory.** Issue #1035.

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

Measured on RTX 4090 with the same frame and the same TOWERS shape on the Trp-cage cell (60°, 60°, 90°) and on an orthorhombic control of the same particles and volume, $L\times L\times L/\sqrt2$. To be completed with the measurements of this branch.

## 5. Correctness

To be completed.

## 6. Timing

To be completed.

## 7. AUTO

To be completed.
