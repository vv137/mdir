### Performance

- **The TOWERS lateral filter on triclinic cells bounds a target's distance to the column by four axes**, the two face normals and the two diagonals of the column's cross-section, instead of the larger face gap: it removes 8% of the targets that entered the screen on the Trp-cage cell and, with D[towers-split-interleave], 2.3 µs of the 67 µs force launch there. The field offsets and the face normals of a tile are converted from binary64 once per traversal. Orthorhombic cells are unchanged (D[towers-triclinic-filter], #1035).
