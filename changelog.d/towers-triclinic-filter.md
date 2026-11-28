### Performance

- **The TOWERS lateral filter on triclinic cells bounds a target's distance to the column by four axes**, the two face normals and the two diagonals of the column's cross-section, instead of the larger face gap; the field offsets and the face normals of a tile are converted from binary64 once per traversal (D[towers-triclinic-filter], #1035). Orthorhombic cells are unchanged.
