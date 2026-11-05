### Changed

- **The scalar and expansion parameters of the PME templates on a device are resolved in IR**: the order, the lanes, the bytes of the real type, the constants of the tiles, and the atomic additions to the grid and to a tile are calls of declared `@mdir.template.*` functions, replaced after the parse, not substituted in text (D369, #739). The lowered programs do not change (`docs/pme.md`, Section 3).
