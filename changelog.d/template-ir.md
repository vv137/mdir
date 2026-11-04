### Changed

- **The f32 instance of the neighbor-matrix templates is made in IR** from the parsed f64 template, not by text substitution: types and constants retyped, truncations to casts, and only defined functions renamed (D368, #739). The lowered programs do not change (`docs/ops.md`, Section 7.6).
