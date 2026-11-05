# D[template-ir-pme-scalars]

Recorded at its merge.

**The scalar and expansion parameters of the PME templates on a device are calls of declared functions, `@mdir.template.*`, that the compiler resolves in IR after the parse, not placeholders substituted in text.** This is step 2 of #739, by ruling 19. Before it, `instantiatePMETemplates` replaced several placeholders by matching them as substrings anywhere in the text, comments included:
- `PME_ORDER`, `PME_LANES`, `PME_REAL_BYTES` (also inside the attribute `alignment`), `PME_TILE_SIDE_E`, and `PME_TILE_POINTS_E` as a constant;
- `PME_ATOMIC_ADD %pointer, %value`, which relied on those operand names;
- `PME_SHARED_ADD_E %tile[%t], %value`, which expanded to seven lines with fixed SSA names.

`scripts/generate-pme-gpu-template.py` and `scripts/generate-pme-prolate-gpu-template.py` now write the following, and declare each callee once at the end of the template (`declare_template_calls`):
- for a constant, `func.call @mdir.template.NAME.TYPE() : () -> TYPE`, where NAME is `order`, `lanes`, `real_bytes`, `tile_side_E`, or `tile_points_E`;
- for an addition to global memory, `func.call @mdir.template.global_add(%pointer, %value)`;
- for an addition to a tile, `func.call @mdir.template.shared_add_E(%tile, %t, %value)`;
- for the alignment of a load or a store, the attribute `mdir.template.real_alignment`.

`kernels::resolveTemplateParameters` checks every call and then replaces each where it stands:
- a constant becomes an `arith.constant` of the order, of the lanes (4 up to the order 4, else 8), of the bytes of the type of the forces, or of the side, E + P − 1, or the points, (E + P − 1)^3, of a tile;
- an addition to global memory becomes `llvm.atomicrmw fadd` at the scope of the device in f64, and PTX's `red.relaxed.gpu.global.add.f32` in f32;
- an addition to a tile becomes the address arithmetic and a relaxed `llvm.atomicrmw fadd` at the scope of the block;
- a marked load or store takes the bytes as its alignment.

It then removes the declarations. The device lowering calls it after the parse and before `expandPrimitiveCalls`, the precedent that it follows (D296).

The ops are those that the text made, in the same places, so the lowered programs do not change. The text still provides the aliases of the types, the conversions, `PME_FFT`, the renaming of `@mdrt`, and `PME_TILE_POINTS_E` inside the shapes of the tiles; steps 3 and 4 of #739 take those. See [pme.md, Section 3](../pme.md#3-spreading-that-does-not-depend-on-the-order-d70).
