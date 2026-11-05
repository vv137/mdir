# D[template-ir-pme-convert]

Recorded at its merge.

**A conversion of floating point in a PME template is one op, `mdrt.template_convert`, that the compiler makes an extension, a truncation, or a cast of the bits in IR for the types of the instance; the seven text placeholders are gone.** This is step 3 of #739, by ruling 19. `PME.mlir`, written by hand, and the templates of a device (D369's generators) wrote seven placeholders where a conversion between the types of the positions, the charges, the forces (`!pme_real`), and f64 depended on the instance:
- `PME_EXTEND_POS`, `PME_EXTEND_CHG`, `PME_NARROW_FRC`;
- `PME_CHG_TO_REAL`, `PME_F64_TO_REAL`, `PME_REAL_TO_F64`, `PME_REAL_TO_FRC`.

`instantiatePMETemplates` replaced each with an op name by matching it as a substring of the text. All seven followed one rule: a cast of the bits between equal types, an extension to a wider type, and a truncation to a narrower one (`PME_REAL_TO_FRC`, between `!pme_real` and `!pme_frc`, was always the cast).

The templates now write `mdrt.template_convert %x : T to U`, a new op of the MDRT dialect. Its operand and result are floating point, or vectors of it of the same shape. After the parse, `kernels::resolveTemplateConversions` checks every op, then replaces each where it stands with `arith.bitcast`, `arith.extf`, or `arith.truncf`, by that rule. The host and the device lowerings call it after the parse. The ops are those that the text made, in the same places, so the lowered programs do not change.

The text still provides the aliases of the types, `PME_FFT`, the renaming of `@mdrt`, and `PME_TILE_POINTS_E` in the shapes of the tiles; step 4 of #739 takes those. See [pme.md, Section 3](../pme.md#3-spreading-that-does-not-depend-on-the-order-d70) and [mdrt.md](../dialects/mdrt.md).
