# D[template-ir-pme-header]

Recorded at its merge.

**The PME templates take the types of an instance from a header of alias definitions above their unchanged text, and the names of the instance in IR through the symbol table; no template of the lowering is instantiated by rewriting its text any more.** This is step 4 of #739, by ruling 19 (P1), after D368, D369 and D370. `instantiatePMETemplates` still rewrote three things in the text of `PME.mlir`, `PMEGPU.mlir`, `PMEGPUTriclinic.mlir` and `PMEGPUProlate.mlir`:
- the lines `!pme_pos = f64`, `!pme_chg = f64`, `!pme_frc = f64` and `!pme_real = f64`, which it matched by their exact spelling;
- `PME_TILE_POINTS_E` inside the shapes of the tiles;
- `PME_FFT` inside the names of the transforms of the host, and every name that began with `@mdrt.` or `@mdrt_`, to which it added the suffix of the instance, comments included.

Now:
- **The aliases come from a header.** The templates define no alias. `kernels::getPMETemplateHeader` writes `!pme_pos`, `!pme_chg` and `!pme_frc` with the types that the positions, charges and forces are stored in, and `!pme_real` with the type of the forces. On a device it also writes `!pme_tile_E = memref<Nx!pme_real, #gpu.address_space<workgroup>>` for the edges 4, 8 and 16, with $N = (E+P-1)^3$. The lowerings parse the header followed by the text of the template, unchanged. The parser resolves an alias only as a whole token and checks the body with its real types, so nothing in the text can be matched by accident; a template that used an edge without its alias would fail to parse.
- **The names come from the symbol table.** `kernels::instantiatePMESymbols` renames the symbols that the parsed template defines (the ops with a body) to their names with the suffix of `getPMESuffix`. The audit of #739 checked that these are exactly the `@mdrt.` and `@mdrt_` names. The runtime's `@mdrtFFTPrepare` and `@mdrtFFTColumnBlocks`, the device primitives and the parameters of D369 are declarations and keep their names. The host template declares its transforms as `@mdir.template.fft.NAME`, which become `@mdrtFFT64NAME` or `@mdrtFFT32NAME` for the type of the forces.
- **`instantiatePMETemplates` is removed.** `test/Runtime/template-text.test` shows that the templates hold no `PME_` placeholder and no alias definition, and that no function of the compiler rewrites the text of a template.

The ops and the names are those of before, so the lowered programs do not change. The alternative of ruling 19, P2 (role types of the MDRT dialect that implement the interface of floating point, parsed once and retyped), would have needed a change of the LLVM dialect's ops, which accept only builtin floats. See [pme.md, Section 3](../pme.md#3-spreading-that-does-not-depend-on-the-order-d70).
