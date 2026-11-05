# D[template-ir-select]

Recorded at its merge.

**A sum keeps the functions of a PME template that it needs by attributes that state what each function is for, not by substrings of their names.** This is the last step of ruling 19 on #739, after D368, D369, D370 and D374. After the parse, the lowerings dropped the functions that a sum does not need by testing parts of their names:
- on the host and on a device, `contains("pme_potential")` for the potential at the particles (D231) and `contains("_listed")` for a list of the particles (D262);
- on a device, `starts_with("mdrt_gpu_pme_bin")` or `contains("_spread_tiles")` for the binning and the tiles (D282), with the own edge by `contains(... + E + "_")`, a test that depended on the `_` of the suffix of the instance.

Nothing checked these conventions. For example, the potential kernels of the prolate template, `mdrt_gpu_pme_prolate_potential`, do not contain `pme_potential`, and so are kept in every instance.

Now:
- **The marks.** A template function carries `mdir.template.potential` or `mdir.template.listed` (unit attributes), or `mdir.template.tile_edge = E : i64`.
- **Where they are written.** `generate-pme-gpu-template.py` marks the functions as it writes them (`template_marks`, `mark_template_functions`), and `generate-pme-prolate-gpu-template.py` reuses that. Both mark exactly the functions that the old tests selected, so the selection does not change; the prolate potential kernels stay unmarked. `PME.mlir` is marked by hand: 5 functions of the potential and 5 of a list. The other templates carry 15 marks (`PMEGPU.mlir`), 5 (`PMEGPUTriclinic.mlir`) and 7 (`PMEGPUProlate.mlir`).
- **Where they are read.** `kernels::keepTemplateFunction` keeps a function unless one of these holds:
  - it is marked potential and the sum has no potential;
  - it is marked listed and the sum has no list;
  - it has a tile edge other than the edge that the sum spreads through, or the sum has no tiles.

  It then removes the attributes, so that the lowered programs are those of before. `MDExecToLoops::addPMETemplates`, `MDExecToGPU::addPMETemplates` and `MDExecToGPU::addProlateTemplates` call it in place of the tests of names.

The checks of whether a module already holds an instance (by the exact name of the instance) and the exact name of the centers build of the neighbor matrix are not tests of parts of a name and are unchanged. `test/Runtime/template-text.test` shows that the lowerings test no part of the name of a template function, and counts the marks. See [pme.md, Section 3](../pme.md#3-spreading-that-does-not-depend-on-the-order-d70).
