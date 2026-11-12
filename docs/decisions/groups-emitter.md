# D[groups-emitter]

Recorded 2026-11-12.

**The device build of the groups of neighbors (D89) is written by a C++ emitter, `lib/Conversion/MDExecKernels/GroupsBuild.cpp`, with the cell (orthorhombic or triclinic) and the scan of its counting sorts (three launches, or the decoupled look-back of D321) as its parameters, in place of the text generator `scripts/generate-groups-gpu-template.py` and its four checked-in templates `lib/Runtime/Templates/NeighborsGroups{,Triclinic,SinglePass,SinglePassTriclinic}GPU.mlir`; the code of every variant is that of its template, op for op.** Issue #856, part of #558 and #604.

## What moved

The generator wrote about 14,600 lines of MLIR for each of the four variants (59,384 lines in all, with their comments) from 2,758 lines of Python: the bitonic networks of the chunks and of the partners of the excluded pairs written out stage by stage, and the triclinic and single-pass variants made by replacing text in the orthorhombic one. `convert-md-exec-to-gpu` embedded the four files as strings, chose one by the cell and by its option `single-pass-group-scan`, and parsed it. The emitter builds the same functions with an `OpBuilder`, in the order of the text: the declarations of the device primitives, `mdrt_gpu_groups_grid`, `mdrt_gpu_groups_partner`, `mdrt_gpu_groups_image`, in a triclinic cell `mdrt_gpu_groups_image_triclinic` and `mdrt_gpu_groups_offset_triclinic`, and the build. Its functions follow those of the generator: `emitKeys`, `emitChunks` with `bitonic`, `emitBoxes` with `reduceBox`, `emitLists` (the lists, and with `images` the kernel of the images of D115, which the generator made from the text of the lists), `scan`, `cursorScatter`, `sortBins`, and the atomics.

The calls of the device primitives (`@mdir.primitive.*`, D296) stay calls with their declarations, as in the templates, and `expandPrimitiveCalls` replaces them after the emitter has run, as it did after the parse; the emitter writes none of the primitives again.

## Each variant a choice of the emitter

Every parameter of the emitter is a choice made when it runs (the maintainer, on the PR of #856): a variant has the code of its own choices and none of the others', by a branch of the C++, never by a select or a flag that a kernel reads. The triclinic build has the tilts, the windows of the rows across the faces, and the five bits a lattice vector of the shifts of the entries; the orthorhombic build has none of them. The single-pass build declares and calls `single_pass_scan` with its i64 states; the three-launch build neither declares nor calls it. Parameters beyond these two (the size of a group, the lanes) and the reuse of the emitter by the spatial order, search, and pruning of #795 (#817) are for later decisions.

## No loss (#444)

The module of each variant that the emitter writes, printed in the generic form of MLIR (in which the values are numbered by their place), is identical to the parsed template, for all four variants. The builder of `memref.load` and `memref.store` sets `nontemporal = false` where the parser sets nothing; the emitter sets no attribute, as the parser. The programs as executed are identical as well: on the nine systems of the Amber suite and the rhombic dodecahedron of `test/Driver/triclinic.test` with the groups, in mixed and double precision, with the three-launch scan and with `single-pass-group-scan=true`, the module after `convert-md-exec-to-gpu` and the lowered program with the PTX of every kernel (`mdir emit --stage=lowered`, `MDIR_GPU_BINARY=ptx`, `MDIR_GPU_ARCH=sm_89`) are byte-identical to those of `main`. No control key selects the single-pass build; it is reachable through the option of the pass, which `MDIR_PIPELINE` set for the comparison.

## Tests

`test/Conversion/MDExecToGPU/groups-build.mlir` pins what distinguishes the four variants (the declarations, the cell of the build, the three scans, the shifts of the entries, the offsets of the triclinic images) and that none holds the code of another; a variant parameter wired to a constant fails it. The pass `add-groups-build` (options `triclinic`, `single-pass-scan`) adds the build to a module for the tests, replacing a declaration of it; `test/Runtime/neighbors-groups-gpu.mlir` and `neighbors-groups-triclinic-gpu.mlir` run the build through it with their checks unchanged, and `test/Runtime/generated-templates.test` no longer compares the groups.
