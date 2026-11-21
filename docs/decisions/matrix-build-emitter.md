# D[matrix-build-emitter]

Recorded 2026-11-21.

Emit the GPU neighbor-matrix build with MLIR builders in the style of the
GROUPS build. Preserve the row and entry order, precision specializations,
launch geometry and control-file interface. Keep the CPU template as the
ordered-row oracle. Use the existing grid, bin, scan and warp compaction
operations where their lowering reproduces the baseline.

Acceptance requires identical lowered corpus output and byte-identical
matrix-build PTX against the same main revision and toolchain. The
cell-cooperative traversal will be measured in the subsequent M1 change.
