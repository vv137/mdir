# D[kernel-names]

Recorded at its merge.

**A kernel is named after its function, the op of the loop it comes from, what a fused kernel holds, and its ordinal in the lowering, not after the line of that op in the text of the module** (#568; the ruling of the maintainer on #568, 2026-10-26).

**Before.** `MDExecToGPU`'s `setPurpose` appended `_l<line>` when the op's location was a `FileLineColLoc`: `mdir_run_particle_for_l2451_integration3_188`. That line is the line of the op in the builder's text. No decision row recorded this; it was described in `docs/debugging.md`, `docs/principles.md`, the roadmap, and the white paper (Section 12), and it came with the tiers of the sanitizer and of the Amber suite.

**Why it changes.**
- A kernel's name changed whenever a line was added above its loop in `Builder.cpp`, or when the module was printed differently.
- Step 2 of #568 (#645) prints the module with MLIR's printer, and it had to keep the builder's positions as locations only to keep these names.
- A loop that a later stage builds with `OpBuilder` would have had to be given a made-up position.
- The gate of #568, that the lowered program stays byte-identical, failed on every such change. Those failures said nothing about the program.

**Now.** `setPurpose` takes the op's name alone (`particle_for`, `tuple_for`).
- `nameKernel` appends what it appended before: the kind of a fused kernel (`_integration3`, `_run2`), and the ordinal of the kernel among the kernels of the lowering of the module, `numKernels`.
- **Unique:** the function and the ordinal are unique within the module, as before; the line was never needed for that.
- **Deterministic:** the ordinal follows the walk of the lowering over the ops, the same order in which it outlines them. The name therefore depends on the ops of the program and their order, never on locations or text.
- **Alternatives not taken.**
  - A hash of the op's structure would survive the insertion of an unrelated kernel before it, but it is opaque in a profile.
  - An ordinal among the ops of one kind would read better, but the global ordinal was already there and is what the reports print.
- **What a reader loses** is the pointer from a kernel to a line of `mdir emit`. `MDIR_PRINT_AFTER=gpu-kernel-outlining` prints the kernels in the order of their ordinals, and `docs/debugging.md` says so.

**What changes.**
- The lowered program and its PTX change once, in the names of the kernels alone.
- No result changes.
- The compile cache's key hashes the module, which this does not change. It does not hash the lowering's code, so a host object that an earlier build cached keeps its old kernel names. That object is self-consistent: the runtime launches the kernels by the names the object holds.

**Validation** (PR):
- **Names only:** across the #606 corpus, main's lowered programs and the branch's are identical once `_l<line>_` is removed from main's names, 488 of 488 cases. Without that, the 143 GPU cases with kernels differ.
- **Results:** deterministic checkpoints of JAC NVE, Factor IX NPT, Cellulose NVE, the ala3 minimization, and argon on a GPU are identical to the bit between main and the branch.
- **Suites:** the full suite and the sanitizer tier pass.

After this, the positions in the builder's text no longer bear on the lowered gate of #568 (`docs/driver.md`, Section 2.11).
