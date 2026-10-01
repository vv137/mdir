# 13. Limitations and next steps

**Scope of the first milestone.** MDIR runs on one device, a CPU with
OpenMP or one NVIDIA GPU. Cells are orthorhombic; the barostat is
isotropic (a semi-isotropic coupling is designed but stops with "not
supported yet"). Electrostatics are particle mesh Ewald or a cutoff; there
is no implicit solvent, no polarizable model, and no learned potential.
The distributed lowering (`md_dist`), the joint planner that would choose
structures and layouts, and the `mlff` and `ensemble` dialects of the
architecture are designs, not code.

**The deterministic mode does not take groups.** The loop over groups adds
with floating-point atomics; in the deterministic mode a run keeps the
neighbor matrix, whose sums have a fixed order, and loses the rate of
Sections 4.4 and 4.5. Fixed-point accumulation in the loop over groups, as
the spreading of PME has (D70), would close the gap.

**Open observations from the measurements of this paper.**

- *A transient at the start of the runs of JAC, explained.* It comes
  from the inputs of the suite, and pmemd.cuda has it too. At 2 fs the
  total energy falls by about 6 kcal/mol in the first 3 to 5 ps and is
  then flat in both programs (MDIR from $-58163.7$ to about $-58169$;
  pmemd.cuda from $-58134.4$ at 1 ps to about $-58140.5$), and in MDIR in
  double precision as well (Figure 7.1, right). The input at 4 fs
  (`JAC_production_NVE_4fs`) is the restart file of the input at 2 fs,
  the same coordinates and velocities, with a topology whose hydrogen
  masses are repartitioned: hydrogens of three times the mass move at the
  speeds of light ones, so both programs start at 568 K. In the first 50
  steps both relax to about 404 K, and the total energy rises by 160
  kcal/mol in MDIR and 177 in pmemd.cuda (from its step 1), the error of a
  step of 4 fs at that temperature, before both run flat at about 401 K.
  The change of the energy that Section 10 reports for these runs
  measures the transient, not a drift.
- *Groups against the matrix in a narrow cell, found and fixed.* The
  difference between the groups and the matrix on the argon–krypton
  mixture of `barostat.test` ($7\times10^{-4}$ kcal/mol in the potential
  energy, $1.4\times10^{-6}$ relative, and 0.15 kcal/mol in the virial)
  was a defect of the build: an entry took the image of its candidate
  nearest to the center of its group, which misses pairs in a cell
  narrower than twice the reach and the extent of a group (Section 4.4,
  D115). The build now keeps an entry for each image within the reach,
  and the two agree to every printed digit.
- *The order of the state decays.* Over 2 ns of JAC the rate fell by 2%
  and then stayed flat: the state is sorted where a run or a segment
  begins (D44), and only the loops over pairs read it in an order renewed
  at every build (D86). Sorting the state at intervals, with the members
  of the tuples renumbered on the device, is planned with the domain
  decomposition, which needs it anyway.

**Measurements.** The rates of Section 10 are from one RTX 3090 capped at
300 W in a node whose other GPUs ran other users' jobs; the device of the
measurements was not shared (the benchmark script checks), but the node's
power and host were. The cap matters: under it, MDIR's kernels ran at a
lower clock than pmemd.cuda's (about 1590 against 1695 MHz on Cellulose),
so a kernel that does the same work with less power gains speed.

**Next steps.** The white paper closes the first milestone. The next
milestones are the distributed lowering with a domain decomposition (in
which the disjoint union of the constraints is the unit of ownership,
D83), learned potentials with a halo strategy chosen per model, the
deterministic loop over groups, and a Python interface whose buffers
follow DLPack, so that the state is shared with machine-learning
frameworks without copies.
