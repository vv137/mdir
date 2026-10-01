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

- *A transient at the start of the runs of JAC.* From the restart files
  of the suite, the total energy of JAC at 2 fs falls by 5.7 kcal/mol in
  the first 40 ps and then stays within the scatter of the rows; the run
  in double precision falls by as much (Figure 7.1, right), so the
  transient comes from the input, not from the precision. At 4 fs
  with repartitioned masses (`jac_nve_4fs`) the restart starts, by MDIR's
  count of the degrees of freedom and masses, at 567 K; the total energy
  jumps by about +154 kcal/mol in the first 1000 steps and stays flat
  within $\pm 7$ kcal/mol over the remaining 9000. The jump is the same
  with one list and with the dual list. Whether it comes from velocities
  of the file that do not satisfy the constraints, or from the masses
  with which they were written, is not yet established; the change of the
  energy that Section 10 reports for these runs of 10,000 steps measures
  the transient, not a drift.
- *Groups against the matrix with a switched Lennard–Jones potential.* On
  the argon–krypton mixture of `barostat.test` in double precision, the
  potential energy at step 0 differs between the groups and the matrix by
  $7\times 10^{-4}$ kcal/mol ($1.4\times 10^{-6}$ relative); on the water
  boxes of the tests, with particle mesh Ewald and no switch, the two agree
  to every printed digit. The cause is not yet known.
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
