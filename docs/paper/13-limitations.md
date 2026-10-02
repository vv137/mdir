# 13. Limitations and next steps

**Scope of the first milestone.** MDIR runs on one device, a CPU with
OpenMP or one NVIDIA GPU. Triclinic cells (truncated octahedra, rhombic
dodecahedra, the hexagonal cells of CHARMM) run on the CPU and on the
device, with the neighbor matrix or the groups, at constant volume and
pressure (D123, D125 to D127). The barostat
is isotropic or semi-isotropic, with no coupling of the shape of the
cell. Topologies are read in the formats of Amber,
GROMACS, and CHARMM; a CHARMM force field runs with its Urey–Bradley
angles, harmonic impropers, and force switch (D121, D122), but not with
lone pairs or the Drude model. Electrostatics are particle mesh Ewald or a cutoff; there
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
- *The conserved energy at constant pressure drifted with rigid groups,
  found and fixed.* On 1039 rigid OPC waters the default count of the
  work of the barostat, of Trotter type (Section 6.4), drifted by about
  $-230$ kcal/mol/ns at every period of coupling, where the exact count
  drifted by about $+10$ and GROMACS by $-0.4$; with flexible water the
  counts agreed. It took the virial of the step with twice the internal
  kinetic energy, which is the virial of the groups only when the forces
  of the constraints are those of one configuration, and in the step that
  scales they straddle the scaling. Counted from the virial of the groups
  of the evaluations (D116), the drift is $+12.7 \pm 13$ kcal/mol/ns on
  the water box and $+11.4 \pm 33$ on ubiquitin in OPC, from $-228$. The
  sampling was never in question (Section 9.5): the count changes nothing
  in the trajectory.
- *Slower than GROMACS on a protein in OPC.* On ubiquitin in OPC with a
  cutoff of 9 Å (Section 10.6) MDIR's rate is 72% of GROMACS's at constant
  energy and 66% at constant pressure, though its loop over pairs costs
  the same as GROMACS's nonbonded kernel and its PME less. GROMACS runs
  PME beside its nonbonded kernel and the bonded terms, the update, and
  the constraints on the host; MDIR runs them in series on the device
  (123 µs of the device in a step of 310), and the host's time between its launches is
  48 µs. Fewer and fused launches outside the loop over pairs, and work
  beside it, are where the rest of the step is.
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

**Next steps.** The white paper closes the first milestone; the roadmap
(`docs/roadmap.md`) orders what follows. What a production run on a
cluster needs first is in place: `steps` as the length of a run that
`mdir run --continue` carries over as many jobs as it takes, a trajectory
that continues with it, a stop on a signal or a limit of time that falls
on a checkpoint so that the continuation stays exact, and the checkpoint
before the last kept (D129 to D132); the log in a file of the run is
next. Langevin dynamics by the middle scheme is in place (D135), without
yet a conserved energy, and so are terms given by expressions over bonds,
angles, and dihedrals, which give restraints beyond positions and the
dihedrals of OPLS-AA (D136), and terms over the pairs of a topology, in
its charges and Lennard-Jones parameters and between interaction groups
(D137), with tabulated functions of one argument (D138), and terms over
the centers of groups, which restrain pull groups (D139), whose loops
still run over every particle of the system; so are the reaction field
(D140) and runs without a periodic cell, in a cell that no image reaches
(D142). Then the features that general molecular dynamics asks of MDIR:
such terms over positions, with parameters of each particle, and pulling
at a rate; generalized Born solvation; outputs for
analysis (velocities, a compressed trajectory, the pressure tensor);
coarse-grained models; and free energy, whose $dH/d\lambda$ the
differentiation of the IR is designed to give (D2). A Python interface whose buffers follow DLPack
shares the state with machine-learning frameworks without copies, from
the same IR and validation as the control file. The distributed work
begins with a graph of the dependencies of the `md` ops, ownership and
freshness of fields, and a verifier of two domains (in which the
disjoint union of the constraints is the unit of ownership, D83), then
a potential with an intermediate field and learned potentials behind a
versioned interface. The deterministic loop over groups is open as well.
Analysis comes last: frames correct by construction (molecules whole,
the solute in one image) and observables compiled into the run, sharing
its loops; on one trajectory compared across analysis programs, what went
wrong was images and conventions, never the arithmetic.
