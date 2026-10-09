# 13. Limitations and next steps

**Scope of the first milestone.** MDIR runs on one device, a CPU with
OpenMP or one NVIDIA GPU. Triclinic cells (truncated octahedra, rhombic
dodecahedra, the hexagonal cells of CHARMM) run on the CPU and on the
device, with the neighbor matrix or the groups, at constant volume and
pressure (D123, D125 to D127). A cell must be at least twice the cutoff
wide along its diagonal; the reach of the neighbor matrix is not bound by
the cell (D241), and that of the groups must stay below the
least edge of the cell, in an orthorhombic cell as well, or the run is
refused or stopped (D242). The barostat
rescales the cell isotropically, semi-isotropically, or along three
independent axes (D163c), with no coupling of the shape of the cell. Topologies are read in the formats of Amber,
GROMACS, and CHARMM; a CHARMM force field runs with its Urey–Bradley
angles, harmonic impropers, and force switch (D121, D122), but not with
lone pairs or the Drude model. Electrostatics are particle mesh Ewald, a
cutoff, a reaction field (D140), or generalized Born (D144, D152), and the
dispersion may be summed by particle mesh Ewald as well (D162); there is no
polarizable model and no learned potential. Terms over triplets (D160) run
on the CPU only.
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
(`docs/roadmap.md`) orders what follows into milestones: the Python API
(M2a), differentiable simulation (M2b), learned potentials on one GPU (M3),
and distributed execution (M4), with classical features between them.

*In place since the first milestone.* Production runs on a cluster: `steps`
as the length of a run that `mdir run --continue` carries over as many jobs
as it takes, stops on a signal or a limit of time that fall on a checkpoint,
outputs that form one system and continue with the run (D129 to D132,
D149), and an optional manifest of provenance (D168). Dynamics: Langevin
dynamics by the middle scheme (D135), Nosé–Hoover chains, Brownian
dynamics, and anisotropic cell rescaling (D163). Terms given by expressions
over tuples, the pairs of a topology, the centers of groups, and single
particles, with parameters of each particle, tabulated functions, and
compound terms (D136 to D139, D145, D148, D165); terms over triplets on the
CPU (D160); their energies and generalized forces as observables (D189).
Electrostatics and dispersion: the reaction field (D140), runs without a
periodic cell (D142), generalized Born (D143, D144, D152), and particle mesh
Ewald for the dispersion (D162). Alchemical free energy, whose
$dH/d\lambda$ the differentiation of the IR gives (D2, D161).

*M2a, the Python API* ([plan](../python-m2.md)). In place: owned loaded
data, a typed model, and the preparation that the control file shares
(D191); lowering to immutable IR (D192); read-only NumPy arrays at the
boundary (D193); persistent simulations that run any number of steps and
continue the coupling phase, with failures that return to Python (D196),
host code compiled by one machine (D197), and JIT memory and unwind frames
owned and checked before registration (D199; LLVM's relocation and the
frame index of libgcc remain assumptions, [JIT ownership](../jit-invariants.md));
initial velocities drawn by
the CLI's code and typed positional restraints (D198); unit quantities of
OpenMM at the setters (D200); the minimizer of `mdir run` in a
simulation, in parts that carry its step length (D202), with a tolerance
on the largest force that both check at the rows of the energies
(D219); reporters whose
times are arguments of the compiled program, so that the energy file and
the trajectory of `mdir run` are written inside a part (D207);
tunable parameters, whose new values a simulation takes without compiling,
an update equal to a compile with them to the bit (D213), the table of the
Lennard-Jones among them by pairs of types (D226); parts that
continue one activation of the entry, whose buffers stay on the device, so
that a run in parts is the run in one part to the bit
(D215); read-only DLPack views of those buffers, whose leases
block runs while a consumer holds them (D220), and writable
borrows of them with an explicit commit, which rebuilds what depends on
the fields written (D229), the tilts of a triclinic cell among them
(D238); read-only views
of the topology, with the constraints of a compiled program, and the masks
of the control file evaluated from Python (D221); checkpoints
shared with the CLI, which either front end continues
(D223); the package `mdir` as a manylinux_2_28 pip wheel for Python
3.10–3.13, which carries its runtime and libdevice, takes cuFFT from
NVIDIA's wheel, and runs the four-stage tutorial from the installed
package at the rate of `mdir run` (D228); `observe` on pair, tuple, and
external terms of the Python model, with the observables file of `mdir run`
written by a reporter and the values in the state (D232,
Section 6.8), and terms of the absolute positions (D233).
Remaining:
in the Python model, the time and parameters of each particle in a term of
the positions, terms over centers and compound terms, and the derivative
in a tunable of several entries at every row of a run, which `observe`
does not give (the frame evaluator of M2b);
NPT in a triclinic cell or with a coupling period of 1.

*M2b, differentiable simulation* (D195). The parameters of a potential
fitted to ensemble averages by reweighting stored frames
[[ThalerZavadlav2021]](references.md#thalerzavadlav2021): MDIR samples and
evaluates the energy at the frames with its derivative in the parameters;
the reweighting and the loss stay in the framework, through adapters for
PyTorch first and JAX second. DLPack shares storage, not gradient graphs,
so each adapter needs explicit derivative rules. Begun
(D230, Section 3.6): the derivative of the energy in the
tunables at the state of a simulation, for constants of pair terms,
parameters of tuple terms, $\sigma$ and $\epsilon$ per type and by pairs
of types, and the charges, under a Coulomb cutoff and with particle mesh
Ewald (D231); and the evaluation at stored frames with
its vector-Jacobian product in the tunables, as an operation of PyTorch
(D240, Section 3.6), at one start of an activation per frame:
1.05 ms on the dipeptide in water and 6.85 ms on JAC, 15 and 31 steps.
Remaining: the frames inside one activation, at a few steps each; the
gradients in the positions and the strain as outputs; the derivative of
the virial and of observed terms in the parameters, which a fit to the
pressure needs; and the JAX adapter.

*M3 and M4.* Learned potentials behind a versioned interface, one model on
one GPU first. The distributed work begins with a graph of the
dependencies of the `md` ops, ownership and freshness of fields, and a
verifier of two domains, in which the disjoint union of the constraints is
the unit of ownership (D83).

*Between milestones.* Coarse-grained models (tabulated potentials,
Martini, DPD); native collective variables and biases; terms over triplets
on a device; outputs for analysis (frames of the velocities, the pressure
tensor); fixed-point sums in the loop over groups, so that the
deterministic mode keeps its rate. Analysis comes last: frames correct by
construction (molecules whole, the solute in one image) and observables
compiled into the run, sharing its loops; on one trajectory compared
across analysis programs, what went wrong was images and conventions,
never the arithmetic.
