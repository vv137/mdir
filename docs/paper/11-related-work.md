# 11. Related work

**Engines with hand-written GPU kernels.** pmemd.cuda, the GPU engine of
Amber, sorts the particles into boxes and along a space-filling curve and
takes them in groups that share a list of neighbors
[[SalomonFerrer2013]](references.md#salomonferrer2013), and accumulates forces in fixed point so that its
sums do not depend on the order of the threads (SPFP, [[LeGrand2013]](references.md#legrand2013)).
GROMACS computes pairs over clusters of particles with masks of
interactions [[Pall2013]](references.md#pall2013), keeps a pair list for a fixed number of steps
with a buffer estimated from a tolerance of the drift of the energy, and
prunes an inner list dynamically [[Pall2020]](references.md#pall2020) [[Abraham2015]](references.md#abraham2015). LAMMPS
[[Thompson2022]](references.md#thompson2022) and HOOMD-blue [[Anderson2020]](references.md#anderson2020) are extensible engines for
particle-based materials and soft matter. MDIR's groups of 16 (Section 4.4)
follow the layout of [[SalomonFerrer2013]](references.md#salomonferrer2013); its fixed-point spreading of
charges in the deterministic mode follows [[LeGrand2013]](references.md#legrand2013); and its dual
list (Section 4.5) shares the purpose of the dynamic pruning of
[[Pall2020]](references.md#pall2020) but prunes when an exact test fails rather than on a schedule
sized by an estimate. The difference from all of them is that MDIR does
not contain kernels: it generates them from a description of the
potential and of the step, for the run at hand.

**Engines that generate code.** OpenMM [[Eastman2017]](references.md#eastman2017) lets users
describe a force by algebraic expressions of the relevant variables, which
it parses, analyzes, and compiles just in time into an implementation of
that interaction.
PPMD [[Saunders2018]](references.md#saunders2018) is the closest in structure to the execution layer
of MDIR: the user writes the algorithm in Python and the computation per
particle or per pair as a short kernel in C, and the framework generates
the loops around it, compiles them at run time, and runs them on CPUs with
MPI or on GPUs; access descriptors tell it what a kernel reads and writes,
and it follows OP2 [[Mudalige2012]](references.md#mudalige2012) and PyOP2 [[Rathgeber2012]](references.md#rathgeber2012), which do the
same for meshes. MDIR's `md_exec.particle_for` and `md_exec.pair_for` take
PPMD's particle and pair loops, but their bodies are regions of IR that the
compiler can analyze and differentiate, rather than opaque kernels, and
MDIR adds the layer that PPMD does not have: the energy as an object of
the IR (`md`), from which the forces, the loops, and their access patterns
are derived. AutoPas [[Gratl2022]](references.md#gratl2022) selects at run time among containers,
traversals, and layouts of short-range particle interactions, and Cabana
[[Slattery2022]](references.md#slattery2022) is a library of particle data structures, neighbor lists,
and halo exchange for portable performance.

**Differentiable and compiled frameworks.** JAX MD [[Schoenholz2020]](references.md#schoenholz2020)
expresses molecular dynamics in JAX and obtains forces by automatic
differentiation of the energy; Enzyme [[Moses2020]](references.md#moses2020) differentiates at the
level of LLVM IR. MDIR differentiates in its own dialect (Section 3.3),
where the derivative of a sum over pairs or tuples is again a gather over
the same relation and carries an exchange contract that later passes use.

**Compilers on MLIR.** MLIR [[Lattner2021]](references.md#lattner2021) provides the infrastructure of
dialects, passes, and lowerings that MDIR is built on; MDIR contributes
dialects whose objects are those of molecular dynamics rather than tensors
or loops.

**Learned potentials.** Strictly local equivariant potentials such as
Allegro [[Musaelian2023]](references.md#musaelian2023), and message-passing potentials whose features
cross the boundaries of domains at each layer, as in [[Park2024]](references.md#park2024), set the
requirements that a planned `mlff` dialect and a distributed lowering will
have to meet (Section 13). They are not part of the first milestone.
