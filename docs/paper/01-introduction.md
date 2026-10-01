# 1. Introduction

A molecular dynamics engine is, for the most part, a fixed program that
reads a description of a system and of a protocol and interprets it. The
force field arrives as tables of parameters, the integrator as a switch,
the neighbor list as a data structure of the engine, and the kernels that
run on a GPU are written once, by hand, for the combinations of terms and
options that the authors foresaw. The engines that set the standard of
speed on GPUs, Amber's pmemd.cuda [[SalomonFerrer2013]](references.md#salomonferrer2013) [[LeGrand2013]](references.md#legrand2013),
GROMACS [[Abraham2015]](references.md#abraham2015) [[Pall2020]](references.md#pall2020), and OpenMM [[Eastman2017]](references.md#eastman2017), each
represent years of such kernels.

MDIR takes the other route: it is a compiler. A run is described in
intermediate representations (IR) built on MLIR [[Lattner2021]](references.md#lattner2021), in
dialects whose objects are those of molecular dynamics, particle sets,
tuple sets and the relations that name their members, fields over either,
neighborhoods, potentials as expressions, and programs of a time step.
The forces are derived from the potential by differentiation in the IR,
the loops over particles, pairs, and tuples are chosen and fused by
passes, a precision is assigned to every value by its role, and the
result is lowered to code for a CPU or a GPU. The compilation happens
before every run, with the topology, the force field, the cutoffs, and
the protocol in hand, so the code that runs is specialized to them: a
term absent from the run is absent from the kernels, and a constant of
the force field is a constant of the code.

A compiler is only worth this generality if what it produces is as fast
as the hand-written engines and as correct. The first milestone of MDIR
(M1) set a concrete target: at least the rate of pmemd.cuda (Amber 26,
SPFP) on every system of the Amber GPU benchmark suite, from 23,558 to
1,067,095 atoms, at constant energy and at constant pressure, on the same
GPU, in mixed precision, without giving up energy conservation. This
paper describes the system that reaches it.

## Contributions

1. **A compiler for molecular dynamics** (Section 3): semantic dialects
   for what is computed (`md`) and how the state advances (`dyn`), an
   execution dialect of loops over particles, pairs, and tuples
   (`md_exec`), a runtime dialect (`mdrt`), and the passes between them:
   differentiation with exchange contracts, fusion, the reuse of neighbor
   structures across steps, the assignment of precision and of storage,
   and lowerings to OpenMP and to NVIDIA GPUs.
2. **Exact neighbor structures** (Section 4): a test of validity that
   holds under the scaling of a barostat (with proof); groups of 16
   particles that compute each pair once, in frames that need no minimum
   image; and a dual list whose inner list is pruned from the outer one
   whenever its own exact test asks, packed in the order of the outer
   list.
3. **Numerics of mixed precision** (Section 7): a policy that assigns a
   type and the license to approximate to each role of a value; and the
   analysis and removal of a drift of the energy, which showed that a
   random error in the constraints acts as a thermostat at infinite
   temperature, and that a closed-form solver of rigid water loses in f32
   what an iterative one keeps.
4. **Kernels from loops** (Section 8): the lowering of loops onto warps,
   an integration kernel that fuses the kick, the drift, and the
   constraints by giving each warp whole constraint groups, and
   asynchronous reads of flags that keep the device busy while the host
   decides.
5. **Results** (Sections 9 and 10): agreement of the energy terms and of
   the force on every particle with sander and GROMACS; conservation of energy over 2 ns of JAC better than
   that of pmemd.cuda on the same input; on an RTX 3090, a rate between 102%
   and 127% of pmemd.cuda's on every system of the Amber suite; and on
   ubiquitin in OPC, 64% to 72% of the rate of GROMACS, with the loop over
   pairs as fast as GROMACS's and the difference in the rest of the step.

## How the paper is organized

Section 2 fixes the notation. Section 3 describes the design of the
compiler and its runtime. Sections 4 to 6 derive the methods as they are
implemented: neighbor structures, the electrostatics of Ewald summation,
and the integrators with their constraints and couplings. Section 7 states
the precision policy and the drift that shaped it, and Section 8 the
lowering to a GPU. Section 9 collects the evidence of correctness and
Section 10 the measurements of performance. Section 11 places MDIR among
related work, Section 12 states the principles of development and the
defects that taught them, and Section 13 the limitations. The appendices
give the control file and a guide for contributors.
