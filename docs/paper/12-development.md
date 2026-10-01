# 12. Principles of development

The principles below were adopted after defects reached runs of real
systems, and each is tied to the case that taught it
(`docs/principles.md`).

**1. A fact that a pass relies on is in the IR.** A pass may rely only on
what the IR states. When a buffer of the members of a SHAKE set served as
scratch for the particles of Cellulose, kernels wrote beyond it, because
the storage form did not say which set a buffer belongs to; the pools now
key buffers by set. That the groups of SETTLE and SHAKE share no atom was
known only to the driver; the IR now states it (`md.disjoint_union`, D83),
and a pass relies on it to let a constraint loop read a field before the
updates of the other sets. A fact in the IR is also a license: the driver
proves `disjoint`, and the lowering evaluates each tuple once because the
IR says it may.

**2. An optimization states its precondition, and the precondition is
checked.** Fusion of loops requires the same positions, no read of what an
earlier loop writes, and buffers for sums of their own; the fusion checks
each. An optimization that reorders or overlaps work is exact or wrong, and
no test of a few systems tells a race that did not happen from one that
cannot: such an optimization carries an argument in its decision, a claim,
a proof, and a list of premises each with the place where it is checked.
The second stream of D87 is the pattern (Section 8.3). An optimization that
changes results beyond the rounding of sums, such as a list kept for a
fixed number of steps (D88), is a mode that is off by default and warns
when a run chooses it.

**3. Every optimization can be turned off, and the two are compared.**
Fused and separate loops, the integration kernel and the loops apart
(`fuse-integration`), groups and the matrix, the dual list and one list:
each has a switch, and a test compares both ways, bitwise where the sums
have a fixed order, to the rounding of the sums otherwise.

**4. Numerical assumptions are measured on the hard cases.** SHAKE by 12
sweeps of relaxation assumed that $m_\text{H}/m_\text{X}$ is small; masses
of hydrogen repartitioned to 3 amu made it about 1/2, and steps of 4 fs
became unstable. SHAKE is now solved by Newton's method on all the bonds of
a group (Section 6.2). The drift of Section 7.3 is a second instance: an
approximation bounded where it was introduced (a term of the potential)
was unbounded where a pass also applied it (an operator divided by the
step). Tests include repartitioned masses, steps of 4 fs, hot starts,
and large cells, and since D112 the conservation of energy is also
measured over 2 ns against pmemd.cuda (Section 9).

**5. Scale is a separate axis of testing.** Defects in sizes and indices
show only where counts differ: more particles than tuples, counts that are
not multiples of a block, rows longer than a warp, more excluded partners
than a warp can hold (D106, found on Cellulose and nowhere else in the
suite). Short runs of the Amber suite are a tier of the tests
(`test/Scale`).

**6. Tools check what tests cannot.** The GPU tests run under
compute-sanitizer (`lit -Dsanitize=1`); the host code is built with
AddressSanitizer and UndefinedBehaviorSanitizer; kernels are named after
the op and the line of the module they come from, such as
`mdir_run_particle_for_l279_integration3_82`, so that a profile or a report
of the sanitizer points at the IR. A hang of the device at 100% utilization
and half the power was found with `cuda-gdb` to be a sort of every particle
into one bin after positions had become NaN (D106, D107).

**7. Inputs are varied on purpose.** The readers are tested with
topologies in the format before Amber 7, without atomic numbers, with
repartitioned masses, and as written by converters; a reader rejects what
it cannot read with a message.

**8. Accuracy is a gate, not a by-product.** A change to a kernel is
checked against the energies of the log in double precision and against
the conservation of energy of the tests. A rate is reported with the
conservation of the run that measured it (Section 10).
