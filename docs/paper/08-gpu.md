# 8. Lowering to a GPU

`convert-md-exec-to-gpu` turns the loops of the `md_exec` dialect into
kernels and host code (Section 3). This section describes how the loops
map onto threads, two fusions that matter for the rate, and how the host
and the device are kept from waiting on each other. The target of the
first milestone is NVIDIA hardware through the `gpu` and `nvvm` dialects of
MLIR, compiled at run time (Section 3.6).

## 8.1 From loops to kernels

| Loop | Threads | Writes |
|---|---|---|
| Over particles (`md_exec.particle_for`) | One a particle | Its own particle; global sums through partials per block, then a reduction in a fixed order |
| Over pairs, the matrix | 16 lanes a row (`row-lanes`), owner computes | Its own particle, no atomics |
| Over pairs, groups | A warp a block of 64 entries (Section 4.4) | Atomic additions for the entries, one per block for the group |
| Over tuples that may share particles (bonds, angles, …) | A thread a tuple; with global sums, a thread a particle, the tuples $p, p+n, \dots$ evaluated once each (D103) | Atomic additions to the members; from 100,000 tuples, additions to a table of the particles of the block in block memory, then one atomic addition a particle a block (D283); sums of its own tuples |
| Over disjoint tuples (constraints) | A thread a particle; the member at place 0 evaluates the tuple (D83) | Every member of its tuple, which no other thread writes |

**Runs.** Consecutive loops that take the same positions and do not read
what another writes become one kernel: the loops over pairs and over
tuples of a term of the potential (rows of the same particle), the loops
over the disjoint tuple sets of a union of constraints (D83), and the
integration runs of Section 8.2. Adjacent loops over tuples that evaluate
each tuple once and have no global sums, the bonded terms of a step and the
forces of a term over centers among them, are launched as one kernel over
all their tuples, each thread finding its loop by the running totals of the
tuples (D181); on JAC the five bonded kernels of a step
become one. A run is lowered where its last loop is.

**The sums of a block** (D283). In that kernel the global
atomics, three for the force on each member of each tuple, were about
70% of its time: on Cellulose it took 223 µs a step, and 65 µs with plain
stores in their place (for timing, wrong forces). The tuples of a block
of 128, in the order of the files, touch about a quarter as many distinct
particles as they have members (0.24 on Cellulose, 0.31 on JAC). A block
therefore keeps a table of its particles in block memory, found by a
multiplicative hash and the next slots in turn
[[Knuth1998]](references.md#knuth1998) and claimed by a compare-and-swap;
a thread adds its values there, and after a barrier each particle of the
table is added to the forces once. A search gives up after 8 slots and
adds that value to the forces directly, which bounds a block whose tuples
share few particles. On an RTX 4090 the kernel takes 84 µs a step on
Cellulose and 189 on STMV, where it took 221 and 439, and Cellulose runs
9 to 10% faster. A run of fewer than 100,000 tuples keeps the atomics,
chosen on the host by the count: with the table JAC's kernel, whose
blocks the device holds all at once, took 13 µs where it took 12, and
the suite ran 1.5% slower. The forces differ by the rounding of their sums, as two runs with
atomics differ: by $10^{-6}$ of the RMS force on the systems of the
suite. The excluded pairs, half the tuples of Cellulose, stay in this
kernel: evaluated by the rows of their particles, without atomics, they
took as long as with them.
Section 8.5 derives the conditions under which the table gives the sum
of the atomics.
Destinations that the forces of a step add are merged where a loop gives
two of them by the same contract: the forces of the sums over the centers
of a term (D150), and those of a term over the pairs of a topology fused
into the loop of the nonbonded pairs, which then adds one contribution to
each particle instead of two (D157).

**Global sums.** A loop writes the contribution of each row to a buffer
for each sum; a kernel of at most as many blocks as a block has threads
(128) adds the rows into partials, each thread every so manyth row, and a
kernel of one block adds the partials, so that the order of a sum depends
only on the numbers of rows and parts. A block adds all the sums of its
kernel in one pass: each warp by a butterfly of shuffles, then, after one
barrier, the four warps in their order from shared memory (D150). With at
most 512 rows the first kernel has one block, which writes the results,
and the second is not launched; a loop that evaluates each tuple once
launches no more rows than it has tuples, so the sums over the centers of
groups, some 170 tuples on JAC, are one block. That block also computes
the contributions of the rows itself, so that such sums are one kernel
in all (D181). When the sums reach nothing but a later loop that
evaluates each tuple once, through arithmetic, as the centers of a term
over the centers of groups reach its forces, a block of the kernel of the
run of that loop does the work of that kernel instead: the same rows, added
in the same order, then the arithmetic, then the tuples of the forces,
while the other blocks evaluate the other bonded loops; with more than
512 rows the kernels of several parts still run, and leave the totals in
a buffer for that block (D194). A harmonic bias on the
distance between two centers then costs JAC no kernel of its own and
2.7 µs a step, against 9.2 µs with its sums in a kernel of their own and
8.6 µs for OpenMM 8.2's `CustomCentroidBondForce`. A kernel passes each
buffer as its descriptor, and PTX before ISA 8.1 allows 4352 bytes of
parameters: the sums of a loop go to the kernels in runs whose
descriptors fit 3 KB (D145). The results go to a buffer of the device;
the host copies them only where it reads them, for the log. A kernel
whose arguments are arithmetic of the sums, such as the weights of the
forces of a term over the centers of groups, evaluates that arithmetic
itself from the device's buffer, so that a step does not wait for its
sums (D156).

**Counting in 32 bits** (D85). MLIR's `index` is 64 bits on NVPTX. Device
code counts particles, cells, tuples, and entries in `i32`, and only a
flattened element number or a byte address is 64 bits: the search of the
matrix needed more than 64 registers in `index` and spilled (612 µs a
build of JAC, against 430 in `i32`).

**Divisions and reciprocals** (D98, D180). A division marked approximate becomes
`x * rcp.approx.ftz(y)`: the backend's `div.approx.f32` scales operands
that are subnormal or beyond $2^{126}$, which never occur in the
arithmetic of a pair, at a cost of two comparisons, two selections, and a
product a pair. The product that forms an approximate quotient rounds in
f32 before subsequent arithmetic; it does not contract into a fused
multiply-add with a cutoff shift. Such contraction changed the shifted
soft-core Lennard-Jones parameter derivative by about
$5.5\times10^{-4}$ kcal/mol on the ethanol validation system (issue #48).
Other products may still contract under `fast_math`.

## 8.2 The integration kernel

A step of velocity Verlet with constraints is, between two evaluations of
the forces,

$$
\mathbf v \leftarrow \mathbf v + \tfrac{\Delta t}{2}\,\mathbf a,\qquad
\mathbf x' = \mathbf x + \Delta t\,\mathbf v,\qquad
\mathbf x \leftarrow \mathbf x' + \Delta\mathbf x_c(\mathbf x', \mathbf x),\qquad
\mathbf v \leftarrow \mathbf v + \Delta\mathbf x_c/\Delta t,
$$

where $\Delta\mathbf x_c$ is what the constraints add (Section 6.2). In the
IR these are a loop over particles (the kick and the drift), a loop over
the tuples of each constraint set, and a loop over particles that adds the
corrections; three kernels that write and read every position and velocity
in between: 83 + 98 + 64 µs a step on Cellulose, and 183 for the same
pattern on the velocities. The *integration run* (D110) is one kernel for
all three.

**Within a warp.** A thread computes the loop before for its particle and
keeps the result in registers. The member at place 0 of a tuple whose
members all lie in its warp gathers what the kernel of the tuple takes of
each member by shuffles from their lanes, computes the tuple, and every
member takes its correction back from that lane and computes the loop
after. Because the sets of a disjoint union share no particle, one set of
shuffles serves all the sets: the source lane of member $q$ is
$\mathrm{select}_k(\text{in}_k, \text{lane}(\text{member}_q^{(k)}))$
over the sets $k$. Each lane loads its own inputs once, narrowed to f32
where the kernel narrows them at once, and takes its position relative to
that of the member at place 0 in f64 before narrowing it, so that the
member at place 0 converts nothing and the differences that the kernel
takes are of small numbers. Destinations of the tuples that nothing reads
after the run are not written.

**Whole groups in a warp** (D111). A group whose members straddle the
boundary of two warps cannot be served by shuffles. With 32 particles a
warp, about 2 waters in 32 straddle a boundary. A second kernel that took
them was bound by the latency of one group, whatever their number: on JAC
it took 28 and 20 µs of a step, more than the three loops it replaced.
Instead, the warps of the kernel take variable spans of whole groups.
Let $a_\text{max}$ be the largest arity of the sets of the run, and

$$
S = 32 - (a_\text{max} - 1).
$$

Warp $w$ takes the particles from $b(Sw)$ to $b(S(w+1))$, where $b(n)$ is
$n$ moved up past the group that contains $n$ if that group's members
follow one another and $n$ is not its first member:

$$
b(n) = \begin{cases}
\max(\text{members of } G) + 1 & \text{if } n \in G,\ G \text{ contiguous},\ \min(G) < n,\\
n & \text{otherwise.}
\end{cases}
$$

*Claim.* A warp then holds at most 32 particles, and no contiguous group
straddles two warps. *Proof.* A contiguous group has at most $a_\text{max}$
members, so $n \le b(n) \le n + a_\text{max} - 1$, and the span of warp
$w$ is at most $S + a_\text{max} - 1 = 32$. A contiguous group $G$ with
$\min(G) < b(Sw) \le \max(G)$ would contain $n = Sw$ with $\min(G) < n$,
and then $b(n) = \max(G) + 1$, a contradiction. $\blacksquare$

The lanes beyond the span of a warp take its first particle, take part in
the shuffles, and write nothing; they cost $1 - S/32$ of the lanes, 6% for
water ($a_\text{max} = 3$) and 9% with methyl groups (4). The groups whose
members do not follow one another are listed by their member at place 0,
their members write their values of the loop before, and a second kernel
of one block takes them and clears its count for the next step. For the
members of a group to follow one another, the driver orders the particles
by the position of an *anchor*, the heavy atom of the group (Section 4.2),
and a virtual site by that of the first atom that places it; every group
of the Amber suite is then contiguous, and the second kernel finds nothing
to do. Without the sites, the extra point of an OPC water near a face of a
cell of the order went to another cell than its atoms: ubiquitin in 5700
OPC waters listed 232 tuples a step, 13.8 µs of one block, against 1.5
with them (525 to 543 ns/day). `test/Driver/integration-gpu.test` runs propane with
its hydrogens numbered after its carbons, whose groups are not contiguous,
against the loops apart and the CPU.

**Two runs in a row** (D250). A loop over particles
is the loop after of one run or the loop before of the next, not both: the
kernel of a run computes its loop after once every member of a group has
its correction, and a second run would need those values of every member
before its own tuples. A leapfrog step under Langevin dynamics with
constraints (Section 6.1) has two runs one after the other, the projection
of the velocities after its first half kick and the constraints of the
positions after its drift, and the fusion of loops over particles
(Section 3.4) made of everything between them one loop. That loop closed the
first run, and the constraints of the positions were kernels of their own
as before D110. For a device the pass therefore leaves apart the loop that
closes a run and the one that opens the next (`integration-runs` of
`md-exec-fuse-loops`): the first takes the half kick, the half drift, and
the friction and the noise, the second the other half of the drift, and
each run is one kernel. On JAC the step went from 0.1482 to 0.1447 ms
(RTX 4090, mixed precision), and velocity Verlet, whose two runs have the
evaluation of the forces between them, takes 0.1408: what remains is the
sum of the forces, which closes a leapfrog step as a launch of its own and
which velocity Verlet computes in the kernel of its second half kick.

*Table 8.1. Integration kernels (µs per step, the run of the positions
plus the run of the velocities, RTX 3090).*

| | Loops apart | One kernel, groups across warps in a second kernel (D110) | Whole groups per warp (D111) |
|---|---|---|---|
| JAC | 23 + 19 | 40 + 31 | 13 + 12 |
| Cellulose | 245 + 183 | 161 + 128 | 135 + 107 |

## 8.3 The host and the device

The host issues work on one stream and waits only where it reads what the
device computed. Three mechanisms keep the device busy.

**A pool of device memory.** The runtime keeps freed blocks and hands one
of the same size to the next allocation without a call to the driver;
buffers that a step allocates and frees cost a lookup. A block is reused
only for a request of its own size, so the sizes that a step asks for must
not follow the builds of the neighbor structure (D312): the
buffers that a loop over the groups gathers its positions and fields into
take the room of the structure's places, which changes only when it
grows, and not the number of places that the last build occupied, which
moves by 64 at nearly every build. Sized by that number, they left blocks
in the pool for every new number, 783 MiB at STMV after 1,000 steps and
more the longer the run; a buffer of the runtime that a growth replaces
goes back to the driver, after the wait that the growth makes anyway. With
both, STMV takes 5.6 GB of an RTX 4090 against 6.5 GB, and its fourfold
replica (4.27 million atoms) runs in 20.8 GB, where it ran out of memory.

**Ragged rows** (D314). The incidence structure of a tuple
set (Section 3.6) was a table of $N$ rows, each as long as the row of the
particle with the most tuples: 247 values for the dihedrals of STMV,
where most particles have a few. It is now one buffer of the offsets of
the rows and their entries, and every kernel and template reads a row
through one accessor. STMV's ten structures take 228 MiB instead of
2,096; the device peak falls from 5.6 to 3.7 GB, the fourfold replica
takes 12.9 GB, and the sevenfold (7.47 million atoms) runs in 22.9 GB. The
kernels of the integration runs, which read the first entry of every
particle's row, now read adjacent entries: Cellulose and STMV step 2% and
4% faster.

**Offsets of the groups** (D321). The counting sorts
that build groups require an exclusive prefix of their integer bin counts.
Their previous scan reduced the counts by tile, scanned the tile totals,
then read the counts again to write the offsets, in three launches. A
single-pass scan reads each count once. Running blocks claim consecutive
tile numbers and publish an aggregate of their counts; a block obtains the
prefix of its tile by looking back over aggregates until it reaches a
completed prefix [[MerrillGarland2016]](../references.md#merrillgarland2016).
The claim order prevents a block from waiting for an unscheduled
predecessor. An initialization launch clears the state before the scan
launch, so an opt-in compiler strategy makes the three group scans use
two launches each. All offsets remain exact integers, and the permutation
of each bin is still sorted after scatter. In an RTX 4090 measurement the
scan kernels of a JAC group build fell from 52.65 to 38.41 µs, but the
ordinary Amber runs did not gain end to end and STMV slowed by 0.55%.
With an outer skin of 0.5 Å, JAC rebuilt every 2.2 steps and ran 1.02%
faster. The three-launch strategy remains the default.

**One set of buffers for the reciprocal sum** (D315). A
program evaluates the reciprocal sum in several places: before the first
step, and in each copy of the step that the loop nest specializes by what
the step must produce (the energy for the log, the virial for the
barostat, or neither), seven in a run at constant pressure. Each held its
grid, its transform, the bricks of the spreading, the places of the
particles, and the bins of the tiles for the whole run, about 179 MiB at
STMV. A sum writes each of these buffers before it reads it within one
evaluation, except the bricks, which it finds zero and leaves zero, so
sums that never run at once can take one buffer. On one stream they never
do: its work runs in the order in which it was issued. A sum on the second
stream (D87) declares that it reads and writes these buffers, so the
analysis of independence that places its join puts the join before the
next sum that shares them. The buffers are now allocated once for each
shape and taken by every sum of the program: the device peak of STMV
falls from 3.7 to 2.6 GB, the sevenfold replica takes 15.4 GB, and the
elevenfold (11.7 million atoms) runs in 22.8 GB, with the deterministic
mode unchanged to the bit and the rates of the suite unchanged.

**Only the scratch a spreading uses** (D320). With the
tile strategy (D282), particles are binned and spread directly through
tiles; the bricks of D108 are neither read nor written. The storage pass
therefore omits their allocation and initial zeroing. The gathering of
order 4 in f32 still takes the particle weights, chosen from the order,
grid type, and algebra rather than from the presence of bricks. A
non-tiled f32 sum retains bricks where its spreading consumes them, and
the fixed-order path retains its fixed-point grid and its addition order.

**A shared grid gather** (D325). The Gaussian and prolate PME
gathers read $\phi_{(aK_2+b)K_3+c}$ with
$c=(c_0+\ell)\bmod K_3$. One device emitter computes the wrapped z
coordinate and another reads the grid at the row base plus that
coordinate. The order-4 stored-weight gather keeps i32 address
arithmetic; the general Gaussian gather keeps index arithmetic; the
prolate gather retains its signed wrap test and safe read for inactive
points. The same emitters serve potential, listed-particle, and
triclinic instances. The thread layout and summation order stay as
before; the remaining spline or prolate window work belongs to each
instance. The scalar gather has no extra storage or control key.

**One storage for the neighbors of the start and of the steps**
(D330). The evaluation before the first step builds a
neighbor structure that the steps do not continue: they begin with a
structure of their own, built at step 1, so that a run builds at the same
steps whether or not it is cut into segments (D218). The two structures
have the same shape, and the first is no longer used when the second
begins, yet each kept its lists and buffers for the whole run: 466 MiB at
STMV. The storage pass now gives the steps the start's storage when
nothing uses it any more, and resets it, as every segment of a run with
checkpoints already does. A reset structure is valid for no
configuration, so the first refresh builds whatever the storage held, and
a build writes everything its loops read. The device peak of STMV falls
from 2.5 to 2.0 GB and that of the elevenfold replica from 22.3 to 17.2
GB. The fourteenfold replica (14.9 million atoms) runs in 23.0 GB on the
24 GB device. The deterministic mode is unchanged to the bit.

**Lists of groups sized by their first build** (D[density-lists]). The
pool of blocks of the lists of groups (Section 4.4) started with as many
entries a group as a row of the matrix, from the particle with the most
neighbors and half as many again: 1.84 times what STMV's lists took. A
structure now starts with one block a group, and its first build, which
counts the blocks it takes even where they do not fit, grows the pool to
a quarter more than that count and is made again. Section 4.4 derives why
the count of one build gives the need of every build of an equilibrated
state; a later build that needs more grows the pool the same way. The
device peak of STMV falls from 2.05 to 1.92 GB and that of the
fourteenfold replica from 23.0 to 21.1 GB, and the fifteenfold (16.0
million atoms) runs in 22.6 GB.

**Flags read without waiting for the stream** (D113). The test of validity
(Section 4.1) sets a flag on the device, and the host must read it to
decide whether to build. A copy to pageable memory waits for everything
issued before it, after which the device idles until the host has issued
the next kernels: on JAC the device was idle 45 µs of a step of 249, and
86 of 331 at constant pressure. The runtime now copies a flag into memory
of the host that it pins, records an event after the copy, and waits for
that event alone (`mdrtFlagStart`, `mdrtFlagFinish`); the lowering puts the
wait before the first op that uses the flag rather than after the loop
that sets it. The reciprocal sum of PME moves before the test of the
structure when its operands dominate that test and every intervening op
is independent of the sum by its memory effects and buffer aliases
(D179). In particular, a kernel that fills its
charge buffer must run first, even when the buffer itself was allocated
earlier; otherwise PME reads stale or uninitialized charges (issue #26).
When the move is safe, the device computes PME while the host waits. On JAC the rate went from 695 to 739 ns/day at constant
energy and from 522 to 600 at constant pressure, and the waits for the
stream fell to the builds alone. The flags are now memory of the host
that is mapped for the device (D118): a kernel that sets one stores into
it across the bus, the host waits for an event after the kernel and reads
it, and clears it as it reads it. The copies are gone from the stream,
where each held the device for some microseconds between two kernels, and
so is the copy of a zero that cleared a flag where it was set, which
waited for the whole stream, the reciprocal sum included, every time the
inner list was pruned. Since a flag is read after the kernels issued
before the read rather than at its place in the stream, each loop that
sets a flag has one of its own. On ubiquitin in OPC the device stood idle
49 µs of a step and now 29; the rates rose by 4% there, by 6 to 7% on JAC,
and by 1 to 3% on FactorIX and Cellulose. The argument (Section 12,
principle 2): the value the host reads is the value the flag had at its
place in the stream, given that only one loop sets the flag, that no
launch of that loop is issued between the event and the read, and that a
store of a kernel to mapped memory of the host is visible to the host once
an event recorded after the kernel has completed, and a store of the host
to a kernel launched after it. The last premise holds for pinned memory
mapped for the device on x86-64 with an RTX 3090 and driver 595.84, where
it was checked; on a platform whose mapped memory is not coherent, the
copy of D113, which the runtime keeps, is the fallback.

**A second stream** (D81, D87). The reciprocal sum may run on a second
stream beside the loops that follow it, with a join where its forces are
read. The pass `md-exec-assign-streams` marks the op, moves it past the
ops it is proven independent of, and places the join; the lowering checks
the window again. The argument for its correctness is that two ops that
are independent (neither reads what the other changes, by their declared
memory effects and an alias analysis of the buffers) give the same memory
in either order. It is off by default: once the kernels of PME became
cheap, sharing the device with the loop over pairs, the longest part of a
step, lengthened the step.

## 8.4 Where the time of a step goes

*Table 8.2. Cellulose NVE (408,609 atoms, 2 fs), mixed precision, dual
list 11/8.6 Å, RTX 3090 at 300 W; the device time of a step from nsys,
2000 steps.*

| Part | µs per step | Share | Launches per step |
|---|---|---|---|
| Loops over pairs (Lennard–Jones and the direct sum) | 1011 | 37.3% | 3.0 |
| PME (spread, FFT, convolution, gather) | 739 | 27.2% | 12.0 |
| Loops over tuples (bonded terms, excluded pairs) | 352 | 13.0% | 5.1 |
| Integration (kicks, drifts, constraints) | 265 | 9.8% | 4.0 |
| Prunings of the inner list | 175 | 6.4% | 1.4 |
| Builds of the outer list | 154 | 5.7% | 1.8 |
| Other loops over particles | 17 | 0.6% | 1.0 |
| Total | 2713 | | |

The wall time of the same steps was 2783 µs (62.1 ns/day): the device is
busy 97% of a step on a system of this size. On JAC, a step of about
240 µs, the share of the host is larger, which is what D113 addressed.

## 8.5 Privatized scatter-accumulation

Two kernels add values into a field at indices that their threads share:
the spreading of PME adds $q_j\prod_a M_n(\cdot)$ into the points of the
grid (Section 5.3), and the kernel of the tuples evaluated once adds the
forces of each tuple into its members (Section 8.1). Item $k$ adds
$v_k$ at index $i_k$, and the result is

$$
a_i' = a_i + \sum_{k : i_k = i} v_k .
$$

With one atomic addition of the device for each item, the atomics are the
cost: about 70% of the kernel of the tuples on Cellulose (D283). A
*privatized* scatter gives each block $\mathcal B_b$ of threads a copy, in
block memory, of the indices its items touch. The block adds its items
there, and after a barrier adds each touched index to the field once:

$$
a_i' = a_i + \sum_b \Big(\sum_{k \in \mathcal B_b : i_k = i} v_k\Big).
$$

This is the atomic specialization of CUB's `BlockHistogram`
[[CUB]](references.md#cub), with a value in place of a count; the cost of
such privatized sums in block memory is modeled in
[[GomezLuna2013]](references.md#gomezluna2013). The strategy is the
attribute `#md_exec.scatter` of the ops that scatter, which the pass
`md-exec-choose-scatter` sets and the lowering applies. Each condition
below is one function that both of them call (D285,
`docs/device-primitives.md`).

**The reordering.** The two forms are the same sum, grouped differently.
For integers, fixed point included, addition is associative and
commutative, modulo $2^{64}$ as well, so the grouping gives the same bits.
Floating-point addition $\oplus$ is commutative but not associative: a sum
of $m$ terms in any order and grouping differs from the exact sum by at
most $(m - 1)\,\mathsf u\sum_k\lvert v_k\rvert + O(\mathsf u^2)$
[[Higham2002]](references.md#higham2002) (Section 4.2), with
$\mathsf u$ the unit roundoff. The atomics already add in an order the
threads decide, so a privatized sum lies within the bound that any run
with atomics satisfies, and its results differ from them as two runs with
atomics differ from each other: $10^{-7}$ to $10^{-6}$ of the RMS force in
mixed precision, $10^{-16}$ to $10^{-15}$ in double (D282, D283). The
default mode therefore allows a privatized floating-point sum, and the
deterministic mode does not: it fixes the bits, which a regrouping in f32
or f64 changes. Its sums are in fixed point (D70, D84), where a privatized
sum would give the same bits; that is the next step of #416. No strategy
of floating-point values keeps the deterministic mode bitwise.

**The algebra as a stated fact (D301).** Which of these
forms a sum may take is a property of the sum, and the IR states it for
each value that an op reduces, rather than leaving it to the mode:
`any_order_exact` (integers and the logical or: any order and grouping gives the
same bits), `any_order` (the terms may be taken in another order and grouping,
and the result moves within the bound above; the default mode), or
`fixed_order` (the result may not depend on the threads: a fixed order, or a
sum made exact in fixed point; the deterministic mode). The chooser of
strategies and the lowering read the stated algebra, through one set of
functions, and refuse a sum that states none. A block reduction whose
order depends on the count alone, as every global sum on a device is
(D150), meets all three, so stating the algebra changed no generated code;
what it changes is where a lowering may reorder. The licence is about
order only: `any_order` does not permit another precision of the
accumulator or another growth of its rounding error, such as a tree
replaced by a serial chain of atomics: the worst-case bound of a sum of
pairwise sums has the factor $\lceil\log_2 m\rceil\,\mathsf u$ where a
serial chain has $(m - 1)\,\mathsf u$
[[Higham2002]](references.md#higham2002) (Section 4.2). The sums across domains of a
distributed run are `fixed_order` for a given decomposition: the same bits
on every run with the same decomposition and build, not with another
(#409).

**The tile: a bounded footprint.** In the spreading, item $j$ with first
point $s_{aj}$ adds only to the points $s_{aj} + t_a$, $0 \le t_a < n$,
since $M_n$ vanishes outside $(0, n)$. Binned by $\lfloor s_{aj}/g\rfloor$,
the items of the bin with origin $g\mathbf b$ add only to the points
$g\mathbf b + \mathbf t$, $0 \le t_a < g + n - 1$: the footprint of a block
is bounded by the IR, by the order and the bin, and a dense tile of
$(g + n - 1)^3$ values holds it. The tile maps to the grid by
$(g b_a + t_a) \bmod K_a$, which the kernel takes as one subtraction of
$K_a$ where $g b_a + t_a \ge K_a$. That is exact where $g b_a + t_a <
2K_a$, which $g b_a < K_a$ and $g + n - 1 \le K_a$ ensure, and it maps the
points of a tile to distinct points of the grid. Hence the condition: every
edge of the grid has at least $g + n - 1$ points. The tile and the items of
a round must also fit the 48 KiB of block memory that a kernel takes
without asking for more, or the launch fails.

**The hash: a fallback.** The members of the tuples of a block are not
bounded by the IR. The table keys a slot by the index, found by a
multiplicative hash and the next slots in turn
[[Knuth1998]](references.md#knuth1998), and claimed by a
compare-and-swap, so that a slot holds one index. A search gives up after
a bounded number of slots, and the item then adds its value to the field
with an atomic of its own. Each value is therefore added once, either to
the slot of its index, which the flush adds once after the barrier, or to
the field directly. The sum is the same as above with some items in a
group of their own. Without the fallback a bounded search would lose
values, and an unbounded one has no bound on its time: one block of JAC
with 494 particles in 512 slots took the kernel from 12 to 42 µs. With a
bound of 8 slots it took 13.2 µs.

**The choice.** The conditions above make a strategy correct. Cost
decides between strategies that are correct: a grid of at least 500,000
points in f32 for the tiles, and at least 100,000 tuples for the table,
counted on the host. Below these thresholds the blocks are too few to
occupy the device, and the privatized form is slower (Section 5.3, D283).
Moving both kernels onto the attribute and the pass left the generated
code as it was.

**When a table pays.** Let a block add $m$ values to $d \le m$ distinct
indices. With atomics it issues $m$ additions to device memory. With the
table it issues $m$ lookups and additions in block memory and $d$ device
atomics, plus one for each value whose search fails. The table therefore
saves device atomics in proportion to the reuse $m/d$, and costs a lookup
for every value. A lookup is a compare-and-swap whose result the thread
waits for, while a device `red` is issued without waiting. With $m/d$
near 1 the table can only lose, and the measurements below show that a
large $m/d$ is not enough. We attribute the difference to contention: the
table paid where many threads of a block add to one address at once, and
not where their atomics spread over addresses. No counters were available
to measure the contention itself. The tuples of a block share their members,
about four values to an index on Cellulose (D283), and gain. A loop over
groups gives the warps of a block consecutive blocks of one group's list
(Section 4), which name each particle once, so its $m/d$ is near 1. The
table made those loops slower on every system measured, 1.18 to 1.32
times the step of the Amber suite and 1.49 times an evaluation of an EAM
potential in mixed precision. A loop over the rows of a matrix that adds
at both ends of each pair (the domains of D290) has rows in the order of the cells,
which share their entries, so its $m/d$ is large. Yet the table took 2.06
times its time in mixed precision. Each row adds to an entry once, so the
atomics of a warp go to different addresses. The table is off at both
(D291).

**One op, one emitter.** The table is one emitter that every site calls.
The tuples' table moved onto it with byte-identical PTX. A value that no
loop computes, such as the contributions of the ghost rows of a domain to
their owners ($G^\mathsf T$, Section 2), is added by
`md_exec.scatter_add`, which computes $a_i' = a_i + \sum_{k : i_k = i}
v_k$ with the strategies and the conditions above. A loop that computes
its values in registers keeps them there and states the strategy of its
own destinations, since a separate op would store every value and read it
again.
