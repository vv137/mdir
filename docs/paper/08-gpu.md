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
| Over tuples that may share particles (bonds, angles, …) | A thread a particle, the tuples $p, p+n, \dots$ evaluated once each (D103) | Atomic additions to the members; sums of its own tuples |
| Over disjoint tuples (constraints) | A thread a particle; the member at place 0 evaluates the tuple (D83) | Every member of its tuple, which no other thread writes |

**Runs.** Consecutive loops that take the same positions and do not read
what another writes become one kernel: the loops over pairs and over
tuples of a term of the potential (rows of the same particle), the loops
over the disjoint tuple sets of a union of constraints (D83), and the
integration runs of Section 8.2. A run is lowered where its last loop is.
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
groups, some 170 tuples on JAC, are one block. A kernel passes each
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

**Divisions and reciprocals** (D98, D[approximate-quotient-rounding]). A division marked approximate becomes
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
$\operatorname{select}_k(\text{in}_k, \text{lane}(\text{member}_q^{(k)}))$
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
buffers that a step allocates and frees cost a lookup.

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
