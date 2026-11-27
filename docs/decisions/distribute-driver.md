# D[distribute-driver]

Recorded at its merge.

**The CPU control-file driver runs fixed-cell NVE velocity Verlet with
finite-range pair terms over the domains of `[distribute]`, and in the
deterministic mode reproduces the same bits on every grid.** The CPU control-file driver lowers the fixed-cell velocity Verlet program
onto domains. The potential rewrite supplies the force and energy kernels.
The step state carries positions, velocities, forces, masses and the real
per-particle parameters used by the force. The driver retains its output
schedule. At a checkpoint segment that reorders particles, the domain loop
starts after that permutation. Collected rows use the same particle-set
identity as the other inputs to the output calls.

The keys and default choices follow Section 13 of
[ML2](../distributed-ml2.md). The acceptance comparisons are CPU NVE with
finite-range LJ and a truncated Coulomb/LJ mixture, with both mixed and
double output storage. The reference executor computes its domain kernels
in double. In deterministic mode the same build must reproduce the same physical
trajectory across decompositions. Round each force component to Q40 and each
energy or virial component to Q32 before its first addition. Encode each term in a signed 64-bit word. Accumulate in checked
signed 128-bit words, including reverse halo sums, and range-check the
completed quantity before narrowing. At most $2^{32}$ particles ensures
that the number of directed pair visits is below $2^{64}$.

Log every candidate grid score, selected geometry, import size and source
of the traversal and transport. Sample occupancy, migrations, directed halo
bytes and phase times at the energy cadence. Record optional rank traces
under the prefix in `MDIR_DIST_STEP_TRACE`. Checkpoints keep their physical
format and atomic commit. Domain and ghost records belong to the manifest.
Initial draws are indexed by particle ID at step zero. Deterministic drifts wrap on every step. Reconstruct the halo from
checkpoint rows and evaluate each displacement from untranslated coordinates
with an explicit image shift. The ordinary CPU path retains its numerical
contract. An explicit one-domain grid selects the distributed contract.

Disable wall-clock dataset timestamps for deterministic distributed HDF5
output. Preserve checkpoint part history and integrity checks on continuation.

The CPU pair executor bins the current owned and imaged ghost positions
at each traversal. A center visits the neighboring cells, with the same
identity predicate, strict cutoff and fixed-point arithmetic as the full
row scan. The cell width includes a rounding margin for the separate
image subtraction in deterministic mode.

## Validation against the ordinary path

`test/Driver/distribute-oracle.test` takes one step from the checkpoint of
an ordinary liquid run of 100 steps (forces of a liquid, not of the input
lattice, where they cancel) on 2 x 2 x 2 domains, in mixed and double,
deterministic or not, directed and unique: 16 distributed runs against 8
ordinary ones on argon and on `charged.toml`. It compares the potential
energy and the virial of the energy file, the H5MD potential energy, and
every force component of the checkpoint. The bound of a quantity
$Q=\sum_t q_t$ is $2[(n+32)\,e\,S + e\,X\,D]$ plus the fixed-point
quantization (Q40: $n_i\,2^{-41}$ kJ/mol/nm on particle $i$; Q32:
$2^{-33}$ kJ/mol per visit), the pairs within the coordinate error of the
cutoff, and the print quantum (5e-7 kcal/mol). Here $e$ is the unit
roundoff of the ordinary kernels ($2^{-24}$ mixed, $2^{-53}$ double), $n$
the number of terms, $S=\sum_t|q_t|$,
$D=2\sqrt3\sum_t|\partial q_t/\partial\mathbf d|$ the response to the rounding of a coordinate difference, and
$X$ the largest coordinate; NumPy computes $S$ and $D$ from the positions.
The script derives each term. The largest ratios of difference to bound:

| System, mode | Energy file | H5MD potential (kJ/mol) | Forces (kJ/mol/nm) |
|---|---|---|---|
| argon, double | 0 (bound 1.0e-6 kcal/mol) | 5.5e-12 (bound 2.4e-8) | 6.2e-13 (bound 1.6e-11; ratio 0.039) |
| argon, double, deterministic | 0 (bound 1.6e-6) | 3.1e-10 (bound 2.7e-6) | 5.4e-12 (bound 3.0e-11; ratio 0.18) |
| argon, mixed | 4.0e-6 potential, 5.7e-4 virial (bounds 3.1, 20) | 2.0e-5 (bound 12.9) | 8.4e-4 (bound 1.1e-2; ratio 0.075) |
| charged, double | 0 (bound 1.0e-6) | 4.5e-13 (bound 1.9e-7) | 1.5e-12 (bound 6.0e-11; ratio 0.025) |
| charged, double, deterministic | 0 (bound 1.3e-6) | 5.3e-9 (bound 1.1e-6; ratio 0.005) | 9.6e-12 (bound 5.5e-11; ratio 0.18) |
| charged, mixed | 6.0e-6 potential, 8.3e-4 virial (bounds 24, 26) | 2.3e-5 (bound 101) | 1.4e-3 (bound 2.6e-2; ratio 0.052) |

The reference values in double: argon $U=-1087.664634$ kcal/mol and virial
$-294.468869$ kcal/mol; the mixture $-334.238738$ and $-90.353355$. The
energy bounds of the mixed mode take the worst case of a sum of all pairs
in binary32, and catch only errors of order one there; the double mode
checks the same energies to the print quantum. Directed and unique
traversals give bitwise equal results in the deterministic mode: the two
directed visits of a pair add two equal Q32 words that one more fractional
bit halves exactly, and a unique pair adds $\mathrm{round}(2^{40}f)$ at one
end and $\mathrm{round}(-2^{40}f)=-\mathrm{round}(2^{40}f)$ at the other.

`charged.toml` as written is refused: its `[[energy.pair_override]]` gives
the pair a table of types, whose integer field the reference executor does
not distribute ("the reference executor runs f64 only"). The oracle and the
restart test take it without the override, with the combination rule.

The test fails on realistic defects, each tried in a scratch build of this
branch and reverted:

| Defect | Where | Fails |
|---|---|---|
| The extra fractional bit of a directed sum dropped | lowerAllReduce | energy, virial, H5MD of deterministic directed runs (both systems, both precisions; ratio up to 8.5e8) |
| The weight 0.5 of a directed sum made 1 | lowerSumPairs | energy, virial, H5MD of non-deterministic directed runs (ratio up to 1.9e11) |
| The sign of the image shift $s$ | mdrtDistImageShifts | all quantities of every deterministic run (32 checks of 32) |
| The decoding scale of `finish` doubled | Lowering::finish | energies of every deterministic run; forces of directed runs, and of unique runs with two terms |
| The Q40 decode of a directed gathered force | lowerGatherPairs | forces of deterministic directed argon (2592 of 2592 components); none on the mixture, whose two terms add their words before one decode, so that this decode is not on its path |

## Refusals

`test/Driver/distribute-refusals.test` gives each refusal a control file
that differs from an accepted one in its condition alone: no skin, single
precision, a device target, PME and LJPME, an open or barostatted cell, a
thermostat, leapfrog, a minimization, the forms of `domains` (0, negative,
a float, a float division), constraints and a triclinic cell (the builder),
the exclusions of bonds (a diagnostic of the passes, recorded in the log and
the manifest), and a step trace named as the checkpoint. Only the refusals
of `[distribute]`, of the builder, and of the passes of `md_dist` are
recorded as a refusal of the distributed setup; another error of a control
file with the table is reported as without it.
`test/Runtime/dist-wide-words.mlir` calls the runtime checks of the word
count ($2^{32}$ particles accepted, $2^{32}+1$ refused), of a wide addition
reaching $2^{127}$, of a Q32 term of $2^{31}$, and of the narrowing of a
completed sum.

## Formats

The format of a fixed-point word is fitted to its quantity, which the
builder names on each collected sum (`md_dist.quantity`): Q32 for pair
energies and virials, kinetic energies, and momenta; Q40 for forces and
gathered fields; Q16 for $\sum_iF_i^2/m_i$ of the kinetic correction. Q32
held it below $2^{31}$: 864 atoms of argon with 250 times the example's
$\epsilon$ give $7.6\times10^9$ (kJ/mol/nm)$^2$/amu. A force inside the Q40
range (below $2^{23}$) over a mass of at least 1 amu has a square below
$2^{46}$; Q16 gives the sum a
range of $2^{47}$, and its quantum, scaled by $\Delta t^2/8$, stays below
$10^{-10}$ kJ/mol per particle at $\Delta t\le 0.01$ ps. A range error
names the quantity, and the total energy, whose Q33 sum adds the words of
the potential and the kinetic energy, stops at an operand outside Q33
instead of adding zero (`test/Driver/distribute-word-formats.test`). The
conversion of a completed word $k$ to binary64 is exact for
$|k|\le2^{53}$: below $2^{13}$ kJ/mol/nm for a force, $2^{21}$ kJ/mol for
a Q32 energy, $2^{20}$ for the Q33 total, $2^{37}$ for a Q16 sum. Above, it
rounds to nearest; it is still a function of $k$ alone, so that the result
is the same on every grid. The executor through `md_exec` refuses the
fixed-point mode, which is the reference executor's.

## Checkpoint fingerprint

The execution group of the fingerprint records `[distribute] traversal`
and `[distribute] transport` as resolved, and outside the deterministic
mode `[distribute] domains`, the grid resolved for the cell. The
deterministic contract makes the trajectory, and the checkpoint file, the
same on every grid (`distributed-restart.test` compares the bytes of
checkpoints written on 1, 2, 8, and 16 domains), so its grid is left out
and a continuation on another grid is accepted without a note. Under that
contract, `--continue` of a deterministic run refuses a control file that
adds or drops `[distribute]`: the sums of the other driver are of another
arithmetic, and the continuation could not give the bits of the
uninterrupted run. Outside the deterministic mode every change of
`[distribute]` is a note, as for `[execution] threads`.
`test/Driver/distribute-continue.test` checks the records, the refused
continuations in both directions, and the accepted ones.

## Rebuilds in the deterministic mode

Each deterministic drift wraps the positions into the cell, and the halo
keeps explicit images, so that a wrap breaks coverage and forces a
migration and a new halo. On the argon liquid after 2000 steps of
equilibration (864 atoms, 99 K, 1 Angstrom skin, 2 x 2 x 2 domains), 1000
steps rebuilt 326 times in the deterministic mode and 55 times otherwise;
the trajectory had 391 wraps, in 326 steps. The expected number of wraps per
step is $\lambda=N\sum_a\langle|v_a|\rangle\Delta t/L_a$ with
$\langle|v_a|\rangle=\sqrt{2k_BT/(\pi m)}$: 0.42 here, and a fraction
$1-e^{-\lambda}=0.34$ of the steps, against 0.39 and 0.33 measured. At fixed
density $\lambda$ grows as $N^{2/3}$, so that above about $2\times10^4$
atoms nearly every deterministic step rebuilds. Keeping the halo across a
wrap needs an image count per row in every owned-owned displacement, in the
stored ghost shifts, and in the pair cells; patching the coverage test alone
would lose pairs. It is left open.

