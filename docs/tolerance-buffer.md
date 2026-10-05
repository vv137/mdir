# Tolerance-based neighbor buffers

Status: proposed, awaiting the maintainer's control-file decision for #59.
Decision label: D[tolerance-buffer]. No runtime behavior is implemented yet.

## Contract proposed for review

The displacement-checked policy remains the default. A positive
`[energy].neighbor_drift_tolerance` would opt into statistical buffers,
with units of kcal/mol/ps per particle. Absence keeps the existing policy;
zero, negative, and nonfinite values would be errors. This tolerance
controls the estimated error from missing pairs, not the total integration
error or a guarantee on an individual trajectory.

The proposal uses the existing `rebuild_interval` as an explicit positive
outer-list lifetime, and adds `prune_interval` as an explicit positive
maximum age of an inner-list partition. Both are in integration steps.
Computed reaches replace manual distances only in this mode: specifying
`pairlist_distance` or `pruned_distance` with the tolerance would be an
error. There are no new files or overwrite rules.

The proposed first implementation covers built-in Lennard-Jones and
real-space electrostatic pair terms at constant volume, on CPU and GPU in
mixed and double precision. Unsupported custom pair terms, implicit
solvent, changing lambda, and pressure coupling would produce errors.
The maintainer must decide whether this initial scope is sufficient or
pressure coupling is required in the first implementation.

Temperature needs an explicit contract, particularly for NVE runs. The
proposal adds `neighbor_temperature` in kelvin as the temperature assumed
by the displacement model. It is required in the opt-in mode rather than
silently inferred from a thermostat or one velocity sample.

## Estimator and scheduling

The method follows the statistical displacement model discussed in
[[Pall2013]](references.md#pall2013) and the dual-list scheduling of
[[Pall2020]](references.md#pall2020). The
[GROMACS reference manual](https://manual.gromacs.org/2024.6/reference-manual/algorithms/molecular-dynamics.html)
describes the Gaussian free-particle displacement approximation. Its
cluster-specific reduction factors must not be transferred to MDIR's
lists without measurements.

Use mass-dependent relative displacement distributions and the actual
cutoff behavior of each potential. Sum absolute error estimates over
pair types, without cancellation between types. Split the total error
budget between the outer and inner lists; the sum must not exceed the
requested tolerance. Solve for the two buffers independently against
their maximum lifetimes. Validate the integral by independent numerical
quadrature before using it to choose a reach. A discontinuous potential
must include its cutoff jump, not only derivatives at the cutoff.

Rolling pruning needs a separate partition cursor and age bound. Rebuild
the outer list, prune every partition immediately, then rotate through
partitions with a schedule that never exceeds the modeled inner lifetime.
A partition must always be pruned from the outer list, since pruning an
already pruned list cannot restore returning pairs.

Current implementation obstacles:

- `Control.cpp` rejects combining `pruned_distance` and `rebuild_interval`.
- `ReuseNeighbors.cpp` attaches inner skins only to displacement checks.
- GPU interval refresh returns before the dual-list path; merely relaxing
  the parser would not implement fixed-lifetime dual lists.
- Existing pruning refreshes the entire inner list. Rolling partitions
  require runtime and IR changes, not only selecting smaller skins.

Restart must reset list storage and partition ages consistently. Any new
keys affecting forces belong in the checkpoint fingerprint.

## Reporting and acceptance

Print the selected reaches, temperature assumption, lifetimes, partition
count, and estimated error in physical units. Report realized conserved
energy drift over the sampled time range, together with sample count and
uncertainty; insufficient samples must be reported as unavailable. In NVT
use the conserved quantity including thermostat work. Total realized
drift contains integration error and cannot alone isolate missing-pair
error; compare with the displacement-checked baseline.

Before marking the implementation ready:

1. Compare estimator integrals with independent quadrature and test
   monotonicity in buffer, lifetime, mass, temperature, and tolerance.
2. Compare forces and energies at saved configurations with an independent
   complete-pair oracle on CPU/GPU, mixed/double. Include returning pairs,
   exclusions, periodic boundaries, partition wraparound, and restart.
3. Run the local suite, short Amber suite, and device sanitizer under the
   prescribed GPU locks; report repeat counts for intermittent failures.
4. Time the Amber suite against main on GPU 0, with matched ensemble,
   precision, input, and integration step. Report ms/step, speedup,
   estimated error, realized drift, and baseline drift at each tolerance.
   Claim acceptance only after a measurable speedup is demonstrated.

No CPU/GPU validation or performance result is available at the design
stage. Update the control reference, roadmap, decisions, changelog, and
white paper alongside the implementation after the design ruling.

## GROMACS 2026.3 source analysis

The following observations come from reading the released source, not
copying its implementation. Source paths below are relative to the
[GROMACS v2026.3 source tree](https://gitlab.com/gromacs/gromacs/-/tree/v2026.3/src/gromacs).
They refine the proposal; they do not settle MDIR's user-visible contract.

### Buffer estimator

`mdlib/calc_verletbuf.cpp`, especially `energyDriftAtomPair`,
`energyDrift`, and `calcVerletBufferSize`, integrates a cutoff Taylor
expansion against Gaussian displacement tails. It includes the potential
value and derivatives through third order, with separate LJ and Coulomb
cutoffs. Inertial displacement variance scales as temperature times the
square of lifetime divided by mass. Constraint-aware displacement combines
center-of-mass translation with bounded rotational motion; Brownian
displacement has a separate model. Treating all constrained hydrogens as
free particles would sacrifice much of the potential buffer reduction.

The total estimate sums the absolute contributions of atom-type pairs,
after combining LJ and Coulomb within each pair type. This allows
cancellation within a pair type but not between types. Our proposed
uncancelled estimate can be more conservative; that difference must be
reported when comparing computed reaches.

`computeEffectiveAtomDensity` bins coordinates into cells about a cutoff
wide. It uses the sum of squared occupancies divided by particle count
and cell volume. Using only particle count divided by total volume can
underestimate the density seen by particles in heterogeneous systems.

Buffer selection uses bisection on a 0.001 nm grid. The estimate is
evaluated at the last-use age, then divided by rebuild period, time step,
and particle count to give kJ/mol/ps per particle. Cluster surface factors
reduce estimated missing pairs. MDIR must initially use an atom-pair
factor of one unless its own grouping benefit is independently validated.
The same search can enforce a separate pressure-error bound.

### Lifetimes and rolling pruning

`nbnxm/pairlist_tuning.cpp` distinguishes outer rebuild period from outer
last-use age: with one force evaluation per step, a period of N has age
N minus one. CPU inner-list age similarly excludes the current force
step. GPU pruning prepares a list for the next step, so its age includes
that additional step. MDIR must derive ages from its own launch order.

GPU rolling pruning launches every two steps and divides the inner
refresh period by two to choose the partition count. The source tunes the
outer period from candidates 20, 25, 40, 50, 80, and 100 using list-size
heuristics. Automatic period tuning is disabled for unthermostatted MD;
that restriction is separate from estimating a buffer with a supplied
temperature.

`nbnxm/cuda/nbnxm_cuda_kernel_pruneonly.cuh` retains an outer mask and a
working inner mask. Fresh-list pruning initializes both. Subsequent rolling
visits check only outer-mask entries missing from the inner mask and add
entries that have approached the inner reach. They do not remove entries
already active; those are cleared at the next fresh-list prune. Partitions
are interleaved supercluster entries, with a rolling cursor per block.
Thus the inner list grows between outer rebuilds. This is more specific
than periodically recomputing a compact inner list and should be evaluated
as an MDIR implementation option.

### Error budgets and NPT acceptance

GROMACS requires outer and inner energy estimates independently to meet
the same tolerance. Its source explicitly acknowledges a small possible
underestimate, while adding the two estimates would double count some
errors. The proposed MDIR split budget is a conservative departure and
needs a speed/accuracy comparison before choosing it.

For pressure error, GROMACS subtracts the estimated outer contribution
from the inner allowance and reports the sum of both contributions.
Energy drift alone therefore does not settle NPT accuracy. To satisfy
the full Amber acceptance in #59, the implementation plan must address
cell deformation and pressure bias, rather than declare completion with
constant-volume validation alone. Whether to expose a separate pressure
tolerance remains a maintainer decision.
