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
