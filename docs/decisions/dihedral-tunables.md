# D[dihedral-tunables]

Recorded at its merge.

**The force constants of built-in proper periodic dihedrals are tunable runtime values, with explicit topology-row maps, updates without recompilation, and derivatives through the existing native tuple IR.**

Part of #730, following ruling 18. Impropers, angles, bonds and CMAP remain
separate design and implementation slices.

## Sites and values

`mdir.Tunable(name, "dihedral_k", map=None, values=None)` has no `term` or
mixing option. Sites are the full rows of `Topology.dihedrals`, in reader
order, one site for each Fourier contribution, including periodic improper
rows. An explicit map has that full row count; improper rows must map to
`-1`, meaning fixed. Without a map, proper rows receive dense independent
entries in reader order and improper rows remain fixed. No proper row, no
selected site, a selected improper row, a sparse entry range, or an invalid
shape is an input error. There is one declaration for this physical parameter.

Mapped entries are dense `0..M-1`, each used by a site. `values` has shape
`(M,)`; without it all sites tied to one entry must have the same initial
coefficient. Repeated quadruples and repeated Fourier contributions remain
separate rows. Tying coefficients, including across different periodicities
or phases, is explicit. No class is inferred from Lennard-Jones types.

The read-only copied topology arrays `dihedral_force_constants` (kJ/mol),
`dihedral_periodicities` (int64) and `dihedral_phases` (radians) align with
`dihedrals` and `improper_dihedrals`. They describe the model snapshot, not
the current values in `Simulation.tunables`.

The coefficient is exactly $k_r$ of
$U_r=k_r[1+\cos(n_r\phi_r-\delta_r)]$, in kJ/mol. Finite signed values,
including zero, are accepted; NaN and infinity are rejected. No factor or
phase shift is added. Periodicity, phase, row order, exclusions and 1–4
parameters remain structural and unchanged by an update.

## Updates and derivatives

Declarations and maps participate in the existing structural identity and
version checks. Atomic value updates retain D213/D254/D256's version,
borrow, retained-array and continuation contracts. Coefficient arrays are
filled by the same path as compilation; no lowering or JIT occurs on update.
Rows and derivative paths survive coefficients crossing zero.

For map $m(r)$, the derivative in entry $a$ is
$\partial U/\partial\theta_a=\sum_{r:m(r)=a}[1+\cos(n_r\phi_r-\delta_r)]$.
It is dimensionless in public units and uses D230/D240's existing gradient
interfaces and typed differentiation. Tied rows contribute once each. A
missing rule is an error; a zero at one frame is not a structural-zero proof.
The dihedral adapter retains its tuple-field and scalar operation order.

## Validation gate

The implementation PR records independent NumPy energies, forces and analytic
coefficient derivatives, finite-difference step sweeps, and CPU/GPU results
in double and mixed precision. It checks maps and errors, immutable metadata,
atomic updates across zero, unchanged 1–4 data, coexistence with existing
tunables, update-versus-fresh-compile agreement and no additional compilation.

At the final clean head, the byte-identical printed/lowered corpus, order
matrix and five bitwise checkpoints cover controls **without the new
declaration**. Both D309 shards check the exact expected unsupported list.
Untuned performance is compared with archived main on the same device model.
Tuned controls use independent physics and deterministic update checks;
the old API does not support them. Numerical or intermittent failures stop
the work and are never retried.
