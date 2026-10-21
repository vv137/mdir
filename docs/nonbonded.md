# Nonbonded Terms

Status: kept current; last checked against the code on 2026-10-21. The
decisions are D57 and those that amended it
([decisions.md](decisions.md)); particle mesh Ewald is in
[pme.md](pme.md), and the exclusions and the pairs three bonds apart in
[tuples.md](tuples.md), Section 5.

This document describes the parameters of pairs of types, the correction
for the dispersion beyond the cutoff, and the electrostatics.

## 1. Parameters of pairs of types

### 1.1 Tables

Amber gives Lennard-Jones as coefficients for each pair of types, and a
force field may set the parameters of some pairs apart from its mixing
rule (NBFIX). GROMACS does the same with `[ nonbond_params ]`. The
front end therefore always builds a table for each pair of types: from
the mixing rule, with the pairs that the topology sets overriding it. The
kernel makes one lookup. No mixing rule is left at run time.

```mlir
%sigma = mdrt.from_buffer %sigma_buffer
           : memref<?x?xf64> to !md.table<2, f64, symmetric>

%u = md.sum_relation %n, %x, %cell gather(%type : !kinds) ... {
^bb0(%r: f64, %d: vector<3xf64>, %t_i: i32, %t_j: i32):
  %s = md.lookup %sigma[%t_i, %t_j] : !md.table<2, f64, symmetric>, i32, i32 -> f64
  ...
}
```

| Item | Proposal |
|---|---|
| The type | `!md.table<rank, E>`, with `symmetric` for a table of rank 2 that is; E is f64 or f32, or a vector of them for the tables that a kernel reads at the same entry (D96) |
| Exchange | A lookup in a symmetric table with the two types of a pair is symmetric |
| Differentiation | A lookup does not depend on the positions |
| Storage | A buffer, on the device where the loops are |
| Mixing rules that fill a table | Lorentz–Berthelot [[Lorentz1881]](references.md#lorentz1881), [[Berthelot1898]](references.md#berthelot1898), geometric, and none: the table as the topology gives it |
| In a control file | The topology gives the types and the table; a control file without a topology keeps the mixing rules of M0 |

Charges need the product of the values of the two particles, which the
mixing rules of M0 do not have. The proposal is the rule `product`.

### 1.2 The correction for the dispersion

Beyond the cutoff the attraction of Lennard-Jones is left out. The
correction [[AllenTildesley2017]](references.md#allentildesley2017), [[Shirts2007]](references.md#shirts2007) adds its mean, for a uniform density beyond the cutoff, to
the energy and the pressure:

$$E_\text{disp} = -\frac{2\pi N^2}{3V}\, \frac{\langle C_6\rangle}{r_c^3}, \qquad P_\text{disp} = \frac{2 E_\text{disp}}{V},$$

with $\langle C_6\rangle$ the mean of $C_6 = 4\varepsilon\sigma^6$ over the pairs of particles,
which the front end computes from the table and the counts of the types.
It depends on the volume only, so it is a number on the host that
changes with the cell. It is on by default when a topology is read, as in
Amber; a control file asks for it with `dispersion_correction` in a pair term.

As the engines do [[GromacsManual2025]](references.md#gromacsmanual2025),
the correction takes the part of the term that decays as $r^{-6}$ and leaves
the repulsion out. As GROMACS does, it takes $N^2 \langle C_6\rangle$, with $\langle C_6\rangle$ the
mean of $C_6$ over the pairs of distinct particles that are not excluded;
it agrees with GROMACS to 10⁻⁷. Its virial is six times its energy, so the pressure
changes by $2 E_\text{disp} / V$. A switch or a shift of the potential inside the
cutoff needs the integral of the change as well; M1 took a plain cutoff
only. Since then the correction also takes `POTENTIAL_SHIFT`, adding the
estimate of what the shift takes within the cutoff (D210), and covers the
tails of the `[[energy.pair]]` terms of a topology (D209); a switch still
refuses it.

sander leaves the repulsion out as well, but sums $N_a N_b C_{6,ab}$ over
all pairs of types, with no mean over the pairs that are not excluded
(`vdw_correction` of `ew_setup.F90` in AmberTools). The two differ by
some $N_\text{excl} / N^2$ of the correction, $4 \times 10^{-4}$ of it for the dipeptide in
water; a comparison with sander states the difference.

Not implemented, and kept here as an alternative: the correction from the
whole expression of the term, integrated numerically beyond the cutoff,

$$
\begin{aligned}
E &= \frac{2\pi}{V} \sum_{ab} N_a N_b \int_{r_c}^\infty r^2\, u_{ab}(r)\, dr, \\
W &= \frac{2\pi}{V} \sum_{ab} N_a N_b \int_{r_c}^\infty r^2 \big(-r\, u_{ab}'(r)\big)\, dr,
\end{aligned}
$$

over $x = r_c / r$ in $(0, 1]$, which takes the repulsion and any term
that decays faster than $r^{-3}$. It differs from the correction above by
$(\sigma / r_c)^6 / 3$ of it for Lennard-Jones, $10^{-3}$ at the cutoffs of M1, less
than the error of the uniform density that both assume, and it would not
agree with either engine. Lennard-Jones by particle mesh Ewald, for systems whose density
beyond the cutoff is not uniform, such as membranes, is built since:
`lennard_jones = "PME"` (D162).

## 2. Electrostatics

### 2.1 A cutoff

For the intermediate stage of M1 (D54). sander of AmberTools computes a
plain `q_i q_j / r`, cut at the cutoff with no shift, in a periodic system
with `eedmeth = 4` of `&ewald`; it then leaves out the self term, the
reciprocal sum, and the correction of the excluded pairs. The stage takes
that form, and the comparison with sander is of the energy of each term.
The comparison with GROMACS takes another form instead: the reaction
field of both programs at a permittivity of 78.5, whose `Coulomb`,
`Coulomb excluded`, and `Coulomb self` MDIR compares with GROMACS's
Coulomb (SR) (D140; [validation.md](validation.md), Section 2;
`scripts/validation/gromacs`).

A plain cutoff of the Coulomb term does not conserve the energy: a pair
of charged particles that crosses the cutoff changes the energy by
`f q_i q_j / r_c`, some 10 kcal/mol for two waters at 9 Å. sander shows
the same (155 kcal/mol in 10 steps of 0.5 fs for the dipeptide in water).
The intermediate stage therefore compares the terms at a configuration;
a test of the conservation of the energy needs a Coulomb term that goes
to zero smoothly, or particle mesh Ewald.

### 2.2 Particle mesh Ewald

Decided: particle mesh Ewald on one device or one process, in M1 (D55).
Designed and built in its own document, [pme.md](pme.md) (D69 to
D71), which is the current description.
