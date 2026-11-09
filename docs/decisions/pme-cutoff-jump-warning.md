# D[pme-cutoff-jump-warning]

Recorded at its merge.

**`mdir check` warns (`pme_cutoff_jump`) when the largest jump of the force of a pair of charges at the cutoff of PME with the Gaussian kernel, $f\lvert q_iq_j\rvert[\mathrm{erfc}(\beta r_c)/r_c^2 + 2\beta e^{-\beta^2r_c^2}/(\sqrt\pi\,r_c)]$ over the pairs that the topology does not exclude, exceeds 1 kJ/mol/nm; it gives the jump and the expected number of pairs a frame within the shell where mixed and double precision decide the cutoff differently, and `mdir run` is unchanged.** The maintainer's ruling on #793.

**The case.** #793 found the forces of mixed precision 17 to 34 kJ/mol/nm
(projected) from those of double on TIP3P boxes at a cutoff of 4.5 Å with
the $\beta$ of a 9 Å cutoff. The cause is not a neighbor structure: on the
same frame (12,600 particles, RTX 4090 and CPU, zero-step forces, $\beta$
and the grid fixed) the matrix and the groups give the same error.

| $r_c$ (Å) | $\beta$ of (Å) | Max $\lvert dF\rvert$, matrix / groups (kJ/mol/nm) |
|---|---|---|
| 9, 7, 6 | 9 | 0.43 / 0.43 |
| 4.5 | 9 | 41.8 / 41.8 |
| 4.5 | 7 | 10.4 / 10.4 |
| 4.5 | 6 | 2.86 / 2.86 |
| 4.5 | 4.5 | 0.43 / 0.43 |

These maxima are not projected; the baseline of 0.43 lies mostly along the
constraints. An O–H pair of the frame lies $5.8\times10^{-8}$ nm inside
$r_c$: double counts it, mixed does not, and the predicted jump, 43.18
kJ/mol/nm, is the measured error, 43.14 and 43.32, along the axis of the
pair, with the predicted energy of 2.92 kJ/mol. An H–H pair $2.2\times10^{-7}$
nm inside flips in the matrix alone (21.59 predicted, 21.6 measured), where
the rounding of the positions differs (absolute positions, D79, against the
frames of the groups, D95). Two causes set the decisions apart:

- the f32 cutoff of the pair kernels is pulled in by about four units in
  the last place (D159), so that kernels with a pole at the cutoff are not
  evaluated there: at 4.5 Å the pairs with $r \ge r_c - 1.15\times10^{-7}$ nm
  are dropped;
- the f32 positions round by up to a unit in the last place of the edge
  of the cell ($4.8\times10^{-7}$ nm at 5 nm).

With the default tolerance, $\mathrm{erfc}(\beta r_c) = 10^{-5}$, the jump is
0.03 to 0.3 kJ/mol/nm for the largest pairs of water and ions from 4.5 to
9 Å, below the usual mixed error. With $\mathrm{erfc}(\beta r_c) = 0.027$ it is
43 kJ/mol/nm for O–H of TIP3P, 86 for O–O, and 124 for K⁺–Cl⁻. The jump
also makes the energy drift, in either precision (docs/pme.md, #346).

**The rule.** The threshold of 1 kJ/mol/nm is the maintainer's: the usual
mixed maximum error of the forces is about 0.4 kJ/mol/nm and the default
tolerance gives at most about 0.3. The largest $\lvert q_iq_j\rvert$ is
exact: the particles are grouped by charge, the products of two charges
are taken in decreasing magnitude, and the first that a pair not excluded
carries is taken. The expected number of pairs in the shell of width
$\delta$ (the pull-in of D159 plus $\mathrm{ulp}_{32}$ of the largest edge) is
$N_q(N_q-1)/(2V)\cdot4\pi r_c^2\delta$ for the $N_q$ charged particles at
their mean density: 0.94 on the box of the issue, whose frame had two. The
message advises removing `beta`, or a tolerance of $10^{-5}$ or less. The
prolate kernel (D266), whose direct term vanishes at the cutoff, is not
checked.

**Alternatives.**

- Restricting the pull-in of D159 to kernels with a pole at the cutoff
  is #809. It would change the emitted constant of every mixed program and
  remove only about half of the effect, since the rounding of the
  positions remains.
- A cutoff test in f64 would need f64 differences of the positions, which
  is the double mode.
- Refusing such an input would change how existing inputs run, and the
  combination has a use: subtracting two cutoffs isolates the pairs between
  them.

**Validation.** `test/Driver/check-pme-cutoff-jump.test` (CPU) checks the
warning against a NumPy reference in double precision on the prmtop of
the solvated dipeptide in OPC: the largest $\lvert q_iq_j\rvert$ is
1.84494 e² (two virtual sites), the jump at 4.5 Å with the $\beta$ of 9 Å is
229.065 kJ/mol/nm with 0.0318 pairs a frame in a shell of
$3.54\times10^{-7}$ nm, and at 9 Å with `tolerance = 1e-3` it is 4.0176.
The defaults at 9 Å and at 4.5 Å (0.0678 and 0.271) give no warning, and
`mdir run` prints nothing new.
