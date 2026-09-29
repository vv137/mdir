# Conventions of Potentials

Status: decided (2026-09-29), D62.

MDIR has one form for each term of the potential energy and one meaning
for each parameter. A reader of a file converts what the file holds into
these forms, and the kernels are built from these forms only, so a system
gives the same program whichever format it was read from. The keys in
brackets are those of [references.md](references.md).

## 1. Units

Inside MDIR (D36):

| Quantity | Unit |
|---|---|
| Length | nm |
| Time | ps |
| Mass | amu (g/mol) |
| Energy | kJ/mol |
| Charge | e, the elementary charge |
| Temperature | K |
| Pressure | bar |
| Angle | radian |

A file or a control file may use other units; its reader converts. The
control file is in Å, kcal/mol, amu, K, and atm, with angles in degrees.

The constants come from CODATA 2018 [[Tiesinga2021]](references.md#tiesinga2021):

| Constant | Value |
|---|---|
| Boltzmann constant `k_B` | 0.0083144626181532 kJ/(mol K) |
| Coulomb constant `f = N_A e² / (4π ε_0)` | 138.935457644 kJ mol⁻¹ nm e⁻² |
| Calorie | 4.184 J |

## 2. Geometry

| Item | Convention |
|---|---|
| Displacement | `d_ab = x_a − x_b`, in the minimum image |
| Angle `θ(a, b, c)` | At `b`, between `d_ab` and `d_cb`, from 0 to π |
| Dihedral `φ(a, b, c, d)` | The IUPAC convention: 0 for cis, π for trans, positive where `a` turns clockwise onto `d` as seen along `b → c`. From −π to π. |
| Improper dihedral | A dihedral of the four members in the order that the topology gives them. For an Amber improper the central atom is the third. |
| Virial | `W = Σ d_ij ⊗ F_ij`, with `F_ij` the force on `i` due to `j`, positive for repulsion (B8). The pressure is `(2K + tr W) / (3V)`. GROMACS writes `−W / 2`. |

The sign of the dihedral is checked by a test: with `b` at the origin, `c`
on the z axis, and `a` on the x axis, `d` on the y axis gives +90°.

## 3. Terms

| Term | Form | Parameters |
|---|---|---|
| Harmonic bond | `½ k (r − r0)²` | `k` in kJ mol⁻¹ nm⁻², `r0` in nm |
| Harmonic angle | `½ k (θ − θ0)²` | `k` in kJ mol⁻¹ rad⁻², `θ0` in rad |
| Periodic dihedral, proper or improper | `k (1 + cos(n φ − φ0))` | `k` in kJ/mol, `n` a whole number, `φ0` in rad |
| Lennard-Jones | `4 ε ((σ / r)¹² − (σ / r)⁶)` | `σ` in nm and `ε` in kJ/mol, for each pair of types (D57) |
| Coulomb | `f q_i q_j / (ε_r r)` | `q` in e, `ε_r` the relative permittivity |
| Pair three bonds apart | `s_LJ · 4 ε ((σ / r)¹² − (σ / r)⁶) + s_C f q_i q_j / (ε_r r)` | the factors `s_LJ` and `s_C` for each pair, and `σ` and `ε` of the pair |

A dihedral with several terms is several tuples on the same members, one
for each `n`.

Lennard-Jones is stored as `σ` and `ε` of each pair of types, the
parameters that force fields are written in. A mixing rule gives them from
those of the two types; a table of pairs of types (NBFIX) sets them for
some pairs. Where a file gives the coefficients `A = 4 ε σ¹²` and
`B = 4 ε σ⁶` (`ACOEF` and `BCOEF` of Amber, `C12` and `C6` of GROMACS),
the reader converts:

```text
σ = (A / B)^(1/6)        ε = B² / (4 A)
```

A pair with `A = B = 0` has `σ = ε = 0` and no interaction. A pair with
only one of the two coefficients is not a Lennard-Jones pair; the reader
rejects it.

## 4. What the readers convert

| Item | Amber (`prmtop`) | GROMACS (`.top`) |
|---|---|---|
| Bond | `k = 2 k_file`: the file has `k (r − r0)²` | As in the file (function 1) |
| Angle | `k = 2 k_file` | As in the file (function 1), with `θ0` from degrees |
| Dihedral | As in the file; the file has divided `k` by the number of paths already | Functions 1, 4, and 9 as in the file, with `φ0` from degrees |
| Lennard-Jones | `σ`, `ε` from `ACOEF` and `BCOEF` of each pair of types (Section 3), in the units of MDIR | `σ`, `ε` as in the file with the combination rule 2 or 3, or from `C6` and `C12` with the rule 1; `[ nonbond_params ]` overrides the pairs it names |
| Charge | `q = CHARGE / 18.2223` | As in the file |
| Pairs three bonds apart | `s_LJ = 1 / SCNB`, `s_C = 1 / SCEE`, for each dihedral; 0 stays 0 | `fudgeLJ` and `fudgeQQ`, or the parameters of `[ pairs ]` |

Energies in kcal/mol are multiplied by 4.184 and lengths in Å by 0.1.

**The Coulomb constant of Amber.** Amber takes `18.2223² = 332.0522`
kcal Å mol⁻¹ e⁻² for `f`; the value of CODATA 2018 is 332.0637. A charge
of a `prmtop` divided by 18.2223 and used with `f` of CODATA gives
electrostatic energies about 3.5 × 10⁻⁵ larger than sander gives. MDIR
keeps its constant. A comparison with sander scales the electrostatic
terms by the ratio of the two constants, or states the difference.

## 5. How the conventions are kept

| Rule | |
|---|---|
| One place builds the kernels | The terms of Section 3 are built by one library of the front end, whatever the format. A reader produces parameters, not kernels. |
| Readers are tested against each other | The same system read from a `prmtop` and from the `.top` that ParmEd converts it to gives the same energy for each term. |
| Terms are tested against definitions | Each term is compared with a script that evaluates Section 3 and checks its forces against finite differences. |
