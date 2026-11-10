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

The one definition site for C++ and the C runtime is
`include/mdir/Support/PhysicalConstants.h` (D394).
Its exact defining inputs are from the BIPM SI Brochure, 9th edition
[[BIPM2019]](references.md#bipm2019); vacuum permittivity is the nominal
CODATA 2022 value [[Mohr2025]](references.md#mohr2025), not an exact SI
defining constant:

| Input | Value | Status |
|---|---|---|
| Elementary charge $e$ | $1.602176634\times10^{-19}$ C | Exact SI definition |
| Boltzmann constant $k$ | $1.380649\times10^{-23}$ J K⁻¹ | Exact SI definition |
| Avogadro constant $N_A$ | $6.02214076\times10^{23}$ mol⁻¹ | Exact SI definition |
| Vacuum permittivity $\varepsilon_0$ | $8.8541878188(14)\times10^{-12}$ F m⁻¹ | CODATA 2022, with standard uncertainty |

Every other value derives from those inputs in the header, with f64
arithmetic and exact conversions: 4.184 J/cal and decimal unit prefixes.
The following values show the compiler-rounded expressions, rather than
independently rounded literals:

| Quantity | Derivation | Native value | Control-file value |
|---|---|---|---|
| Coulomb $f$ | $N_A e^2/(4\pi\varepsilon_0)$, converted from J m/mol | 138.93545755023302 kJ nm mol⁻¹ e⁻² | 332.06371307417066 kcal Å mol⁻¹ e⁻² |
| Molar thermal coefficient | $kN_A$, converted from J/(mol K) | 0.0083144626181532403 kJ mol⁻¹ K⁻¹ | 0.0019872042586408316 kcal mol⁻¹ K⁻¹ |
| Pressure per kJ mol⁻¹ nm⁻³ | $(10^3/N_A)/10^{-27}/10^5$ | 16.605390671738466 bar | 16.388246406847735 atm |
| Mass-density coefficient | $10^{21}/N_A$, for molar mass in g/mol per nm³ | 0.0016605390671738467 g cm⁻³ | Same |

Generalized-Born screening uses the same derived Coulomb and molar
thermal coefficients. With $R_\mathrm{MD}=10^{-3}kN_A$ in kJ mol⁻¹ K⁻¹
and salt concentration $c$ in mol/L,
$\kappa^2 = 8\pi f N_A 10^{-24}c/(R_\mathrm{MD}\varepsilon_r T)$ in nm⁻²;
the method's existing factor 0.73 on $\kappa$ remains unchanged.
The physical-input expressions are checked against an independent
high-precision Decimal calculation and are bitwise identical in C/C++.

Native and control-expression results change bitwise wherever the revised
coefficients enter, deliberately before 1.0.0. The native Coulomb input
changes by about −6.75e-10 relative; the control-expression input changes
by about −6.78e-10. Thermal, pressure and density inputs also derive anew;
no compatibility constants are retained. The coupling lifter reads the
same bits from the compiler-generated `scripts/mdir_physical_constants.py`;
`scripts/generate-physical-constants.py --check` verifies regeneration
from the header. CMake copies this module into `mdir._physical_constants`.
Remaining Python consumers and the duplicate-definition check follow in
part 2 of #805.

### Units and symbols in printed text

The text that the `mdir` tool prints to the terminal (the summary of
`mdir check`, the log of `mdir run`, which its log file copies, and the
warnings and errors of the libraries that the tool reports) is ASCII only
(D[ascii-cli-output], #812). A unit or a symbol is spelled as follows:

| Unicode | Printed | Example |
|---|---|---|
| Å, the angstrom | `Ang` | `cutoff: 9 Ang` |
| an exponent (³, ⁻¹) | `^` and the exponent | `g/cm^3`, `Ang^-1` |
| a quotient of units | `/` | `kcal/mol/Ang` |
| ± | `+/-` | `300.1 +/- 0.4` |
| µ, the prefix micro | `u` | `us` |
| × | `x` | `a grid of 48 x 48 x 48` |
| −, the minus sign | `-` | `-1` |
| °, the degree | `deg` | `90 deg` |
| √ | `sqrt` | `sqrt(x y)` |
| a Greek letter | its name | `alpha`, `beta`, `gamma`, `epsilon`, `lambda`, `sigma` |

`A` would be the ampere. The case-sensitive code of the angstrom in the
Unified Code for Units of Measure (UCUM) is `Ao`; MDIR prints `Ang`, which
a reader recognizes without the code.

The rule covers the terminal. Comments and the documentation keep their
text, and so do two parts of the files that MDIR writes: the line of units
of a file of columns (`Å^3`, `kcal/mol/Å`), which a continued run compares
with the file it extends, and the templates of control files that
`mdir template` prints. Changing those is a change of a file format and
needs a ruling of its own. A string that the tool prints and a file also
holds is ASCII as well: the reasons for the automatic choices of the
neighbor structure and of the skins, which the manifest of a run, the JSON
summaries, the cache of the skins, and the settings that Python resolves
record. An entry of the cache written before, with `Å` in its reason, is
still read: the reader takes the key and the two reaches alone. The messages of the builder of the
program (`lib/Driver/Builder*.cpp`, `lib/Driver/Term*.cpp`) follow in a
change of their own (#813).

`scripts/check-before-push.sh` refuses a string literal with a non-ASCII
character in `tools/mdir/*.cpp` outside a comment. A literal there that goes
into a file, not to the terminal, carries the comment `// ascii-exempt` on
its line (the units of the file of the pulled coordinates). The strings of
the libraries are kept by review.

## 2. Geometry

| Item | Convention |
|---|---|
| Displacement | $\mathbf d_{ab} = \mathbf x_a - \mathbf x_b$, in the minimum image |
| Angle $\theta(a, b, c)$ | At $b$, between $\mathbf d_{ab}$ and $\mathbf d_{cb}$, from $0$ to $\pi$ |
| Dihedral $\phi(a, b, c, d)$ | The IUPAC convention: $0$ for cis, $\pi$ for trans, positive where $a$ turns clockwise onto $d$ as seen along $b \to c$. From $-\pi$ to $\pi$. |
| Improper dihedral | A dihedral of the four members in the order that the topology gives them. For an Amber improper the central atom is the third. |
| Virial | $\mathsf W = \sum \mathbf d_{ij} \otimes \mathbf F_{ij}$, with $\mathbf F_{ij}$ the force on $i$ due to $j$, positive for repulsion (B8). The pressure is $(2K + \mathrm{tr}\,\mathsf W) / (3V)$. GROMACS writes $-\mathsf W / 2$. |

The sign of the dihedral is checked by a test: with $b$ at the origin, $c$
on the $z$ axis, and $a$ on the $x$ axis, $d$ on the $y$ axis gives $+90°$.

## 3. Terms

| Term | Form | Parameters |
|---|---|---|
| Harmonic bond | $\tfrac12 k (r - r_0)^2$ | $k$ in kJ mol⁻¹ nm⁻², $r_0$ in nm |
| Harmonic angle | $\tfrac12 k (\theta - \theta_0)^2$ | $k$ in kJ mol⁻¹ rad⁻², $\theta_0$ in rad |
| Periodic dihedral, proper or improper | $k (1 + \cos(n\phi - \phi_0))$ | $k$ in kJ/mol, $n$ a whole number, $\phi_0$ in rad |
| Lennard-Jones | $4\varepsilon \big((\sigma / r)^{12} - (\sigma / r)^6\big)$ | $\sigma$ in nm and $\varepsilon$ in kJ/mol, for each pair of types (D57) |
| Coulomb | $f q_i q_j / (\varepsilon_r r)$ | $q$ in e, $\varepsilon_r$ the relative permittivity |
| Pair three bonds apart | $s_\text{LJ} \cdot 4\varepsilon \big((\sigma / r)^{12} - (\sigma / r)^6\big) + s_C f q_i q_j / (\varepsilon_r r)$ | the factors $s_\text{LJ}$ and $s_C$ for each pair, and $\sigma$ and $\varepsilon$ of the pair |

A dihedral with several terms is several tuples on the same members, one
for each $n$.

Lennard-Jones is stored as $\sigma$ and $\varepsilon$ of each pair of types, the
parameters that force fields are written in. A mixing rule gives them from
those of the two types; a table of pairs of types (NBFIX) sets them for
some pairs. Where a file gives the coefficients $A = 4\varepsilon\sigma^{12}$ and
$B = 4\varepsilon\sigma^6$ (`ACOEF` and `BCOEF` of Amber, `C12` and `C6` of GROMACS),
the reader converts:

$$\sigma = (A / B)^{1/6}, \qquad \varepsilon = B^2 / (4A).$$

A pair with $A = B = 0$ has $\sigma = \varepsilon = 0$ and no interaction. A pair with
only one of the two coefficients is not a Lennard-Jones pair; the reader
rejects it.

## 4. What the readers convert

| Item | Amber (`prmtop`) | GROMACS (`.top`) |
|---|---|---|
| Bond | $k = 2 k_\text{file}$: the file has $k (r - r_0)^2$ | As in the file (function 1) |
| Angle | $k = 2 k_\text{file}$ | As in the file (function 1), with $\theta_0$ from degrees |
| Dihedral | As in the file; the file has divided $k$ by the number of paths already | Functions 1, 4, and 9 as in the file, with $\phi_0$ from degrees |
| Lennard-Jones | $\sigma$, $\varepsilon$ from `ACOEF` and `BCOEF` of each pair of types (Section 3), in the units of MDIR | $\sigma$, $\varepsilon$ as in the file with the combination rule 2 or 3, or from `C6` and `C12` with the rule 1; `[ nonbond_params ]` overrides the pairs it names |
| Charge | $q = \texttt{CHARGE} / 18.2223$ | As in the file |
| Pairs three bonds apart | $s_\text{LJ} = 1 / \texttt{SCNB}$, $s_C = 1 / \texttt{SCEE}$, for each dihedral; 0 stays 0 | `fudgeLJ` and `fudgeQQ`, or the parameters of `[ pairs ]` |

Energies in kcal/mol are multiplied by 4.184 and lengths in Å by 0.1.

**The Coulomb constant of Amber.** Amber takes $18.2223^2 = 332.0522$
kcal Å mol⁻¹ e⁻² for $f$; MDIR's SI/CODATA 2022 value is
332.06371307417066. A charge
of a `prmtop` divided by 18.2223 and used with $f$ of CODATA gives
electrostatic energies larger by a factor of about 1.00003461951 than
sander gives. MDIR
keeps its constant. A comparison with sander scales the electrostatic
terms by the ratio of the two constants, or states the difference.

## 5. How the conventions are kept

| Rule | |
|---|---|
| One place builds the kernels | The terms of Section 3 are built by one library of the front end, whatever the format. A reader produces parameters, not kernels. |
| Readers are tested against each other | The same system read from a `prmtop` and from the `.top` that ParmEd converts it to gives the same energy for each term. |
| Terms are tested against definitions | Each term is compared with a script that evaluates Section 3 and checks its forces against finite differences. |
