# NEP: the neuroevolution potential as `md` ops

Step 1 of NEP and qNEP (#583, `D[nep-design]`): the design of a term that
evaluates a NEP4 model of GPUMD
[[Fan2021]](references.md#fan2021) [[Fan2022]](references.md#fan2022)
[[Fan2022GPUMD]](references.md#fan2022gpumd) [[Song2024]](references.md#song2024)
from its file `nep.txt`, with its energy, forces, and virial, on the CPU and
the GPU. Training is out of scope.

Status: **design**. Nothing in this document is built. The implementation
is #584, after step 3 of #568 (the builder's interface for registering a
term), so that NEP adds a term in a file of its own. qNEP, the charges of
[[Fan2026]](references.md#fan2026), is #585, after NEP (Section 9).

The result of the design in one sentence: **NEP is the shape of EAM**, a
sum over pairs into a field of each atom followed by a function of each
atom, with a field of many components and the neural stage of the ML4 IR
([ml4-ir.md](ml4-ir.md)) as the function. Its angular terms need no
triplet loop (Section 2), so NEP does not wait for GPU triplets (#108), and
the IR needs no new op, type, or form of gather (Section 3).

## 1. What a NEP4 model computes

Notation of this document. Atom $i$ has the species $t_i$ and the position
$\mathbf x_i$; $\mathbf r_{ij} = \mathbf x_j - \mathbf x_i$ in the minimum
image, $r_{ij} = \lvert\mathbf r_{ij}\rvert$, $\hat{\mathbf u}_{ij} =
\mathbf r_{ij}/r_{ij}$. Lengths are in Å and energies in eV, as in the file;
Section 5.3 converts them.

**Radial functions.** For a cutoff $r_c$, the cutoff function is
$f_c(r) = \tfrac12[1 + \cos(\pi r/r_c)]$ for $r < r_c$ and 0 beyond, and
the basis is

$$
f_k(r) = \tfrac12\,[T_k(x) + 1]\,f_c(r), \qquad x = 2\,(r/r_c - 1)^2 - 1,
\qquad k = 0, \dots, K,
$$

with $T_k$ the Chebyshev polynomials of the first kind, so that $f_0 =
f_c$. The radial function of index $n$ for the pair of species $(t_i,
t_j)$ is a linear combination with trained coefficients
[[Fan2022]](references.md#fan2022):

$$
g_n(r_{ij}) = \sum_{k=0}^{K} c^{\,t_i t_j}_{nk}\, f_k(r_{ij}).
$$

The radial and the angular descriptors have their own cutoffs ($r_c^R$,
$r_c^A$), basis sizes ($K^R$, $K^A$), numbers of functions ($n^R_\max$,
$n^A_\max$), and coefficients. A cutoff may depend on the species: the
cutoff of a pair is $\tfrac12[r_c(t_i) + r_c(t_j)]$.

**Descriptors.** For each atom $i$:

- radial, $n = 0, \dots, n^R_\max$:
  $q^i_n = \sum_{j \ne i} g_n(r_{ij})$;
- angular, three-body, $n = 0, \dots, n^A_\max$ and $l = 1, \dots,
  l_\max$ ($l_\max \le 8$):

  $$
  q^i_{nl} = \frac{2l+1}{4\pi} \sum_{j \ne i} \sum_{k \ne i}
  g_n(r_{ij})\, g_n(r_{ik})\, P_l(\hat{\mathbf u}_{ij}\cdot\hat{\mathbf u}_{ik}),
  $$

  with $P_l$ the Legendre polynomial, and **$j = k$ included**: the
  oracle of Section 8 evaluates this double sum and agrees with calorine
  to rounding (Section 8.2);
- angular, four-body (if the file asks for it), $\kappa\,\mathrm{tr}\,\mathsf T_n^3$,
  and five-body (if asked), $c_5\,\lvert\mathbf v_n\rvert^4$, with
  $\mathsf T_n$, $\mathbf v_n$, $\kappa$, and $c_5$ given in Section 2.3
  [[Fan2022GPUMD]](references.md#fan2022gpumd).

They make the vector $\mathbf D^i$ of $N_\text{des} = (n^R_\max + 1) +
(n^A_\max + 1)\,N_L$ components, $N_L$ the number of angular kinds ($l_\max$
plus one for each many-body term). Each component is multiplied by a
constant of the file, $s_\nu$ (GPUMD's `q_scaler`).

**Network.** One hidden layer of $N_\text{neu}$ neurons with $\tanh$, with
weights of each species and one shared output bias (NEP4,
[[Song2024]](references.md#song2024)):

$$
U_i = \sum_{\mu=1}^{N_\text{neu}} w^{(1)}_{t_i\mu}\,
\tanh\Big(\sum_{\nu} w^{(0)}_{t_i\mu\nu}\, s_\nu D^i_\nu - b^{(0)}_{t_i\mu}\Big) - b^{(1)},
\qquad U = \sum_i U_i \;(+\,U_\text{ZBL}).
$$

$U_\text{ZBL}$ is the universal screened nuclear repulsion at short range
that a model with `zbl` in its header adds over pairs, with its own inner
and outer cutoffs [[Song2024]](references.md#song2024).

## 2. The angular terms are moments of one sum over pairs

### 2.1 The three-body term

The addition theorem of spherical harmonics, with $Y_{lm}$ orthonormal on
the sphere,

$$
P_l(\hat{\mathbf u}\cdot\hat{\mathbf v}) = \frac{4\pi}{2l+1}\sum_{m=-l}^{l}
Y_{lm}(\hat{\mathbf u})\,Y^*_{lm}(\hat{\mathbf v}),
$$

turns the double sum of Section 1 into the squared moments of one sum:

$$
q^i_{nl} = \sum_{j}\sum_{k} g_n(r_{ij})\,g_n(r_{ik})\sum_m Y_{lm}(\hat{\mathbf u}_{ij})Y^*_{lm}(\hat{\mathbf u}_{ik})
= \sum_{m=-l}^{l} \big\lvert A^i_{nlm} \big\rvert^2,
\qquad
A^i_{nlm} = \sum_{j \ne i} g_n(r_{ij})\, Y_{lm}(\hat{\mathbf u}_{ij}).
$$

The double sum runs over all $(j, k)$, $j = k$ included, which is what
makes it factor. With $Y_{l,-m} = (-1)^m Y^*_{lm}$, $\lvert A_{nl,-m}\rvert =
\lvert A_{nlm}\rvert$, so

$$
q^i_{nl} = (A^i_{nl0})^2 + 2\sum_{m=1}^{l}\Big[(\mathrm{Re}\,A^i_{nlm})^2 + (\mathrm{Im}\,A^i_{nlm})^2\Big],
$$

$2l + 1$ real numbers for each $(n, l)$: $(l_\max+1)^2 - 1$ for each $n$, 24
for $l_\max = 4$. The cost is one pass over the pairs within $r_c^A$,
$O(N M)$ for $M$ neighbors, against $O(N M^2)$ for the triplets.

GPUMD evaluates the descriptor so (read for this analysis, version 5.9.1,
`src/utilities/nep_utilities.cuh`, `accumulate_s` and `find_q_one`, and
`src/force/nep.cu`, `find_descriptor`): it accumulates, per atom and $n$,
the unnormalized real parts $g_n(r)\,Q_l^m(z)\,\mathrm{Re}(x+iy)^m$ and
$\mathrm{Im}$ of the unit vector, at index $l^2 - 1$ and after, and forms
$q_{nl}$ from them with the squared normalizations as constants. The
moments are kept for the forces (`sum_fxyz`).

### 2.2 The real harmonics

MDIR builds $Y_{lm}$ from Legendre polynomials, without a table of another
program: for $m \ge 0$,

$$
Y_{lm}(\hat{\mathbf u}) = N_{lm}\, \frac{d^m P_l}{dz^m}(z)\,(x + iy)^m,
\qquad N_{lm} = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}},
$$

with $(x, y, z) = \hat{\mathbf u}$; up to the phase $(-1)^m$, which
$\lvert A\rvert^2$ does not see, this is the associated Legendre function
$P_l^m(z) = (1 - z^2)^{m/2}\,d^mP_l/dz^m$ times $e^{im\phi}$, since
$(x + iy)^m = (1-z^2)^{m/2} e^{im\phi}$. The component $d^mP_l/dz^m$ is a
polynomial of $z$; $(x+iy)^m$ is the $m$-th power of one complex number. The
moments are polynomials of the components of the displacement over $r$: no
trigonometric function, and no division but $1/r$.

### 2.3 The four-body and five-body terms

Both are products of the same moments, so they need no further pass over
the neighbors.

- **Four-body** (`has_q_222`): the cubic invariant of the moments of
  $l = 2$. An $l = 2$ representation has one cubic invariant, and its five
  components are those of the traceless symmetric tensor

  $$
  \mathsf T^i_n = \sum_{j} g_n(r_{ij})\,\big(\hat{\mathbf u}_{ij}\hat{\mathbf u}_{ij}^\mathsf T - \tfrac13\mathsf 1\big),
  $$

  so the term is $\kappa\,\mathrm{tr}\,(\mathsf T^i_n)^3$. GPUMD writes it as a
  cubic polynomial of its unnormalized $l = 2$ components whose
  coefficient of $s_{20}^3$, $s_{20} = \sum_j g_n(3z^2 - 1)$, is $C_4 =
  -0.007499480826664$. On that component alone $\mathsf T =
  \tfrac{s_{20}}{6}\,\mathrm{diag}(-1, -1, 2)$ and $\mathrm{tr}\,\mathsf T^3 =
  s_{20}^3/36$; invariance fixes the rest, so $\kappa = 36\,C_4 =
  -0.269981309759904$.
- **Five-body** (`has_q_1111`): the quartic invariant of the moments of
  $l = 1$, which are the components of $\mathbf v^i_n = \sum_j
  g_n(r_{ij})\,\hat{\mathbf u}_{ij}$: $c_5\,\lvert\mathbf v^i_n\rvert^4$, $c_5 =
  0.026596810706114$ (GPUMD's coefficients of the term are $c_5$, $2c_5$,
  and $c_5$ on $s_z^4$, $s_z^2(s_x^2+s_y^2)$, and $(s_x^2+s_y^2)^2$).

The constants are the normalizations of the file format
[[Fan2022GPUMD]](references.md#fan2022gpumd), given as numbers. The
oracle of Section 8 computes both terms in these Cartesian forms and agrees
with calorine to $2\times10^{-15}$ in the descriptors (Section 8.2), which
checks $\kappa$ and $c_5$.

GPUMD 5.9.1 has four further optional terms (`has_q_112`, `has_q_123`,
`has_q_233`, `has_q_134`), each a product of moments as well. The first
version refuses them (Section 5.2); each is a polynomial of the moments
that the atom stage can add later without a change of the IR.

## 3. The IR

### 3.1 Two stages, one gather, one sum

| Piece | Op | Kernel |
|---|---|---|
| Relation | `md.neighborhood` at the largest cutoff of the model | |
| Radial descriptors and angular moments, $\mathbf a_i = \sum_j \phi(i, j)$ | `md.gather_relation ... exchange(none)` into a field of $N_\text{mom}$ components | the pair stage $\phi$: $g_n$ of both kinds, the $Y_{lm}$ |
| Invariants, scaling, network, $U_i$ | `md.sum_particles` | the atom stage $\psi$ of $\mathbf a_i$ and $t_i$ |

The field holds, per atom, the radial descriptors and the real
components of the moments: $N_\text{mom} = (n^R_\max + 1) + (n^A_\max + 1)
\big[(l_\max + 1)^2 - 1\big]$ components, 202 for the water model of
Section 11 ($n^R_\max = 9$, $n^A_\max = 7$, $l_\max = 4$), 229 for the
carbon model, 125 for the model of 16 metals. The field type of #441 takes
any number of components (`!md.field<@atoms, 202 x f64>`), and `md.stage`
and `md.apply_stage` (D288) take the pair and the atom stage:

```mlir
!pos  = !md.field<@atoms, 3 x f64>
!z    = !md.field<@atoms, i32>
!mom  = !md.field<@atoms, 202 x f64>
!pair = !md.relation<@atoms, 2, unordered>

md.stage @nep_pair forward(@nep_pair.f) vjp(@nep_pair.b)
    : (f64, vector<3xf64>, i32, i32) -> vector<202xf64>
md.stage @nep_atom forward(@nep_atom.f) vjp(@nep_atom.b)
    : (vector<202xf64>, i32) -> f64

md.potential @nep(%x: !pos, %cell: !md.cell, %z: !z) -> f64 {
  %n = md.neighborhood %x, %cell cutoff(0.6) : !pos -> !pair
  %a = md.gather_relation %n, %x, %cell gather(%z : !z) exchange(none) {
  ^bb0(%r: f64, %d: vector<3xf64>, %zi: i32, %zj: i32):
    %m = md.apply_stage @nep_pair(%r, %d, %zi, %zj)
        : (f64, vector<3xf64>, i32, i32) -> vector<202xf64>
    md.yield %m : vector<202xf64>
  } : !pair, !pos -> !mom
  %u = md.sum_particles gather(%a, %z : !mom, !z) {
  ^bb0(%ai: vector<202xf64>, %zi: i32):
    %e = md.apply_stage @nep_atom(%ai, %zi) : (vector<202xf64>, i32) -> f64
    md.yield %e : f64
  } : f64
  md.return %u : f64
}
```

**The two cutoffs in one gather.** The relation is taken at the largest
cutoff of the pairs, radial or angular. The pair stage returns the radial
part for $r < r_c^R(t_i, t_j)$ and the angular part for $r < r_c^A(t_i,
t_j)$, and zero beyond each; $f_c$ and its derivative vanish at the cutoff
($f_c'(r) = -\tfrac{\pi}{2r_c}\sin(\pi r/r_c)$, zero at $r_c$), so the
selection is continuous with its first derivative, as the kernel of any sum
over a relation that changes with the positions must be (ml4-ir.md,
Section 1). GPUMD keeps two neighbor lists instead, one at each cutoff;
MDIR keeps one relation and lets the stage select, so that the pass over
the pairs is one gather. With $r_c^A < r_c^R$ (4 Å and 6 Å in the water
model) the angular part is computed for about $(4/6)^3 \approx 0.3$ of the
pairs; whether a second relation at $r_c^A$ pays for itself is a lowering
choice of #584, measured.

**The species.** The coefficients $c^{t_it_j}_{nk}$ and the weights of the
network belong to the stages: the stages take the species as integer
operands, structural, with no cotangent (ml4-ir.md, Section 2.1).

**Answer to the question of #583.** The angular invariants need no new
form of gather: they are moments, which a gather of D288 with a kernel of
$(r, \mathbf d, t_i, t_j)$ yields, and the invariants are a function of each
atom. No new op, type, or attribute is needed.

### 3.2 The bodies of the stages

The reader of `nep.txt` (Section 5) writes both bodies of both stages,
forward and VJP, from the file: a `func.func` of `arith`, `math`, and
`vector` (the first form of ml4-ir.md, Section 2.1) for the pair stage, and
the batched `linalg` form for the atom stage, whose products with
$w^{(0)}$ are matrix products over the atoms of a tile (the cuBLAS
products of step 4b). The VJPs are closed forms:

- **Atom stage.** With $\mathbf h = \tanh(\mathsf W^{(0)}\mathsf S\,\mathbf D -
  \mathbf b^{(0)})$, $\partial U_i/\partial \mathbf D = \mathsf S\,\mathsf
  W^{(0)\mathsf T}[\mathbf w^{(1)} \odot (1 - \mathbf h^2)]$, and through the
  invariants $\partial q_{nl}/\partial A_{nlm} = 2A_{nlm}$ (twice that for
  the real and imaginary parts with $m > 0$), $\partial\,\mathrm{tr}\,\mathsf
  T^3/\partial\mathsf T = 3\mathsf T^2$ projected on the traceless symmetric
  part, $\partial\lvert\mathbf v\rvert^4/\partial\mathbf v = 4\lvert\mathbf
  v\rvert^2\mathbf v$.
- **Pair stage.** The cotangents $\bar r$ and $\bar{\mathbf d}$ of the
  derivative of $f_k$ (the derivative of $T_k$ by the same recurrence, and
  $f_c'$) and of the polynomials $Y_{lm}(\mathbf d/r)$.

D167 says that MDIR writes no neural derivative of an imported graph and
that external AD generates the backward of such a stage. NEP is not an
imported graph: its functional form is fixed by the file format, as EAM's
is, and the VJPs above are the derivatives of that form. **Ruled** (the
maintainer, 2026-10-27, Section 12): the reader writes the forward and VJP
bodies of both stages in closed form, checked against central differences
of the forward bodies and by the adjoint identity of ml4-ir.md, Section 6,
as for EAM. The network stays in the ML4 neural stage, so D167 holds for
it.

## 4. The derivative

`md-differentiate` derives the forces and the virial from the structure,
by the rules of ml4-ir.md, Section 4.1, which it applies to generalized
Born (D143), to EAM on domains (D279), and to PaiNN in stages (D294):

1. **The sum over the atoms.** $\bar{\mathbf a}_i = \partial U/\partial
   \mathbf a_i = \mathrm{vjp}_\psi(\mathbf a_i, t_i; 1)$: a map over the atoms,
   which gives a field $\bar{\mathbf a}$ of $N_\text{mom}$ components, the
   adjoint of the moments.
2. **The gather.** The pair stage reads no field that depends on the
   positions, so the gather's adjoint has only geometric terms: with
   $(\bar r, \bar{\mathbf d}) = \mathrm{vjp}_\phi(i, j; \bar{\mathbf a}_i)$ and
   $\mathbf g(i, j) = \bar r\,\mathbf d_{ij}/r_{ij} + \bar{\mathbf d}$,

   $$
   \mathbf F_m = -\sum_b \big[\mathbf g(m, b) - \mathbf g(b, m)\big],
   \qquad
   \mathsf W = \sum_{\{i,j\}} \mathbf d_{ij}\otimes\mathbf K(i, j),
   $$

   $\mathbf K(m, b) = -[\mathbf g(m, b) - \mathbf g(b, m)]$, a second pass over
   the pairs that reads $\bar{\mathbf a}$ at both ends: EAM's, where
   $\bar\rho$ is read at both ends of each pair. One `md.gather_relation`
   with two results gives the forces and the virial at each center (ml4-ir.md,
   Section 4.3).

The derivation is that of ml4-ir.md, Section 4.2, with no field entering
the pair kernel. The gather form evaluates the VJP of the pair stage twice
per directed pair (at $(m, b)$ with $\bar{\mathbf a}_m$ and at $(b, m)$ with
$\bar{\mathbf a}_b$). The two-ended scatter of track B (#391 over the
privatized scatter-accumulate of #416, D285) evaluates it once and adds
$\mathbf g$ at $i$ and $-\mathbf g$ at $j$; it is a lowering strategy,
compared with the gather form in #584 (principle 3), and the deterministic
mode keeps the gather. The gathers of the moments and of the adjoint are
the gathers of #419.

For comparison, GPUMD computes the radial forces in the same gather form
(the partial force of each directed pair from the derivative of the
descriptor at both atoms) and stores a partial force of each angular pair
(3 floats for each of up to `MN_angular` pairs of each atom), summed in a
second kernel. Storing per pair against recomputing is the choice of
ml4-ir.md, Section 4.3, item 2.

## 5. Reading `nep.txt`

### 5.1 The file

The file is text, one item per line (GPUMD 5.9.1, `src/force/nep.cu`, the
constructor of `NEP`):

| Line | Content | MDIR |
|---|---|---|
| Header | `nep4` or `nep4_zbl`, the number of species $N_t$, then their element symbols | the species of the model, by element |
| `zbl` (with `nep4_zbl`) | `zbl r_inner r_outer [factor]` | the ZBL pair term (below) |
| `cutoff` | $r_c^R$ $r_c^A$ $M^R$ $M^A$, or a radial and an angular cutoff for each species, then $M^R$ $M^A$ | the cutoffs; $M^R$ and $M^A$ are GPUMD's capacities of its neighbor lists, which MDIR does not use (it sizes its own neighbor matrix, D277) |
| `n_max` | $n^R_\max$ $n^A_\max$ | |
| `basis_size` | $K^R$ $K^A$ | |
| `l_max` | $l_\max$, then the flags of the many-body terms: four-body, five-body, and the four optional ones | |
| `ANN` | $N_\text{neu}$ `0` | |
| Then, one number per line | for each species $w^{(0)}$ (row-major, $N_\text{neu}\times N_\text{des}$), $b^{(0)}$, $w^{(1)}$; the shared $b^{(1)}$; the radial coefficients and the angular ones, each as [basis][pair of species], basis $= n(K+1) + k$ and pair $= t_i N_t + t_j$; the $N_\text{des}$ scaling constants $s_\nu$ | the constants of the stages |

The number of parameters is $(N_\text{des} + 2)\,N_\text{neu}\,N_t + 1 +
N_t^2[(n^R_\max + 1)(K^R + 1) + (n^A_\max + 1)(K^A + 1)] + N_\text{des}$;
the reader checks it against the file and names both counts when they
differ. A model with flexible ZBL has $10\,N_t(N_t+1)/2$ more numbers at the
end.

### 5.2 What is read, and what is refused

Read in the first version: `nep4` and `nep4_zbl` with the universal ZBL
and its two cutoffs; one cutoff or one per species; any $n_\max$,
basis size, and $l_\max \le 8$; the four-body and five-body terms.

Refused, each with a diagnostic that names the file, the line, and what is
not supported:

| Input | Diagnostic |
|---|---|
| `nep3` and older, `nep5` (only GPUMD's replica program reads it) | not a NEP4 model; GPUMD 5.9.1 reads NEP4 only |
| `nep4_temperature`, `nep4_zbl_temperature` | the temperature is an input of the descriptor: not supported |
| `nep4_charge1`, `nep4_charge2`, and their `zbl` forms | qNEP: #585 |
| `nep4_dipole`, `nep4_polarizability` | a model of a tensor, not of an energy |
| `zbl` with a typewise factor, or flexible ZBL (`zbl 0 0`) | not supported in the first version |
| any of `has_q_112`, `has_q_123`, `has_q_233`, `has_q_134` | not supported in the first version |
| a count of parameters other than the header's | the expected and the found counts |
| an element of the topology that is not a species of the model | the element and how many particles have it |

A species of the model that the system does not have is not an error.

### 5.3 Species and units

**Species** are mapped by element: each particle of the topology has an
element (as for `[[energy.model]]`, D270), and its species is the index of
that symbol in the header.

**Units.** MDIR works in nm and kJ/mol. The descriptors depend on lengths
only through $r/r_c$ and the unit vectors, so converting the cutoffs to nm
(a factor of 0.1) leaves them unchanged; the energy is linear in
$w^{(1)}$ and $b^{(1)}$, which the reader multiplies by $96.48533212331$
kJ/mol per eV ($e N_A$, exact in the SI since 2019; the value of
`scripts/mlip/water-box.py`).
The ZBL term is converted as a function of $r$. No conversion runs per
step.

## 6. Keys

Ruled by the maintainer on 2026-10-27 (Section 12). The term is a model
of the existing table `[[energy.model]]`, with a new backend:

```toml
[[energy.model]]
name = "water"
backend = "NEP"
path = "nep.txt"        # relative to the control file
```

and in Python `mdir.ModelTerm("water", "nep.txt", backend="NEP")`
([python-model.md](python-model.md)). `stage_field_precision` keeps its
meaning (Section 7); `centers_per_call`, which chunks the calls of a plugin
model, does not apply to a term that MDIR compiles and is refused with
it. The log names it `MODEL
water` among the terms, with the version of the file, the species, the
cutoffs, and $N_\text{des}$.

Unlike the backends of ML1, NEP is compiled by MDIR (no plugin, no
external framework), so the restrictions of ML1 that come from the plugin
do not apply: the CPU as well as the GPU, `mixed` and `double`, and
`[free_energy]` and tunables later if asked. The first version: one NEP
term, alone (no classical terms beside it), in a periodic
cell (orthorhombic or triclinic), without constraints or virtual sites,
under `[minimize]`, NVE, NVT, and NPT. Combining NEP with classical terms
(for example a NEP solute in classical water, or dispersion beside a
NEP model) is a later decision, with its own rules of exclusion. qNEP
(#585) takes the same key; its mode is read from the header of the file.

## 7. Precision

As for every term ([white paper, Section 7.1](paper/07-precision.md)): positions f64; the
displacement of each pair from f64 positions, then narrowed; the pair
stage, the moments, the atom stage, and the network in f32 in `mixed`; the
energy and the virial summed in f64. The moment field and its adjoint are
staged features, f32 in `mixed` unless `stage_field_precision = "DOUBLE"`
(D329). The parameters are stored in f64 and narrowed where they are
loaded. GPUMD computes in f32 throughout, with f64 positions.

**Measured.** The oracle of Section 8 evaluates the same models in both
precisions (`scripts/validation/nep/reference.py`): f32 displacements,
descriptors, network, moments summed in f32, and the sum of the atomic
energies in f64, against all f64. The difference is the error of the mixed
mode:

| Model, frames | Atomic energy (eV) | Force, max (eV/Å) | Force, rms (eV/Å) | Virial per atom (eV) | Descriptor, max |
|---|---|---|---|---|---|
| Water [[Xu2023]](references.md#xu2023), 8 frames of 384 atoms | $4.1\times10^{-7}$ | $1.3\times10^{-5}$ | $2.6\times10^{-6}$ (rms force 0.72 to 1.6) | $2.9\times10^{-7}$ | $3.0\times10^{-7}$ |
| Carbon [[Fan2024]](references.md#fan2024), 2 frames of 512 atoms | $2.7\times10^{-7}$ | $2.1\times10^{-5}$ | $4.6\times10^{-6}$ (rms force 4.2) | $1.0\times10^{-6}$ | $3.9\times10^{-7}$ |

GPUMD's own f32 evaluation differs from calorine's f64 one by about as
much (Section 8.2: $4.8\times10^{-7}$ eV per atom and $6.3\times10^{-5}$
eV/Å on water). An error of $10^{-5}$ eV/Å in a force is four orders below
the error of the model against its reference data: the root mean square
error of the water model's forces on its test set is 0.073 eV/Å (its
`loss.out` in nep-data, Section 11). The f32 rounding is therefore acceptable in all the
values listed, and nothing per pair needs f64. What #584 measures on top:
the drift of the energy in NVE in `mixed` (the f32 sums of the moments over
about 90 neighbors are the largest sums per pair), and whether f64 storage
of the moments (D329) changes it.

## 8. Validation

### 8.1 Oracles

1. **calorine's CPU evaluator** [[Lindgren2024]](references.md#lindgren2024)
   (`CPUNEP`, f64), on the same model and structures: energy, every force,
   the virial, and the descriptors.
2. **GPUMD** on the same model and structures (f32): energy, forces, and
   virial.
3. **The direct triplet sum.** The CPU reference of #584 computes $q_{nl}$
   by the double sum of Section 1, $j = k$ included, independent of the
   moments, on a small system. It is the test that the moment form is the
   descriptor.
4. **Central differences** of the energy in the positions and in the
   strain of the cell; a short NVE run with its drift.

`scripts/validation/nep/` holds the scripts of this design: `calorine.py`
(the reference values), `reference.py` (an evaluator in PyTorch written
from the formulas of Sections 1 and 2, with both the moment and the
triplet forms, in f64 and in the mixed emulation of Section 7), and
`gpumd.py` (run directories for GPUMD and the comparison).

### 8.2 What was measured for this design

On the water model (8 frames of the model's test set, 384 atoms) and the
carbon model (2 rattled diamond cells, 512 atoms); the oracle and calorine
on the CPU, GPUMD on an RTX 4090:

| Comparison | Atomic energy (eV) | Force, max (eV/Å) | Virial per atom (eV) | Descriptor, max |
|---|---|---|---|---|
| Oracle (f64) against calorine, water | $6.8\times10^{-16}$ | $2.0\times10^{-13}$ | $6.1\times10^{-15}$ | $1.3\times10^{-15}$ |
| Oracle (f64) against calorine, carbon | $1.8\times10^{-15}$ | $1.1\times10^{-13}$ | $6.9\times10^{-14}$ | $2.1\times10^{-15}$ |
| Triplet sum against moments (f64), water, 2 frames | $1.8\times10^{-16}$ | $4.9\times10^{-15}$ | | $2.5\times10^{-16}$ |
| Triplet sum against moments (f64), carbon | 0 | $1.6\times10^{-14}$ | | $6.7\times10^{-16}$ |
| GPUMD 5.9.1 (RTX 4090) against calorine, water | $4.8\times10^{-7}$ | $6.3\times10^{-5}$ | $8.9\times10^{-7}$ | |
| GPUMD 5.9.1 (RTX 4090) against calorine, carbon | $7.1\times10^{-7}$ | $3.6\times10^{-5}$ | $1.1\times10^{-6}$ | |

So the moment form of Section 2, with the four-body and five-body terms
in the Cartesian forms of Section 2.3 and the network of Section 1, is
NEP4 to rounding; calorine is an f64 oracle; GPUMD's differences from it
are those of f32.

### 8.3 Tolerances proposed for #584

Against calorine, per atom for energies and virials:

| Mode | Energy (eV) | Force, max (eV/Å) | Virial (eV) |
|---|---|---|---|
| `double` | $10^{-10}$ | $10^{-9}$ | $10^{-10}$ |
| `mixed` | $2\times10^{-6}$ | $3\times10^{-4}$ | $5\times10^{-6}$ |

The `double` row leaves room for the conversion of units and the order of
the sums; the `mixed` row is about four times the largest difference of
GPUMD from calorine in Section 8.2. Both on the CPU and the GPU.

### 8.4 GPUMD and calorine

GPUMD 5.9.1 (released 2026-10-05; commit 11ba6fbd of its repository),
built with CUDA 13.2 for the RTX 4090 (`sm_89`) and the RTX PRO 6000
(`sm_120`), only the `gpumd` program:

```sh
git clone --depth 1 --branch v5.9.1 https://github.com/brucefan1983/GPUMD.git gpumd-5.9.1
cmake -G Ninja -S gpumd-5.9.1 -B build/gpumd-5.9.1 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  "-DCMAKE_CUDA_ARCHITECTURES=89;120"
ninja -C build/gpumd-5.9.1 gpumd     # 24 s with 24 jobs
```

calorine 3.5 (which brings ASE, NumPy, and SciPy), in an environment of
Python 3.12 made with uv:

```sh
uv venv --python 3.12 calorine
uv pip install --python calorine/bin/python calorine==3.5
```

On the cluster both live under the tree of the reference programs, and
GPUMD runs through the GPU jobs of the machine, as every GPU program does
([workflow.md](workflow.md)). `scripts/validation/nep/reference.py` runs in the
PyTorch environment of the tests (PyTorch 2.14.1, NumPy 2.5).

GPUMD's sources under `src/` are LGPL-3.0-or-later. They were read for
this analysis (the layout of the file, the form of the descriptor); MDIR
re-implements, it copies no code, and the code comments of #584 cite the
papers.

## 9. qNEP: what the charges take, and what not

qNEP [[Fan2026]](references.md#fan2026) adds a second output of the
network, the charge $q_i$ of each atom, a function of its descriptor, and
an electrostatic energy of those charges: $U = U_\text{NEP} + U_\text{ES}$.
Its files have the headers `nep4_charge1` and `nep4_charge2` (and their
`zbl` forms), for its two modes:

- **Mode 2**, reciprocal space only: $U_\text{ES} = U^k$, the reciprocal
  part of the Ewald sum of the charges, the short range being left to the
  network;
- **Mode 1**, real space, reciprocal space, and the self term:
  $U_\text{ES} = U^r + U^k + U^s$, with $\alpha = \pi/r_c$ and the real-space
  cutoff that of the model.

Before the energy, the charges are corrected to the total charge $Q$ of the
structure (zero in the paper): $q_i \to q_i - \tfrac1N(\sum_j q_j - Q)$
(eq 33 of the paper). The forces have a static part (the $1/r$ dependence
at fixed charges) and a dynamic one (the charges depend on the positions
through the descriptors): the chain rule through $q_i(\mathbf D_i)$, with
the potential at each particle of the grid that D231 yields, gives the
cotangent of each charge, which enters the atom stage's VJP beside $\partial
U_i/\partial\mathbf D_i$; the correction of the total charge is linear, so
its transpose subtracts the mean of the cotangents.

In MDIR the reciprocal part is `md.reciprocal` (PME) with the charges as a
computed field, not a field of the topology. In scope of #585: both modes,
the charge correction, energy, forces, and virial, against GPUMD's qNEP and
against a direct Ewald sum, and whether MDIR's PME (B-splines and its
influence function) or the PPPM optimal influence function of the paper
(eq 36; related to #468) is used, with the accuracy measured. Out of scope
of #585: Born effective charges, polarization, external fields, and
training.

## 10. Performance plan

Measured in #584, on one RTX 4090, against GPUMD 5.9.1 on the same model,
structure, and device, the two programs alternated in one exclusive GPU job
(`gpu-run --idle`):

- **Systems.** Water at the sizes of Figure 2 of the qNEP paper, whose
  axis runs from $10^2$ to $10^6$ atoms: the 384-atom cell of the water
  model's test set replicated $n^3$ times, $n = 1, 2, 3, 4, 6, 8, 10, 14$
  (384 to 1,053,696 atoms). The model is that of [[Xu2023]](references.md#xu2023),
  not the paper's own NEP model of water, so the rates are compared between
  MDIR and GPUMD on this model, not with the paper's figure.
- **Rate**: atom-steps per second and ns/day at a step of 0.5 fs, as the
  paper reports them, over steps after a warm-up, NVE; the neighbor list's
  skin and rebuilds as each program chooses, stated.
- **Peak device memory** of each program at each size (the headline of
  M3): MDIR's neighbor matrix, the moment field and its adjoint ($2N_\text{mom}$
  floats per atom in `mixed`), against GPUMD's lists and stored partial
  forces.
- **Also**: `mixed` against `double` on the GPU, and the CPU's rate on the
  smaller sizes.
- **No loss** (#584): existing programs keep `mdir emit --stage=lowered`
  and their PTX byte for byte.

Ratios are reported plainly.

## 11. Reference models

| Model | Source | Paper | License |
|---|---|---|---|
| Water, `nep4`, O and H, $r_c^R$ = 6 Å, $r_c^A$ = 4 Å, $l_\max = 4$ with the four-body term, 100 neurons | `2023_Xu_liquid_water/nep.txt` of the repository nep-data (gitlab.com/brucefan1983/nep-data, commit de9d2d35) | [[Xu2023]](references.md#xu2023) | none stated in the repository |
| Carbon, `nep4`, $r_c^R$ = 7 Å, $r_c^A$ = 4 Å, four-body and five-body terms | `potentials/nep/C_2024_NEP4.txt` of GPUMD 5.9.1 | [[Fan2024]](references.md#fan2024), as GPUMD's `potentials/nep/readme.md` cites it | GPUMD's license covers `src/`; none stated for the models |
| 16 metals (UNEP-v1), `nep4_zbl`, Ag Al Au Cr Cu Mg Mo Ni Pb Pd Pt Ta Ti V W Zr | `potentials/nep/Song-2024-UNEP-v1-AgAlAuCrCuMgMoNiPbPdPtTaTiVWZr.txt` of GPUMD 5.9.1 | [[Song2024]](references.md#song2024) | none stated |

No license is stated for the model files. As ruled (Section 12), the
validation scripts of #584 download them, the lit tests use generated
models of the same layout with random parameters, and nothing of the model
files is committed.
GPUMD's own test fixtures include a small water model of `nep4` and one of
`nep4_charge1` (`tests_pytest/fixtures/models/`), for qNEP's tests.

## 12. What was ruled (the maintainer, 2026-10-27, on #653)

1. **Names.** `[[energy.model]]` with `backend = "NEP"` and `path`, and
   `mdir.ModelTerm(..., backend="NEP")` in Python. qNEP takes the same key,
   its mode read from the header of the file.
2. **Scope of the first version.** NEP alone, without classical terms
   beside it, on the CPU and the GPU, in `mixed` and `double` (Section 6).
   Combining it with classical terms is a later decision.
3. **The VJP bodies.** The reader of `nep.txt` writes the forward and VJP
   bodies of the pair and atom stages in closed form, checked by central
   differences and the adjoint identity of ml4-ir.md, Section 6, as for
   EAM (Section 3.2). The network stays in the ML4 neural stage, so D167
   holds for it.
4. **Reference models.** The validation scripts download them; the lit
   tests use generated models of the same layout with random parameters;
   nothing of the model files is committed (Section 11).

## 13. Order of the work

1. This design (#583).
2. #584 after #568 is done (its step 3, the builder's interface for registering a term): the reader, the two stages, the term, the
   CPU reference with the triplet sum, validation (Section 8) and
   performance (Section 10).
3. #585 after #584: qNEP (Section 9).

#108 (GPU triplets) stays independent: NEP and qNEP do not need it. It
matters for potentials whose three-body terms are not moments
(Stillinger–Weber, Tersoff, symmetry functions G4 and G5 written directly).
