# Reciprocal sums as differentiable primitives

Status: design approved for [#452](https://github.com/vv137/mdir/issues/452),
D388, by the
[maintainer's ruling](https://github.com/vv137/mdir/pull/781#issuecomment-6716738603)
on 2026-11-08, including the additions in Section 7.
This document records the remaining composition. D306, D319, D341, and
D360 already supply the FFT operations, their adjoints, and storage.
It adds no implementation or numerical result.

## 1. Boundary and rollout

Expose charge assignment, the FFTs, influence mapping, and interpolation
before differentiation. Keep their mathematical meaning separate from
bricks, tiles, fixed point, reduction trees, FFT plans, and placement on a
device. Those remain execution strategies.

The current **md.reciprocal** remains the builder's compact spelling and
the default execution route during development. A proposed compiler pass,
**md-expand-reciprocal**, expands explicitly selected supported operations
before **md-differentiate**. This is a compiler development selection, not
a new control-file key, environment variable, Python option, file format,
or overwrite policy. Existing callers retain their requests and units.
Unsupported derivative requests continue to fail by operation and operand;
they do not receive a zero or fall back to the opaque force result.

The proposed first implementation covers Gaussian Coulomb PME on a
rectangular cell, SPME and OPTIMAL, followed by the remaining variants in
Section 6. Switching the ordinary run to this route is a later, separately
reviewed rollout after the compatibility and performance gates. A design
merge does not close #452.

## 2. Meaning and pairings

Notation is local to this design:

| Symbol | Meaning |
|---|---|
| $i,k$ | A particle and a periodically wrapped grid index |
| $a_{ki}(\mathbf x,H)$ | Product of assignment-window weights for the stencil entry from particle $i$ to grid point $k$ |
| $A(\mathbf x,H)$ | The assignment matrix with entries $a_{ki}$ |
| $\rho,\widehat\rho,\varphi,p$ | Assigned real grid, its half-spectrum, real grid potential, and potential at particles |
| $\mathcal F,\mathcal F^\dagger$ | The existing unnormalized forward FFT and backward adjoint |
| $D$ | The existing real influence multiplier, including its physical prefactors and normalization |
| $\langle\ ,\ \rangle_{\mathrm H}$ | Hermitian-weighted real pairing on valid half-spectra |
| $\mathbf t,\dot A$ | A fixed position direction and the directional derivative of $A$ |

For a fixed grid, order, and window, the reciprocal energy is the quadratic
form already described in [pme.md](pme.md) and the paper:

$\rho=Aq,\quad\widehat\rho=\mathcal F\rho,\quad
U_{\mathrm{rec}}=\tfrac12\langle\widehat\rho,D\widehat\rho\rangle_{\mathrm H}.$

Its reverse chain yields
$\varphi=\mathcal F^\dagger D\widehat\rho$ and
$p=A^{\mathsf T}\varphi=\partial U_{\mathrm{rec}}/\partial q$.
The forward energy stays a spectral reduction, matching the existing
convolution's energy computation; rewriting it as a particle sum would
change floating-point reduction order.

This chain is the grid contribution alone. Self energy, the neutralizing
background, excluded-pair corrections, and dispersion tails keep their
existing ownership and derivative chain. Their parameter and strain
contributions are included when checking the complete potential; they are
not added a second time by the grid composition.

The real-grid and particle pairings are Euclidean sums. In the
half-spectrum pairing, take the real part of the complex product, weight
interior last-axis frequencies twice, and weight zero and the even-grid
Nyquist plane once. Both FFT directions are unnormalized:
$\mathcal F^\dagger\mathcal F=K_1K_2K_3 I$.
No additional inverse-grid factor is inserted by transposition. Influence
normalization is audited against the existing templates, not inferred from
a library's convention.

The half-spectrum is a value in the valid Hermitian subspace, including
conjugacy across the other axes on its boundary planes. Influence
construction must preserve that subspace. Odd/even grids and triclinic
boundary modes are explicit gates before a variant is admitted; an
unchecked packed array is not proof of this property. Any required change
to legacy boundary-mode treatment needs a separate numerical decision.

## 3. Semantic operations and their transposes

The operation names below are approved design, not implemented syntax.
Use a particle-grid stencil relation and structured kernels rather
than a new opaque PME call with another handwritten force result.

| Semantic operation (approved design) | Contract | Declared transpose |
|---|---|---|
| **md.grid_stencil** | A typed relation between a named particle set and a static periodic grid; support order and window are explicit. Entries identify particle and wrapped grid index. Continuous local coordinates remain separate SSA operands of the weight kernel. | Integer index enumeration is structural; derivatives flow through the explicit continuous operands, not a cotangent of an integer relation. |
| **md.scatter_stencil** | A pure scatter-sum value onto a fresh grid. Its scalar/vector kernel reads the relation's local coordinates and particle operands. | A gather over the same entries, with the reverse derivative of that same kernel. In the amplitude alone this is $A^{\mathsf T}$. |
| **md.gather_stencil** | A pure gather/reduction at particles from a grid, using the same relation and kernel definition. | A scatter-sum onto the grid. In the grid operand alone this is $A$. |
| **md.map_grid**, **md.sum_grid** | Pure structured maps and reductions on real grids or valid half-spectra; the latter declares its pairing and reduction algebra. | Reverse the scalar/complex kernel and the map/reduction structure. A real diagonal influence map has transpose $D$ in its spectrum operand. |
| **md.linear_transform** | Existing FFT value operation, preserving its direction, adjoint, and pairing declarations. | Existing opposite-direction transform, unchanged. |

The stencil is a reusable relation, not a stored dense assignment matrix.
It stores no hidden active real payload. Positions, cell, and the real
local-coordinate calculation remain explicit inputs and arithmetic of the
structured weight kernel. The same positions may determine discrete support
indices and remain active through that kernel; structural index selection
does not prove the position operand independent.
Shapes, particle-set identity, grid extents, periodic index bounds, window
support, scalar types, and result coverage are verified. Every full
particle result defines the omitted rows as zero when a list is used.
The same relation and window body generate spread and gather; separate
copies of spline formulas in the two derivative implementations are
not introduced.

For amplitude and grid operands, reuse the existing declared linear
transpose mechanism. For position, cell, and influence operands, extend
the structured-kernel derivative mechanism explicitly. The current
AdjointOpInterface's convention that non-linear operands are structural
does not license declaring these operands inactive. Each dependent operand
needs a JVP/VJP rule or an unsupported-contract diagnostic. This is a
prerequisite of the reciprocal composition, not a follow-up after charge AD.

At fixed cell, the assignment JVP and VJP are:

$\dot\rho=A\dot q+\dot A q,\qquad
\bar q=A^{\mathsf T}\bar\rho,\qquad
\bar{\mathbf x}_i=q_i\sum_k\bar\rho_k\nabla_{\mathbf x_i}a_{ki}.$

For gathering $p=A^{\mathsf T}\varphi$:

$\dot p=A^{\mathsf T}\dot\varphi+\dot A^{\mathsf T}\varphi,\qquad
\bar\varphi=A\bar p,\qquad
\bar{\mathbf x}_i=\bar p_i\sum_k\varphi_k\nabla_{\mathbf x_i}a_{ki}.$

Cell derivatives contract the same window derivatives with the derivative
of fractional coordinates. Influence derivatives include the reciprocal
metric, volume, splitting parameter, and any differentiable factors.
Integer grid sizes, support order, and choice of window are structural.
Host-generated moduli or coefficients are constants only when activity
analysis proves independence of the requested operand.

The relation enumerates a local polynomial piece. Differentiation follows
its continuous weights, not an integer derivative of floor or wrapping.
Boundary tests check agreement of the complete periodic stencil across
changes of enumeration. A higher derivative is supported only for a window
with the required validated smoothness and derivative rules. Default
order-four B-splines are the first curvature case; unsupported orders or
windows receive a named diagnostic.

## 4. C(ii), HVP-3, and strain

This section states the consumers' contracts before choosing an emitter.
It does not add a new evaluator API. The forward tangent of #778 retains
its two reciprocal sums along the direction. A possible migration to this
chain's JVP is decided by measurement in slice 3, not by this design or
slices 1 and 2. At fixed independent charges,
$\partial U_{\mathrm{rec}}/\partial\mathbf x_i
=q_i\sum_k\varphi_k\nabla_{\mathbf x_i}a_{ki}$; forces have the negative
sign. Position-dependent charges require their upstream chain rule, or a
diagnostic if that dependency lacks a rule.

For force-matching C(ii), differentiate the position tangent
$\dot U_{\mathbf x}=\langle\dot A q,\varphi\rangle$ in the charge field:

$\partial\dot U_{\mathbf x}/\partial q
=\dot A^{\mathsf T}\varphi+A^{\mathsf T}\dot\varphi,\qquad
\dot\varphi=\mathcal F^\dagger D\mathcal F(\dot A q).$

Both terms are required. This replaces the present per-scalar
polarization workaround with a field derivative through the same chain;
the host still applies the existing tunable-entry mapping.

For HVP-3, at fixed cell and charges:

$(\mathsf H\mathbf t)_i
=q_i\sum_k\left[
\dot\varphi_k\nabla_{\mathbf x_i}a_{ki}
+\varphi_k(\nabla^2_{\mathbf x_i}a_{ki})\mathbf t_i
\right].$

The first term couples particles through another reciprocal propagation;
the second is assignment-window curvature. Differentiating only gathering,
or holding the assigned grid fixed, omits the first term. No Hessian is
materialized. JVP and reverse rules for the stencil kernels and grid maps
must compose in either order on the supported subset, including derivatives
of the rules themselves.

Extra points continue to use the caller's placement tangent and transpose.
For a position derivative of a force product, placement curvature remains
the separate D387 contribution. Assignment-window curvature above does not
replace or duplicate it. Test the composition on both linear sites and the
Amber bisector.

Cell and strain follow the existing row-cell convention and the simultaneous
affine variation of positions and cell, with
$\partial U/\partial\varepsilon_{ab}=-W_{ba}$.
The design declares both dependencies so the virial can be derived and
checked. It does not silently enable unsupported higher cell derivatives
in the evaluator or framework adapters.

## 5. Execution, algebra, and numerical contracts

Recognize proved compositions after differentiation and lower them through
the existing spreading, convolution, gathering, and FFT strategies. The
recognizer must establish relation identity, pairing, influence, requests,
types, ownership, and algebra before selecting a fused template. Unproved
compositions take a correct explicit primitive lowering; a derivative with
no such lowering is refused. A marker alone never licenses fusion.

Keep spectral energy/virial reductions and their existing order. Reuse
the default scatter choices and deterministic fixed-point strategy under
their declared algebra. The semantic assignment is linear in amplitudes;
the quantized fixed-point implementation is not literally linear on bits.
Its value and derivative tests therefore state the ideal-operator identity
and the implementation's rounding bound separately. They do not assert an
exact numerical adjoint identity for a quantizer or differentiate integer
rounding as if it were the physical model.

Reuse the existing mixed/double type boundaries and f64 energy and
derivative reductions. Every implementation slice provides a precision
table comparing the new and legacy routes: positions as loaded, window
weights, assigned grid, FFT, influence, gathered potential, forces, and
each accumulator. State every difference before any performance conclusion;
compare timings only between routes of the same types. No promotion of
the entire chain to f64 is hidden in a Mixed result.

The execution forms declare operand effects, indexed accumulation, row
coverage, and reduction algebra through D342/D301 and the existing effects
interfaces before fusion or placement. D360 and D358 govern packed FFT
ownership, destructive inverse inputs, producer queues, borrow completion,
readback, and release. No intermediate host round trip or copy removal is
claimed without those proofs. Target capabilities come from the target
description; subgroup width, shared memory, f64, and atomics are explicit
preconditions. NVIDIA instructions remain inside NVIDIA emitters.

## 6. Implementation sequence and gates

Each implementation slice is a draft PR and keeps its own scope and gates.

1. Add the stencil relation, structured spread/gather, and inspectable
   transposes/JVPs. Verify positive and negative operand-activity cases,
   coverage, index boundaries, and effects. No ordinary PME route changes.
2. Add grid maps/reductions and the opt-in rectangular Gaussian chain,
   preserving the existing FFT pairing and normalization. Validate primal
   values, forces, charge fields, and strain before admitting the route.
3. Compose C(ii) and HVP-3, including both terms in each formula above and
   the existing site derivatives. Remove a refusal only for the tested
   supported subset. Existing scalar derivative paths remain available.
   Measure and seek a decision in this slice on whether #778's forward
   tangent should migrate from its existing two-sum construction to the
   JVP route; it remains unchanged until that decision.
4. Cover triclinic Gaussian PME, P3M's supported cells, prolate windows, and
   LJ-PME's coefficient fields and zero mode in separately gated slices.
   Preserve listed-particle semantics: a zero charge can still have a
   nonzero charge derivative, so an amplitude-dependent list must not erase
   its cotangent. Lists are rebuilt with particle reordering as today.
5. Select resident execution and existing fused strategies only after
   equivalence, lifetime, and whole-step performance gates. A default-route
   switch requires a separate ruling. Distributed FFT transport and DMK
   remain outside this implementation, while sharing the operator contracts.

Validation is specified before implementation measurements:

- Independent small-grid NumPy direct DFT and window references: odd/even
  axes, zero/Nyquist modes, rectangular/skew cells, single modes, and
  normalization. Weighted FFT and spread/gather adjoint identities use
  independently formed inputs, including signed amplitudes and zero charges.
- Forward/reverse duality and cross-checks of the same scalar function:
  position/charge/cell JVPs, VJPs, charge-field force products, and HVP
  symmetry. Curvature checks include local stencil boundaries and nonlocal
  cross-particle responses. No missing rule is accepted as a zero.
- CPU/GPU, Mixed/Double, ordinary/deterministic, windows and orders, sparse
  lists, cells, and virtual sites. Compare energy, forces, all virial
  components, and derivatives against the legacy route and independent
  references. torch-pme supplies a separate differentiable Gaussian PME
  check at matching discretization; its tolerances come from the existing
  PME validation contract. Quantization bounds come from the actual scale,
  rounding rule, and number of stencil contributions. Establish numerical
  thresholds in each implementation PR before its first comparison.
- The existing module/lowered corpus, order matrix, five checkpoints,
  full D309 suite, and device sanitizer tier when device code changes.
  Unselected routes stay byte-identical. Default NVIDIA kernels are kept
  byte-identical where existing emitters are reused; any selected-route
  difference is reported and ruled on rather than relabeled equivalent.
- Whole-step and frame-product timings against the same main on the same
  device model, including the Amber suite and OPC systems. Count launches,
  transforms, transfers, and scratch; inspect plans and lifetimes.
  No slowdown or overlap claim follows from removing an opaque IR node.

## 7. Maintainer's ruling

The [2026-11-08 ruling](https://github.com/vv137/mdir/pull/781#issuecomment-6716738603)
approves all four contracts: the structured particle-grid relation and
spread/gather transpose pair; explicit nonlinear and higher-derivative
rules before C(ii)/HVP-3; staged coverage and opt-in rollout; and the
compatibility, quantization, and performance gates in Section 6.
Order-four B-splines are the first curvature case, both terms of each
product remain required, and D387's placement curvature stays separate.

The ruling adds two requirements:

- Each implementation slice gives the new-versus-legacy precision table
  listed in Section 5 before any performance conclusion; any difference
  is stated and timings compare routes of the same types.
- #778's forward tangent keeps its two reciprocal sums. Whether to migrate
  it to the JVP route is decided in slice 3 by measurement.

Implementation proceeds in the slices of Section 6. The ordinary
reciprocal route and its strategies remain the default; changing that
default requires a ruling of its own.
