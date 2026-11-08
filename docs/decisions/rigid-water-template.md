# D[rigid-water-template]

Recorded at its merge.

**Propose a rigid-water template for the short-range pairs on a device in mixed precision: waters that match a template become units anchored at their oxygen, listed by a covering bound on the anchors, evaluated by one warp-cooperative fused kernel over water–water, solute–water, and solute–solute cluster pairs with the template's parameters as constants and the extra point formed in the kernel, under an execution choice `water.template` (`AUTO`, `ON`, `OFF`).**

Status: design accepted on #795; implementation in progress. This decision
closes #774 when its implementation gates pass. The seven accepted choices
are recorded under [Maintainer's ruling](#maintainers-ruling).

The existing measurements below are those of the prototype, not validation
of the implementation. The
derivation is in Section 4.7 of the white paper and the paragraph on the
template's extra point in Section 6.1; this file states what is decided,
why, and the gates.

## Why

In an aqueous system most pairs within $r_c$ are pairs of water sites:
93.3% in ubiquitin with OPC and 0.15 M KCl (25,929 atoms, 6,166 waters),
against 4.1% solute–water, 2.3% solute–solute, and 0.26% involving ions.
The general structures (white paper, Sections 4.3 to 4.5) treat every site
as a particle: an OPC water is four members of the lists, four positions
tested for displacement, and 16 site pairs per pair of molecules whose
charges and types are read from tables. When the water is rigid, its
geometry and parameters are known when the program is compiled.

Step 0 of #774 measured the alternatives first (RTX 5090): skipping zero
terms in the present kernel was 1.4 to 3.3% slower per step, grouping the
particles by class 1.3 to 8.7 times slower, and a kernel of one thread per
pair of molecules 7 to 10 times slower than `main`'s group kernel. Only the
warp-cooperative layout, (c3) of #774, gained. Stages 1 and 2 then measured
the whole short-range path of that layout on an RTX 4090.

## What is decided

### 1. The template and its match

A *template* is a water model that the compiler recognizes by matching the
topology, molecule by molecule. A molecule matches when all of these hold:

- the atom count and order: O, H, H for three sites; O, H, H and one extra
  point for four;
- the charges, equal to the template's;
- the Lennard-Jones types, with Lennard-Jones on the oxygen alone in the
  first version (the hydrogens' types carry $\varepsilon = 0$);
- three distance constraints, O–H, O–H, and H–H, that hold the molecule
  rigid (`rigid_water`, SETTLE in double and M-SHAKE in mixed precision);
- for four sites, the extra point's definition: Amber's distance along the
  bisector, with its owner the oxygen and its two atoms the hydrogens;
- no other nonbonded term on its sites: no `[[energy.pair]]` term or
  `[[energy.parameter]]` that reaches them, no LJ-PME, and not in the
  selection of `[free_energy]`.

Molecules that do not match stay on the general path, and existing inputs
run as today. `mdir check` and the run's summary give each reason a count
(below). CHARMM's TIP3P, whose hydrogens carry Lennard-Jones
($\varepsilon = 0.046$ kcal/mol, $R_\mathrm{min}/2 = 0.2245$ Å), is a
different template; the first version reports it as a mismatch, "LJ on the
hydrogens". Making the set of LJ sites a parameter of the template is
later work, to be measured before it is chosen, since a pair of such
molecules has 9 LJ terms instead of 1.

Flexible water and partially constrained water stay on the general path.
Single-site ions are not a template in the first version: they go into
the solute clusters (below), where stage 2 measured them as part of the
solute–water and solute–solute work.

### 2. The anchor and the covering bound

Each matched molecule $A$ is listed by its oxygen, the anchor
$\mathbf x_{o(A)}$. Its *extent* is
$\varrho_A(\mathbf x) = \max_{a\in A}\lVert\mathbf x_a - \mathbf x_{o(A)}\rVert$,
the extra point included. The compiler takes

$$
r_\mathrm m = \max\big(r_\mathrm{OH},\ r_{os}\big) + \delta_\mathrm m ,
$$

the largest distance of a site from the oxygen on the rigid geometry: the
constraint length $r_\mathrm{OH}$, and for four sites the distance
$r_{os} = \alpha\lVert\mathbf d_j + \mathbf d_k\rVert$ of the extra point,
from its weight $\alpha$ (Section 3),
plus a margin $\delta_\mathrm m$ for the residual of the constraints. For
the models in use the hydrogens set it: OPC $0.8724 + \delta_\mathrm m$ Å,
TIP3P $0.9572 + \delta_\mathrm m$ Å; the prototype took
$\delta_\mathrm m = 10^{-3}$ Å, and the largest extent it observed on its
trajectories was 0.872444 Å (OPC) and 0.957214 Å (TIP3P).

**Covering bound.** If $\varrho_A, \varrho_B \le r_\mathrm m$, the
triangle inequality gives, for every site $a\in A$ and $b\in B$,
$\lVert\mathbf x_a - \mathbf x_b\rVert_\mathrm{mi} \ge
\lVert\mathbf x_{o(A)} - \mathbf x_{o(B)}\rVert_\mathrm{mi} - 2r_\mathrm m$,
so every site pair within $r_c$ belongs to a pair of molecules whose
anchors are within $r_c + 2r_\mathrm m$. The water list is therefore a
structure over the anchors with the cutoff $r_c + 2r_\mathrm m$ and the
reach $R + 2r_\mathrm m$, and the theorem of validity of Section 4.1 (D80)
applies to it unchanged, with $d$ the largest displacement of an anchor:

$$
2d \le \min(\mathbf m)\,(R + 2r_\mathrm m) - (r_c + 2r_\mathrm m) .
$$

For a solute particle $i$ and a water $B$ the bound is
$\lVert\mathbf x_i - \mathbf x_b\rVert_\mathrm{mi} \ge
\lVert\mathbf x_i - \mathbf x_{o(B)}\rVert_\mathrm{mi} - r_\mathrm m$; the
solute–water list has the cutoff $r_c + r_\mathrm m$ and the reach
$R + r_\mathrm m$, and its test takes $d$ over the solute particles and the
anchors. The solute–solute list is the general one. The inner lists of
the dual list (D114) take $R_\mathrm{in}$ in place of $R$ and the same
conditions. In the cell of the build ($\mathbf m = \mathbf 1$) every
condition is the half-skin test $d \le s/2$. The proof needs the extents
at the current positions, not only at the build: the extent check of
Section 6 below provides it.

A molecule is read whole: the packing takes each site in the image of its
oxygen, which is exact because $r_\mathrm m$ is far below half an edge.
The lists hold every image of every anchor pair within their reach when
$R + 2r_\mathrm m < \min_a L_a$, the condition of D242 with the reach of
the water list; the builder refuses a longer reach and the run stops a
barostat that takes the cell below it, as D242 does.

**The cutoff stays a cutoff of sites.** The lists only select pairs of
molecules. Each site pair is still evaluated with its own $r < r_c$, so
the energy is that of the general path. A cutoff on the anchors' distance
would include or leave out whole molecules at the boundary, a different
model [[Steinbach1994]](../references.md#steinbach1994); the template does
not use it.

### 3. The extra point: placed in the kernel with constant weights

On the rigid geometry the bisector placement of Amber's extra point is the
linear combination

$$
\mathbf x_s = \mathbf x_o + \alpha(\mathbf d_j + \mathbf d_k), \qquad
\alpha = \frac{r_{os}}{2r_\mathrm{OH}\cos(\theta_\mathrm{HOH}/2)},
$$

with $\mathbf d_j = \mathbf x_j - \mathbf x_o$ and $r_{os}$ the distance of
the site from the oxygen ($\alpha = 0.1477206$ for OPC). The template
kernel forms the extra point from the three atom positions with this
constant $\alpha$: a molecule is read as three positions, and the site is
never loaded or stored for the short-range pairs. The definition of the
site in the topology is kept (#774): force outputs, derivatives, and every
path outside the template still use the bisector.

The maintainer chose rule (A) for dynamics (white paper, Section 6.1).
Rule (B), used by the prototype, is described for comparison:

- **(A) The constant transpose of the linear placement.** The oxygen takes
  $(1 - 2\alpha)\mathbf f$ and each hydrogen $\alpha\mathbf f$. These are the
  exact derivatives of the energy that the kernel evaluates, at every
  configuration; the site needs no accumulator and no geometry, and its
  correction of the virial, $\sum_k(\mathbf x_k - \mathbf x_s)\otimes\mathbf
  F_k$, is zero.
- **(B) The bisector Jacobian evaluated with the rigid geometry's
  constants**, which the stage 2 prototype used: with
  $\hat{\mathbf u} = \mathbf d_j/r_\mathrm{OH}$,
  $\hat{\mathbf v} = \mathbf d_k/r_\mathrm{OH}$,
  $\hat{\mathbf b} = (\mathbf d_j + \mathbf d_k)/\lVert\mathbf d_j + \mathbf d_k\rVert_\mathrm{rigid}$,
  and $\mathbf g = \mathbf f - (\mathbf f\cdot\hat{\mathbf b})\hat{\mathbf b}$,
  hydrogen $j$ takes $\alpha(\mathbf g - (\mathbf g\cdot\hat{\mathbf u})\hat{\mathbf u})$,
  hydrogen $k$ likewise with $\hat{\mathbf v}$, and the oxygen the rest. No
  square root or division; on frames constrained by SETTLE it matched
  `main`'s spreading to $1.3\times10^{-9}$ (#774, (c3)). Its virial needs
  the correction above.

On the manifold of the constraints the two rules, and `main`'s bisector,
give atom forces that differ by a vector in the span of the gradients of
the water's three constraints (Section 6.1 of the paper proves it). The
constraint solver removes such a vector exactly, the total force and
torque of the molecule are unchanged, and so are the constrained
trajectory, the kinetic energy, and the virial of the rigid groups that
the barostat takes. What differs is the unconstrained atom force that an
output would write, and everything off the manifold: the coordinates of
the input before the first step (which are not constrained, as in sander)
and the residual of the constraints.

The two rules therefore have different validation targets.
*Atom-wise equality* of the forces with `main` holds for (B) on the
manifold and not for (A); *dynamical equivalence* under rigid constraints
holds for both. The gates of each are under Gates below.

### 4. The class-segmented pair list

One spatial sort per kind, with the same keys: water clusters of 16 whole
molecules, sorted by their anchors, and solute clusters of 16 particles,
sorted as the groups of D89, with the ions among them. Cluster pairs are
tagged:

- **WW**, water cluster × water cluster, each unordered pair once;
- **SW**, solute cluster × water molecule, every pair;
- **SS**, solute cluster × solute particle, each unordered pair once, with
  the excluded pairs cleared from the masks at the build as in D89.

The class-segmented pair list is one buffer of cluster pairs. Its three
class-pair segments are ranges of that buffer, compacted by tag,
with one offset per segment. The spatial search and sort run once. A warp
stays within one segment in the fused launch, so its tag branch is uniform.
Two mobility groups set the displacement-test cadences: water anchors,
and atoms. WW follows the anchors; SW and SS follow the faster participating
group. Each cadence keeps the outer and pruned references of D114.
The pruning of the solute segments is folded into the pruning in the loop
(D284), rather than the separate launch of the prototype.

For $P$ particle classes there are $P(P+1)/2$ unordered class-pair segments,
including self-pairs, but only $C\le P$ displacement tests, one per mobility
group, and one storage buffer. Four classes (water, solute, ions and
perturbed atoms) give ten segments and still two tests when only water
uses anchors. Segment $(a,b)$ refreshes at the faster participating cadence.
Segments without a measured specialization gain retain the general path
inside the same launch; a segment is not a separate kernel.

Only intramolecular pairs of a water are excluded, and two different
molecules have none, so the WW and SW work has no exclusion check: it is
removed structurally. The validity of each entry and the cutoff test of
each site pair remain.

### 5. One fused, warp-cooperative launch

The three classes run in one launch, a class per warp, each warp taking a
block of 64 entries as the groups do (`emitGroupPairKernel`):

- a lane holds one water of the warp's cluster: its three atom positions,
  its extra point formed once, and its force accumulators;
- the partner molecules' positions and accumulators rotate through the
  half-warps by shuffles, so a partner's sites, its extra point included,
  are formed once per entry and reused by the 16 molecules of the cluster;
- the partner's forces are accumulated in registers along the rotation and
  written once per entry, and the cluster's forces once per block, with one
  atomic addition per atom: fewer global atomics than one per site pair;
- the template's site pairs are unrolled with their charge products and
  Lennard-Jones coefficients as constants, and the pairs whose terms are
  zero are not emitted: for OPC 9 Coulomb and 1 Lennard-Jones term per
  molecule pair, against 16 general site pairs; for TIP3P 9 Coulomb and
  1 Lennard-Jones term.

SW lanes hold a solute particle and rotate molecules; SS lanes are the
general group kernel. The pair expression is cloned from the existing
scalar primary-pair body, with its actual tables and parameters. The
image-aware `md.sum_template_window` exposes the geometry and per-site
cutoffs; its first force and strain rules differentiate that arithmetic.
The physical specialization uses the class-segmented pair list (#774,
"Built from primitives").

The kernel reads the waters packed by molecule in f32, in the frames of
D95, and writes their forces back: a pack and an unpack each step, which
take the place of `main`'s gather into the order of the places. Keeping the
waters in a layout by molecule is #794.

### 6. The test of validity, fused into the integrator

As for the general structures (D80, `md-exec-expose-validity`), the tests
are reductions of booleans fused into the loop that writes the positions
that the forces will see, with no kernel and no wait of their own: the
anchors' test for the water list, the solute particles' and the anchors'
for the solute lists. The prototype read its test back to the host every
step; the implementation does not.

The covering bound also needs $\varrho_A \le r_\mathrm m$ at the current
positions. The proposal is to test it in the same fused loop, by the
thread that constrains the molecule, which holds its three atoms. For
$0\le\alpha\le1/2$, two squared O–H distances cover the formed site by
the triangle inequality. For larger weights the check includes that site's
squared extent as well, a distinction made at compile time. Before the first step no
constraint has run, since the positions of the input are not constrained
(as in sander): there the extents are measured at the build, by the
packing, which reads each molecule whole, and the water list takes
$\max(r_\mathrm m, \max_A\varrho_A)$ for that build. Between builds the
constraint step tests them. A molecule beyond $r_\mathrm m$ after a
constraint step stops the run, as the maintainer ruled below.

### 7. Selection: the execution choice `water.template`

An execution choice in the sense of #790, of its "same accuracy" class:

| Value | Meaning |
|---|---|
| `AUTO` (default) | The template where a declared rule says it pays, the general path elsewhere |
| `ON` | The template wherever it applies; a program where it cannot apply is refused, naming the reason |
| `OFF` | The general path; the program is that of `main` |

It applies on a GPU, in mixed precision, outside the deterministic mode,
to a program of dynamics with at least one matched water. `AUTO`'s rule is
declared with the choice and set by the implementation's end-to-end
measurements; `mdir tune` (#790) may pick per system. It must be able to
choose the general path where the template does not pay: stage 2 measured
ubiquitin with TIP3P at 1.015 times `main`'s step. The proposed first rule
is recorded in the maintainer's ruling below.

The record of the choice follows #790: `mdir check`, the log, and the run
manifest state the value, its source (`AUTO`, tune, or given), and what the
template took. A run continued from a checkpoint may change it, as the
other execution keys (D129, D172), with a note in the log. For example:

```
template kernels: water.template = AUTO -> on (4-site water, mixed, GPU)
  water OPC (4 sites, rigid, 6166 copies): 1 LJ and 9 Coulomb site pairs per pair of molecules,
    the extra point formed in the kernel; r_m 0.8734 A
  cluster pairs: WW, SW, SS in one launch; ions and solute in solute clusters
  not matched: 0
```

and, for a run where a model does not match: `not matched: 2400 TIP3 (LJ on
the hydrogens)`.

## What stays unchanged

- **The deterministic mode** keeps the general path; so do double
  precision and the CPU target.
- **`OFF`, and every program without a matched water**, give the program of
  `main`: `mdir emit` of its stages is byte-identical, and the corpus and
  the checkpoints are unchanged.
- The definition of the extra point in the topology, its placement and
  spreading kernels for the stored particle (the reciprocal sum and the
  other terms read it), the reciprocal sum, the bonded terms, the
  constraints, and the integrators. Forming the extra point in the PME
  spread and the gather is #738, which takes this decision's rule so that
  the two agree.
- The neighbor structures of the general path, D261's choice between them,
  and the skins of D281 for the solute lists.

**Not bitwise equal to `main`.** With the template the step differs from
`main`'s in the order and the rounding of its sums, within the error of
mixed precision: the extra point's placement (linear with a constant
weight, against the bisector's normalizations) and its spreading rule;
the order of the accumulation (along the rotation, one atomic addition per
atom, in another order of the clusters); the order of the f32 operations
(charge products folded as constants, site displacements formed from the
molecule's frame); and the layout of the data (packing by molecule, a
different sort and different lists). This is one reason the deterministic
mode keeps the general path; the summary and the docs say so.

## Measurements of the prototype

Stage 2 of #774: a prototype of the whole short-range path outside the
repository, RTX 4090, mixed precision, against `main` with the groups
(12 / 9.6 Å) on the same device, oxygen anchor. The whole-step figures
are composed: the prototype's path plus `main`'s kernels that stay plus
`main`'s gap between wall time and kernel sum. The implementation's own
measurements replace them.

| System | Composed step (µs) | `main`, same water (µs) | Ratio |
|---|---|---|---|
| ubiquitin, OPC, 0.15 M KCl, two lists, one launch | 166.3 | 183.3 | 0.907 |
| ubiquitin, OPC, one list, one launch | 169.5 | 183.3 | 0.925 |
| ubiquitin, OPC, two lists, three launches | 186.2 | 183.3 | 1.015 |
| OPC water, 16,329 waters | 227.7 | 293.3 | 0.776 |
| OPC water, 4,200 waters | 116.7 | 140.0 | 0.833 |
| TIP3P water, 16,329 waters | | | 0.920 |
| TIP3P water, 4,200 waters | | | 0.898 |
| ubiquitin, TIP3P, two lists, one launch | | | 1.015 |

- **The launch.** For ubiquitin with OPC the short-range path took 68.0 µs a
  step in one launch and 87.8 µs in three; launched apart, the
  solute–water (16.1 µs) and solute–solute (9.8 µs) kernels were bound by
  latency, while one launch took 44.8 µs for all three classes against
  40.0 µs for water–water alone.
- **The lists.** The water lists were built about every 55 steps and pruned
  every 8.5; the solute lists built every 38 to 43 and pruned every 3;
  `main`'s groups build every 20.7 steps and prune every 2.8.
- **Where the saving comes from.** In the water boxes (stage 1) about 60% of
  the saving was the pair kernel and 40% the list. That split holds for the
  water boxes only: in the protein the solute lists and the pack and unpack
  change it, and for ubiquitin with TIP3P they used up the saving.
- **The remainder.** `main` runs OPC at 1.25 times TIP3P's step for
  ubiquitin. With the template the remainder is 19.6 µs. Per-kernel sums
  put about 12 µs of the difference in the kernels outside the short-range
  path in the extra point's placement and spreading, and less than 1 µs in
  PME, since D262 already leaves the oxygen out of the spread (#774,
  remainder analysis; the implementation confirms the assignment). That is
  #738's work.
- **Accuracy.** Against an f64 brute force over every non-excluded site
  pair within $r_c$ with the bisector extra point and its exact Jacobian
  (itself within $6\times10^{-10}$ kJ/mol/nm of `main` in double for TIP3P,
  and $5.9\times10^{-6}$ for OPC with the constraint directions projected
  out, #789), with forces projected the same way:

  | System | Largest reference force | Max / RMS error | Energy (relative) |
  |---|---|---|---|
  | ubiquitin, OPC | 3998.8 | 0.034 / 0.0022 | $1.0\times10^{-7}$ |
  | ubiquitin, TIP3P | 2582.6 | 0.020 / 0.0015 | $7.2\times10^{-8}$ |
  | TIP3P water, large | 3064.2 | 0.035 / 0.0025 | $3.3\times10^{-8}$ |
  | TIP3P water, small | 2759.4 | 0.015 / 0.0014 | $2.4\times10^{-8}$ |

  Forces in kJ/mol/nm. Every case is within `main` mixed's own error on the
  total forces, the criterion of step 0. The one launch agreed with the
  separate kernels to $10^{-3}$, the order of the atomic sums.
- **Coverage.** 50 checks on 300-frame runs, just before a pruning, just
  before a build, and just after one, for both kinds of list and both list
  modes: no needed pair missing, none duplicated, no excluded pair in a
  solute–solute list.

## Gates of the implementation

1. **Forces, energies, and the virial**, on ubiquitin with OPC and with
   TIP3P and on #738's water boxes, against an f64 oracle and against
   `main` in double on the same frame, within `main` mixed's own error
   (step 0's criterion). With rule (A) the forces are compared with the
   constraint directions of each water projected out (#789), and the
   virial is compared as the total with the constraint virial; with rule
   (B) the raw atom forces are compared as well.
2. **Coverage**, just before a pruning, just before a build, and just after
   a build, for every list, against an f64 periodic brute force, including
   runs under a barostat (the prototype's runs were NVE, with no scaling
   of the cell).
3. **Unchanged programs.** For `OFF` and for programs without a matched
   water, `mdir emit` byte-identical to `main`, the corpus and the
   checkpoints unchanged.
4. **End to end.** Whole runs with everything in (integration, builds and
   prunings, PME, the virtual sites) on ubiquitin with OPC and with TIP3P,
   alternated with `main` in one job, steady-state wall time, on one kind
   of device.
5. **Stability over many steps.** NVE energy drift against `main` at the
   same settings; the constraint residuals; the virial and the pressure;
   NPT run and reported separately from NVT and NVE.
6. **Timings** against `main` with the groups on one kind of device:
   #738's boxes, ubiquitin, and the Amber suite, where no system may be
   slower under `AUTO`.

## Maintainer's ruling

All seven recommendations were accepted on #795:

1. Constant-transpose spreading (A) for dynamics; deterministic mode keeps
   the general path.
2. The validity test is fused into the integrator, with solute pruning in
   the existing pruning loop.
3. `[performance]` `"water.template" = "AUTO"`, `"ON"` or `"OFF"`, in the
   same-accuracy class. `ON` is refused on the CPU, in double precision or
   deterministic mode.
4. Initially `AUTO` selects four-site models only; end-to-end measurements
   set the particle-count floor. Three-site models remain general under
   `AUTO` until measurements justify selecting them.
5. The margin is $10^{-3}$ Å. A first-evaluation extent excess widens the
   reach; an excess after a constraint step stops the run.
6. Written forces use the bisector transpose for that evaluation.
   Derivative paths remain general and refuse `ON` by name. Tunable water
   parameters likewise remain general; runtime template parameters and
   their derivatives are an unscheduled follow-up to #774.
7. The first-version scope is the oxygen-only-LJ three- and four-site
   models above, one predominant template per program, with reported
   mismatch counts.

## Left to the implementation

The initial structural domain, `md.template_pairs`, partitions WW, SW and
SS without hiding continuous data. The native window arithmetic forms
each site from an explicit anchor image and whole-molecule offsets; its
constant transpose is derived by the scalar derivative rules. `ON` now
constructs these windows through the native nonbonded adapter and reuses
the scalar primary-pair expressions. GPU list lowering remains in
implementation. `AUTO` stays general until the required end-to-end runs
establish its particle-count floor.

The maintainer authorized the dedicated image-aware reduction on #795.
Ordinary `md.sum_tuples` takes minimum-image coordinates independently, so a window
formed from its single oxygen displacement cannot stand for every anchor
image in the covering reach. Selecting just that image could omit a site
pair across a periodic boundary. The dedicated window reduction enumerates
the covering images, exposes the site arithmetic and site cutoffs, and
derives forces and strain from those explicit displacements. Its verifier
proves each yielded site displacement is the anchor displacement plus
supported affine offsets. The cutoff emitter guards the pair arithmetic
with actual control flow.
Its new derivative admission is limited to that reduction; existing
classifications and refusals remain unchanged. No restriction to half-cell
covering reaches is introduced. Before performance measurements, the
isolated f64 reduction is checked against an independent all-images oracle.
For the quadratic boundary probe, energy, every real-atom force component
and every virial component must agree within
$2\times10^{-13}\max(1,\lvert\mathrm{reference}\rvert)$. This is a
roundoff check of the reduction, not a replacement for the mixed-precision
physical-system bounds above. The probe uses a 2.2 nm edge, 1 nm site cutoff
and 0.1 nm water extent, and checks the neighboring-image pair explicitly.
That isolated quadratic probe passed for all 27 images and their totals:
energy 0.405, virial xx -0.81, with only the neighboring image contributing.
This does not validate the physical pair kernel or the GPU list lowering.

The maintainer admitted every finite nonnegative $\alpha$ on #795, with
no upper bound of $1/2$. The topology matcher admits, for example,
$r_\mathrm{OH}=0.1$ nm, $r_\mathrm{HH}=0.16$ nm and
$r_{os}=0.09$ nm, which give $\alpha=0.75$. The formed site's extent is
tested wherever its bound cannot be inferred from the two O–H extents.
For $\alpha\le 1/2$, the triangle inequality proves that the two O–H
extents cover it, so those models keep the two-distance check. This
distinction is made at compile time. Exact dyadic extent probes have
predeclared squared extents $1/64$ for $\alpha=1/2$, and $9/256$ or
$9/512$ for aligned or perpendicular O–H offsets at $\alpha=3/4$;
the offsets have length $1/8$. These isolated f64 checks require exact
agreement and do not replace the physical-system numerical bounds.

The execution representation now gives the three classes one
`!mdrt.segmented_pairs` value. `md_exec.build_segmented_pairs` declares its
membership and bounds; `md_exec.template_for` holds three tagged scalar
bodies, with per-output site-pair counts. Conversion shares the list
between energy, force and strain reductions. Buffer assignment initializes
scatter destinations and preserves explicit memory effects. Mixed
precision assignment keeps source positions in f64, class coordinates and
pair arithmetic in f32, forces in f32, and energy/strain reductions in f64.
These execution-IR checks do not establish persistent-list or device
lowering correctness.

Compatible template loops now fuse their three tagged bodies, preserving
destination and reduction order. Template-list reuse makes initialization
separate from refresh: the state is carried through nested step/output
loops, while segments and fresh step loops reset it. Refresh declares two
mobility-group references and outer/inner margins; its physical device
implementation is still pending. Focused checks cover those reset
boundaries and the combined fusion, precision and storage pipeline.

The isolated packing emitter adds the lattice shift to whole-molecule
positions before narrowing, without loading a stored extra point. Its
exact dyadic probe uses a large common translation that would erase local
geometry if narrowed first. The warp scheduler rotates a partner's frame,
parameters, mask and force accumulator together through 16 half-warp
turns, then writes each partner once. It combines the two copies of each
first unit before writing. Its structural tests cover WW, SW, SS and
energy-only bodies; physical GPU validation remains outstanding.

The relation kind of template instances and its specialization in the
builder; the packing and its frames; the segmented dual-list buffer and
the remap of SW entries after a water build; the fused kernel and its register budget;
the extent check; the record of the choice; the end-to-end measurements
that set `AUTO`'s rule; Section 10's numbers. Related work:
[[Ruymgaart2012]](../references.md#ruymgaart2012) (water-specific kernels and
water-index lists; SI Appendix C applies an oxygen–oxygen molecular cutoff
to all nine site pairs, a different model from this decision's site cutoff;
no virtual sites or generated code) and
[[Nikolskiy2020]](../references.md#nikolskiy2020) (TIP4P on a GPU: the
virtual charge placed once per molecule and stored, its force spread by a
second kernel).
