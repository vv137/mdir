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

## Glossary

These terms apply to this decision, its documentation and diagnostics.
Implementation identifiers retain their spelling but describe the term
they implement.

| Term | Meaning | Implementation names |
|---|---|---|
| class-segmented pair list (segmented list) | One buffer of cluster-pair records, divided by class-pair tag into segments. | shared arena, pair allocation |
| segment | The WW, SW or SS part of that buffer, with its outer and inner ranges. | WW/SW/SS ranges |
| class | Water template or solute; selects the computation for a unit. | class, class row |
| class pair | WW, SW or SS. | three-class |
| unit | A matched water with anchor O and radius $r_\mathrm m$, or a solute particle with radius zero in this version. | real unit, padded unit |
| anchor | A unit's reference point for sorting and displacement tests. | anchor |
| mobility group | Units sharing a displacement test and refresh schedule: anchors of matched waters; atoms. | group, stale group |
| refresh | Rebuild or prune of the segments invalidated by a mobility group's test. | stale-group refresh, cold refresh |
| template kernel | The fused launch evaluating all three class pairs. | three-class loop, fused window |
| extent | Largest distance of a unit's sites from its anchor, checked against $r_\mathrm m$. | extent, initial extent widening |
| covering bound | The anchor-distance conditions $R+2r_\mathrm m$ for WW and $R+r_\mathrm m$ for SW. | covering predicate |
| rule (A) | The extra point's constant transpose used by template dynamics. | constant site transpose |
| rule (B) | The true bisector transpose used for written-frame forces. | general frame evaluator |

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

The failure-only native checkpoint acquisition exposed an initialization contract error after the native
spatial sort. Device tuple renumbering leaves an original-file host
placeholder and registers the actual device members separately. Template
initialization must resolve that registered buffer before acquiring both
the water members and the exclusions. The correction is local to this new
initializer; ordinary tuple renumbering and incidence construction retain
their existing contract. The temporary failure-only instrumentation was
removed after acquisition.

The short fixtures and preliminary water boxes did not establish this
contract: a sort of homogeneous water can permute complete, identical
OHH/EP blocks while preserving the unlabeled block partition and its
energy. The corpus and order matrix compare emitted programs, rather
than executing the metadata handoff. The regression therefore interposes
solute slots between spatially sorted water blocks and checks exact
members, packed slots and exclusions, with identity and nonidentity
orders. Its CPU check follows the initializer's copy back to the
registered renumbered buffer in device-tuples mode and to actual host
renumbering in the host configuration.

Performance diagnosis counts successful full template refreshes, fused
prune events and automatic arena growths separately from the general
neighbor structures. The log reports mean intervals as the number of
requested steps divided by the event count minus the initial event,
and outer-entry, inner-entry, work-record and capacity occupancy at the
first successful build and the end. Occupancy acquisition waits and reads
the device only at those two diagnostic boundaries; it adds no per-step
readback. NVTX marks delimit host force-evaluation intervals without a
device wait, so a timeline can correlate their launches and copies.
These diagnostics change no physical arithmetic, validity predicate,
cutoff, list policy or comparison bound. The performance cause is reported
before optimization, separately from the remaining correctness gates.

The radial extension uses `PairKernelBodyOpInterface`, implemented by
`md_exec.pair_for` and `md_exec.template_for`. Each view names the squared
distance, its kernel region and its contribution outputs; structured
windows can expose several site-pair views of one region. Both
`md-exec-simplify-distance` and the parent matching in
`md-exec-expand-radial` consume that interface. They retain D94's admission,
error criterion and analytic fallback. A distance computed inside a region
is the radial function's input boundary, and its derived values stay after
its definition. The argument-distance case retains its original placement.
This extension follows differentiation and changes no launch or group
layout. The ordinary corpus and order matrix verify the original pair path.

At implementation checkpoint `f0151e73`, the ordinary corpus checks 1,050
cases: 1,046 are identical, including 454 identical error cases, and the
same four previously accepted keyword-hint diagnostics differ. Its entire
comparison log is byte-identical to the accepted pre-extension log; there
is no new generated-program or diagnostic difference. The order matrix
passes 560/560 cases across its exact, padded and search modes. The new
common-interface CPU fixture checks multiple distances in one region,
all three classes, admitted tables and the unchanged analytic fallback;
it and the four existing radial fixtures pass.

Seven focused RTX 5090 tests pass with the current bounds: the original
zero-tolerance scheduler; independent f64 screened-Coulomb energy, analytic
forces and strain for the differentiated body and radial table; OPC,
TIP3P and solute physical comparisons; and the molecular-virial fixtures.
The screened case uses `erfc(3.5*r)/r`, rotated affine sites, 243 active
pairs and 469 checks. It uses the current physical gate's scale floors
without its additional baseline/cancellation allowances, and checks
untouched padding exactly. It is not an independent full-PME oracle.
The original quadratic fixture retains zero tolerance.

1. **Forces, energies, and the virial**, on ubiquitin with OPC and with
   TIP3P and on #738's water boxes, against an f64 oracle and against
   `main` in double on the same frame, within `main` mixed's own error
   (step 0's criterion). With rule (A) the forces are compared with the
   constraint directions of each water projected out (#789), and the
   virial is compared in D264's molecular convention, with the atomic
   total including constraint reaction retained as a diagnostic; with
   rule (B) the raw atom forces are compared as well.
   The independent native-topology oracles include
   `include/mdir/Support/PhysicalConstants.h` and use its
   `mdir::physical::coulomb`, the same physical input as the compiled
   native route (D394, #805 part 1). Its SI/CODATA 2022 derivation and
   compiler-rounded values are specified in
   [conventions, Section 1](../conventions.md#1-units). Pair/image
   enumeration, force arithmetic and the molecular virial remain
   independently implemented. A separately rounded coefficient can
   dominate a nearly canceled virial's net-relative bound; no comparison
   bound changes when the fixture consumes the public header.
   Fresh comparisons record signed oracle/general-double values and
   exact bits, evaluated bounds, absolute contribution sums and term
   counts before assertions, including every force component.
   Potential energy uses the cancellation-aware bound
   $\lvert E_\mathrm{template}-E_\mathrm{double}\rvert \le
   \max(10^{-5}\lvert E_\mathrm{double}\rvert,10^{-6}S)$ for OPC and
   TIP3P. Here $S=\sum_p(\lvert E_{\mathrm{LJ},p}\rvert+
   \lvert E_{\mathrm{Coulomb},p}\rvert)$ comes from the independent CPU
   oracle at the same frame, summing the two contributions' magnitudes
   separately. Heavy cancellation can make the net energy a small
   remainder of large terms; rounding then scales with those terms.
   The $10^{-6}S$ allowance is about eight f32 epsilons of their magnitude,
   while the first term retains the previous relative allowance. Force
   bounds and the oracle/general-double bounds remain unchanged. The
   physical fixtures retain both absolute pair sums in their output.
   The scalar virial is compared directly in D264's molecular convention:
   $\lvert W_\mathrm{template}-W_\mathrm{double}\rvert \le
   \max(1.5\lvert W_\mathrm{mixed}-W_\mathrm{double}\rvert,
   10^{-5}\lvert W_\mathrm{double}\rvert,10^{-6}S_W)$, where the CPU
   oracle at the same frame computes $S_W=\sum\lvert\mathbf r\cdot
   \mathbf F\rvert$ over the contributions to the molecular virial:
   pair, site-transfer and removal of the internal force moment about
   each rigid group's mass center. The independent-oracle/general-double bound is
   $10^{-10}\max(1,\lvert W_\mathrm{oracle}\rvert)$. The extra term
   accounts for cancellation among large positive and negative virial
   contributions; the mixed reference error and relative term remain in
   the maximum. Record all three reported molecular virials, the oracle's
   pair, site-transfer and internal-force-removal traces, and the sum of
   their absolute contributions.
   The instantaneous physical check uses fresh no-step entries at the
   same constrained positions and full-step velocities for velocity
   Verlet. Its molecular oracle computes the pair and site-transfer
   virial with the internal force moment removed. Keep the atomic total,
   full constraint reaction and $2K_\mathrm{internal}$ as diagnostics:
   $W_\mathrm{atomic}=W_\mathrm{molecular}-2K_\mathrm{internal}$.
   The diagnostic constraint force solves the mass-weighted holonomic
   system with its velocity-curvature term and oxygen-relative force
   arms. D264 pairs the molecular virial with its group-center half-step
   kinetic estimator; the three reported virials are used directly,
   without velocity-dependent adjustment. The step's averaged impulse
   virial, trajectory behavior, NPT virial and pressure remain separate
   validation gates.
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
   Include short template-versus-general trajectories for both velocity
   Verlet and leapfrog in NVE and in NPT with the barostat, with the same
   acceptance criteria. Leapfrog couples $\mathbf v_{n+1}$ in NPT.
   The approved short comparison is 100 steps at 1 fs, sampled every
   10 steps, for OPC/TIP3P, NVE/NPT and both integrators from identical
   constrained states and seeds. Position, velocity and volume bounds
   use $\max(3\epsilon_\mathrm{mixed},10^{-5}s)$, with the general
   mixed/general-double difference as baseline. The scales are the
   largest cell edge, $\max(1\ \mathrm{nm/ps},\max_i\lvert v_i\rvert)$,
   and the general-double volume. Fresh same-frame physical evaluations
   keep the force, energy and virial bounds above unchanged.
   Production NVE drift uses ubiquitin with OPC and TIP3P, mixed, 2 fs,
   1 ns, template versus general mixed for each integrator. Fit conserved
   energy versus time and divide its slope by the degrees of freedom;
   require $\lvert d_\mathrm t\rvert\le
   \max(2\lvert d_\mathrm{mixed}\rvert,5\times10^{-4})$ in
   kJ/mol/ns per degree of freedom. Report energy fluctuation and maximum
   relative squared-bond residual. Report NPT mean and fluctuation of
   volume and pressure separately, without instantaneous or atomic
   equality along different trajectories.
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
mobility-group references and outer/inner margins, now connected to the
native device owner. Focused checks cover those reset
boundaries and the combined fusion, precision and storage pipeline.

The isolated packing emitter adds the lattice shift to whole-molecule
positions before narrowing, without loading a stored extra point. Its
exact dyadic probe uses a large common translation that would erase local
geometry if narrowed first. The warp scheduler rotates a partner's frame,
parameters, mask and force accumulator together through 16 half-warp
turns, then writes each partner once. It combines the two copies of each
first unit before writing. Its structural tests cover WW, SW, SS and
energy-only bodies; physical GPU validation remains outstanding.

The scheduler prepares local O–H offsets once for the first unit, and
once per loaded partner before its 16-turn rotation. A four-site model
also prepares the formed-site offset there and rotates it with the
partner. Exact typed `alpha*(H1+H2)` source expressions bind to those
cached offsets; no alternate placement or force formula is introduced.
Three-site models emit no formed-site products. Structural checks verify
the products lie outside the rotation. An independent $\alpha=0.75$
all-images f64 probe checks the cached energy, constant-transpose forces
and strain at the same bound above; the transpose coefficients are
$(-0.5, 0.75, 0.75)$. Its expected neighboring-image energy is 0.492032
and virial xx is -0.984064.

The physical allocation has one flat integer payload containing the outer
and inner rows, work units and pair records for all classes. The shared
compiler/runtime layout uses i64 physical offsets and bounded i32 payload
indices. Initialization uploads real-unit membership once: stored extra
points belong to water but never enter the atom complement. A growth
discards every class range, so the subsequent build must rebuild all
classes. No growth work occurs on a refresh that fits.

Fixed topology metadata occupies a separate immutable slab: real-unit
membership, the original-particle-to-packed-slot map, and a sorted
symmetric CSR graph for solute exclusions. Duplicate and reversed edges
are collapsed; water exclusions never enter that graph. Initialization
accepts host or already-buffered device topology, with any device-to-host
copy confined to initialization. Borrowed metadata views survive pair
arena growth. An exact integer GPU fixture checks all four views before
and after growth and through both initialization paths; it does not
validate spatial search or refresh.

Whole-frame preparation recovers each hydrogen's image relative to its
oxygen in f64, shifts the entire frame into the oxygen's central cell in
f64, and then narrows it. It uses the small intramolecular covering bound,
not a minimum-image substitute for the enumerated molecule-pair images.
Exact dyadic fixtures cover both cell representations, independently
wrapped hydrogens, face crossings, and large unwrapped coordinates.

The physical work-record adapter reads the compact unit order, real-site
particle mapping and packed frames, decodes the partner's lattice image,
and dispatches the three source bodies through a warp-uniform class tag.
Input fields follow the same real-site mapping; force atomics target the
original particles. An isolated seeded-record GPU fixture uses the
rotated-frame oracle below, with exact equality for all 469 values. It
tests record consumption, not list generation or validity.

Production GPU lowering uses activation-owned compact-order and packed-frame
views that survive pair-arena growth. It launches at most one warp per
original particle and strides over the device-resident work count, so its
sum scratch retains its fixed particle capacity and no work-count readback
is needed. The three classes share that launch and the existing global-sum
reducer. A seeded one-pair runtime fixture checks the complete allocation,
view, launch and reduction path against the exact energy $1/4$; search and
validity are still separate gates.

The same arena also contains build staging, spatial keys, sorted unit IDs,
inverse places, bin counts/offsets/cursors, and shared scan scratch. A spatial
search can emit chunks once, then compact them by fixed row/class order
without repeating its candidate search or allocating a second pair list.
The bin budget is at least twelve and otherwise four times the number of
64-unit groups. Scan scratch covers both fixed rows and bins; all borrowed
arena views must be reacquired after growth.
Physical staging addresses use i64 even when a chunk's index fits i32.
The compactor uses the existing device scans and initializes the inner
view with the full outer blocks; pruning subsequently shortens those
blocks in the fused loop. An integer oracle independently concatenates
rows for all three classes, including empty rows and a 65-entry row, with
staging chunks supplied out of order. Its 1,182 exact checks include
unchanged capacity tails. A separate seeded pair passes through compaction
and production dispatch to the exact energy $1/4$. These are isolated
compaction fixtures; the combined native checks are recorded below.

The shared spatial-order emitter uses class-prefixed x-y serpentine
columns, then z keys with fixed-unit-ID ties. Reference fractions wrap
in f64 before narrowing to f32 for key arithmetic, as the legacy mixed group ordering
does. One stable counting sort orders the joint (class-column, z) key.
It starts with ascending unit IDs and composes the shared histogram,
scan, cursor scatter and `emitRankInBins` primitives over successive
least-significant digits. The scatter carries input positions; ranking
those positions restores their prior order within a digit, preserving
lower digits and fixed-ID ties. Digit widths balance the significant key
bits across passes, with at most 256 buckets and no more buckets than
the existing spatial-bin storage. Four temporary i32 words per unit
use independent planning scratch before search; no allocation or persistent
order of an inactive class changes. The original column counts and
offsets are republished after sorting for the unchanged search.
Each class is cut at column boundaries, adjacent gaps and refined spans
through `emitSpatialGroupPlan` (device-primitives Section 4.12). Defaults
are water/solute gaps 0.4/0.3 nm and spans 0.9/0.6 nm, quantized against
the z face width. Groups have at most sixteen real units. The compact
order and descriptor partitions have fixed class capacities. Rebuilding
one mobility group leaves the other's order, lengths and inverse map
untouched; the common grid stays fixed while a nonempty class is retained.
Four-word work records keep their first-unit-begin meaning and read the
explicit length from a separate view. Row-capacity growth preserves old
completed lists and new spatial data. Inactive class rows use the previous
header's class prefix when copied into their possibly moved new prefix.
An independent lexicographic oracle checks 900 integer values over all
four group-activity combinations, including repeated-point ties and
coordinates that round to the final bin edge.
The larger 516-unit variant also crosses launch blocks and exercises a
different digit decomposition with many equal full keys.
The search forms boxes in the wrapped key frame through the shared
`SpatialFrameEmitter.h` helpers. An at-most-sixteen-unit group queries each
of its distinct lattice-frame subsets once, retaining disjoint masks.
Query images add the first unit's lattice frame before the original f64
subtraction and compensate the partner's frame in the stored image code.
This tightens boxes at the reference-frame seam without changing the pair
predicate or image coverage. All 32 lanes reach each masked box
reduction; a queried subset always contains its representative. The
all-images oracle also covers groups spanning eight distinct frames in
orthorhombic and triclinic cells. The ordinary group builder is unchanged;
its eventual use of these shared helpers belongs to #817.
Before unit enumeration, a shared conservative Cartesian box test
rejects whole partner groups whose wrapped-reference boxes cannot reach
the first-frame subset. A ballot and warp-uniform survivor loop select
up to two groups for enumeration. The previous per-unit-lane box check
is removed. Boxes and distance bounds remain f64; Cartesian and fractional
auxiliary views survive pair-arena growth. The shared single-image helper
admits only the proved integer image when the expanded fractional
intervals each have width strictly below one, retaining explicit images
otherwise (device-primitives Sections 4.11–4.12). Conservative roundoff
allowances admit uncertain cases before the unchanged exact f64 pair test.
All 32 lanes reach every first-unit broadcast in the uniform loop.
The template runtime ABI is 7; ordinary builders and physical bounds
are unchanged. The integer helper fixture covers an eleven-pass cascade
and multiple scan blocks; native correctness and performance are separate
gates, pending for this integration checkpoint.
The native cold refresh owns these bins and their combined search.
Its column dimensions use the population of water anchors plus solute
atoms and the current cell volume: the target edge is
$(16V/(N_\mathrm{w}+N_\mathrm{s}))^{1/3}$. Face widths divided by this
edge are rounded to the nearest positive integer. The dimensions are
clamped before integer conversion so that the two class-prefixed grids
fit the existing spatial-bin arena; empty populations use one unit for
this sizing arithmetic. This is f64 host-only cold-refresh arithmetic.
It replaces the constant three-by-two native grid without changing the
class keys, fixed-ID ties, covering search or physical predicates.

The search's append primitive ballot-compacts sparse candidate batches
into chunks of 64, carrying a partial chunk across rounds. Concurrent
rows reserve chunks with an i64 counter; physical addresses remain wide.
Capacity overflow skips writes beyond the logical capacity while retaining
the exact required chunk/entry counts for growth. An entry total outside
the final i32 domain sets an index flag before compaction. Independent
row-concatenation fixtures check the ordinary path (1,190 integers), the
expected capacity flag and untouched tails (1,574), and a preloaded total
crossing the index bound (14). These test the primitive, not a completed
production refresh.

The combined bin-search emitter visits covering images in one kernel over
the fixed class rows. It searches active rows once and copies retained
outer records for inactive rows through the same chunk writer. WW and SS
use unique-pair ownership; only SS consults the fixed exclusion graph.
Outward column/z-key guards cover f32 key rounding, while the covering
distance predicate uses f64 reference geometry. Its independent oracle
enumerates all pairs and all covering images, without bins or chunks.
Both cells pass all four activity phases with zero missing pairs, extra
pairs or duplicates, including partial clusters and explicit images at a
reach beyond half an edge:

| Cell | Both rebuilt, WW/SW/SS | Atoms rebuilt | Waters rebuilt | Neither rebuilt |
|---|---|---|---|---|
| Orthorhombic | 86/132/25 | 86/98/25 | 86/144/25 | 86/144/25 |
| Reduced triclinic | 86/108/32 | 86/99/32 | 86/169/32 | 86/169/32 |

These are 1,979 exact unit/image pair memberships over eight phases, with
predeclared reaches 135/128, 117/128 and 99/128. They validate the combined
sort/search/staging/compaction emitters. Native refresh ownership, extent
handling, displacement validity, fused pruning and physical dynamics
remain separate gates.

Reference points may retain independent nearby whole-cell frames. Only
the bin keys wrap; the search compensates each partner's integer frame
in its encoded image, preserving the actual packed geometry. The image
payload is checked before writing. Independent-frame fixtures pass both
cells, including retained inactive rows, with zero missing, extra or
duplicate unit/image memberships. These checks do not recanonicalize
the per-step packed frame or change the source site cutoff.

A complete-frame particle-writer emitter evaluates the original scalar
body for O/H/H under one thread, loading all three members before storing
any output. Its callback receives the final values for packing and
validity without a cross-thread position read. Unmatched atoms and stored
extra points retain their scalar updates. The native owner selects the
last position writer before a refresh. Its first device validation reported 323 of 1,059 values
different from the exact dyadic oracle, with every decimal value equal
to that oracle rounded to six significant digits. Work stopped, and the
maintainer authorized a fixture-only change to exact-bit serialization
and one controlled comparison. That comparison passed: all 966 f64 values
are bitwise identical and all 93 integer visit counts equal one. The
bound remains zero, including the signs of floating zeros; neither the
writer arithmetic nor ordinary lowering changed. The other new exact or
tight f64 fixtures also serialize diagnostics as integer bit patterns,
with their original in-program comparisons and bounds unchanged.

The cold-refresh orchestrator now joins shared spatial ordering, search,
capacity handling and compaction. It reads controls only during an actual
build, before compaction, and sizes a required growth from the exact
entry total. Growth reacquires all arena views and prepares both mobility
groups before rebuilding all classes; another overflow is fatal. Two
independent coverage fixtures start undersized and require automatic
growth, then exercise both single-group refreshes and retained inactive
rows. Both cell types match all 1,979 expected unit/image memberships
with zero missing pairs, extras or duplicates. The native refresh consumes
the fused writer's group flags and prepares only stale outer epochs;
this is not a per-step rebuild policy.

Packing-cell coordinates are integer-valued f64 values chosen at the
outer build. The retained-frame emitter reconstructs O/H/H and applies
those coordinates to the current cell before narrowing. A central-image
face crossing therefore keeps the frame of existing entry images, and a
cell change scales the shift rather than reusing its old Cartesian value.
Four independent dyadic CPU cases check all 72 packed, fractional, shift
and cached-cell components exactly, with independently imaged hydrogens.

Outer/inner reference fractions, retained packing cells, current points,
the three class cells for both epochs, and their actual build reaches
have fixed f64 views independent of pair-arena capacity. The stored reach
can retain the initial-input extent widening in the validity theorem.
An isolated runtime fixture checks all seven views before and after pair
growth: 180 bitwise f64 comparisons, the dimensions and a stable-address
check. Native epoch publication is also checked below; complete physical
dynamics remains a separate gate.

The cold snapshot emitter chooses packing cells and updates outer/current
fractions only for requested mobility groups. It reconstructs independent
hydrogen images before packing real sites; stored extra-point coordinates
are never read. An unsigned maximum of nonnegative f64 extent bits records
the measured squared extent, including the formed site where required.
The native owner uses that measurement for initial widening or a
post-constraint check.
An independent rational-geometry oracle checks four activity phases for
three-site and four-site water in both cell types. All 1,264 scalar bit
patterns match exactly, including 3/4-weight formed-site extents, retained
inactive views and new packing frames at an outer-build face crossing.

The native template step marks its matched-water position-constraint
operations with reserved provenance, `md.template_constraint_epoch`.
Conversion, tuple fusion, destination merging and storage preserve it;
ordinary gathers and ordinary native programs receive no marker. This
gives the runtime policy an explicit constraint event instead of
inferring one from a tuple name. Function-level host history observes
those events and is not reset by a fresh list.

The validity emitters use a conservative lower bound on Cartesian cell
deformation: the least edge ratio for rectangular cells, and the inverse
map's induced one/infinity-norm bound for tilted cells, with rounding
guards and exact identity handling. Each mobility group's half-margin is
the minimum over its incident classes. Each class margin is capped at its
fresh-build margin, so resetting SW's cell/reach cannot reduce an inactive
group's allowance, including during expansion. The build and validity
share the same actual per-class reach values and addition order.

Ten analytical cell cases pass 20 interval predicates: identity is exact,
and other results stay below the mathematical norm bound and within
1e-11 of it. Four analytical group cases and two partial-SW resets pass
20 more predicates; ten exact boolean displacement cases include negative
and zero margins, both cells and an unwrapped whole-cell jump that must
invalidate. Projection and squared-distance arithmetic have conservative
rounding guards. These are validity-primitive checks, not physical force
tolerances or physical dynamics. The explicit-reach cold search
also matches all 1,979 expected memberships in both cells exactly.

The final-SSA writer callback now joins retained-frame packing, current
fraction publication and both displacement tests inside its caller's
particle kernel. It loads no world-position buffer. Unbuilt lists skip
the epoch views, while constraint history still enforces the extent bound
even on a fresh list. Its boolean results are supplied to the writer's
flag reduction; the callback adds no launch, synchronization or readback.
An isolated dyadic fixture passes 736 exact comparisons across both cell
types, including independent hydrogen images, retained packing cells,
outer/inner motion in both groups, untouched unbuilt state and the
formed-site extent at alpha = 0.75. Geometry is serialized as integer
bits from the start. The native owner connects this callback and shares
one mapped-flag event with preceding constraint flags whose results are
first consumed after the writer. The ordinary writer path is unchanged.
This is not yet a physical-dynamics gate.

The fixed-block pruning emitter intersects existing I masks with the
current f64 covering predicate, compacts surviving entries within each
outer block and publishes its inner live end. All 32 lanes participate
in a warp barrier before evaluation can read another lane's stores. An
inactive class retains its prior inner block. Requested groups read
current points, and other groups read retained inner references, allowing
the owner to publish new references in the evaluation prologue without
a cross-warp read/write race. Outer records and storage ranges remain
immutable; inner work records carry their actual live ends.
An independent rational all-images oracle passes 16,096 exact integer
comparisons in both cell types and all four group-activity combinations.
It checks sparse masks, two-batch blocks, empty and partial blocks,
inactive retention, immutable outer records, stable compaction and
immediate same-kernel consumption. The tilted-cell case also passes
memcheck, initcheck, racecheck and synccheck with zero errors or hazards;
the focused sanitizer run held the device for five seconds. The native
force-loop prologue now publishes requested inner references, cells and
reaches; its work loop prunes requested classes before evaluating them.

The integrated native fixture joins cold refresh, capacity growth, the
actual scalar particle writer, both validity epochs, fused pruning and
source-derived formed-site energy. Four unconstrained waters have extent
3/4 against the nominal 1/5: the nominal covering radius would miss an
interacting pair, so the first evaluation requires widening. It also
asserts automatic arena growth. Subsequent evaluations translate one
water, wind it by a whole lattice vector, double the cell and restore it.
An independent rational all-images oracle passes all fourteen energies
bitwise in the two cell representations. The final class cells and actual
reaches add 84 exact components, also serialized as integer bit patterns.

The first native sanitizer fixture passed memcheck but stopped in
initcheck on one uninitialized 32-byte control readback; its 49 saved
values still matched exactly. The snapshot had initialized only the
extent word before the owner copied all four controls. One correction
initializes all four in the existing reset launch. The affected rerun
passes memcheck, initcheck, racecheck and synccheck with zero errors or
hazards and all 49 values unchanged, holding the device for about twelve
seconds. No numerical bound, pair arithmetic or shared helper changed.
Six native schedule lowering audits also pass: NVE plus leapfrog and
velocity-Verlet variants of isotropic, semi-isotropic, anisotropic and
first-order pressure updates. Additional boundary coverage and the
physical-system/end-to-end gates remain required.

The fresh-list constraint-history fixture covers unmarked initial-input
widening, marked geometry within the declared extent in a tilted cell,
and marked excessive extent after a fresh list. Its first comparison
stopped numerically: the fixture had supplied a three-integer member list
where a CSR incidence structure was required, and its overwriting tuple
loop zeroed particles outside the tuple. The approved fixture-only
correction uses `md_exec.build_incidence` at the actual particle count,
accumulates zero corrections, and checks all 48 position components
bitwise before the fresh constructor. The single controlled corrected
execution passes fourteen oracle energies bitwise, all 144 position
checks, and the named post-constraint extent refusal. The oracle, bound
zero and all production code remain unchanged. The refusal fixture
captures unbuffered stdout as well as stderr because `cf.assert` lowers
to `puts` followed by `abort`.

The first small physical OPC fixture uses three waters with plain-cutoff
Lennard-Jones and Coulomb interactions and no dispersion correction. It
takes one template step to obtain a constrained frame, then evaluates the
general double and mixed entries at that same frame. An independent CPU
f64 all-pairs sum enumerates periodic images, forms the actual Amber
bisector and applies its analytic transpose. The oracle energy is
$-49.297633223430694$ kJ/mol against general double's
$-49.297633223264853$: difference $1.6584\times10^{-10}$ kJ/mol. The raw
force difference is at most $2.9746\times10^{-9}$ kJ/mol/nm, with RMS
$1.0494\times10^{-9}$, against a largest double component of
$1403.7811390878728$ kJ/mol/nm. The predeclared double-oracle check is
$10^{-10}\max(1,\lvert U\rvert)$ for energy and
$10^{-10}\max(1,F_\mathrm{max})$ for the largest force error.

The force comparison removes the span of each water's three distance
constraint gradients with the same Euclidean projection for both entries.
On an RTX 5090, general mixed's projected max/RMS error against general
double is $0.00165918/0.000631856$ kJ/mol/nm; the template's is
$0.000674345/0.000217867$. Both template errors are below the accepted
bound $\max(1.5e_\mathrm{mixed},10^{-5}F_\mathrm{max})$, which is
$0.0140378114$ kJ/mol/nm for each quantity here. Its relative energy
error is $1.6761\times10^{-7}$ against bound $10^{-5}$. This first
execution passed without a correction or retry. These are small-system
pair-arithmetic checks, not the larger-system, virial, stability or
end-to-end gates. The oracle uses f64 throughout; the two compiled routes
retain the precision assignments in the table below.

The first corresponding three-site TIP3P fixture stopped on its energy
comparison. Its independent f64 energy is $5.3014572263468054$ kJ/mol;
general double gives $5.3014572263438957$, a difference of
$2.9097\times10^{-12}$ kJ/mol. Their raw force max/RMS difference is
$1.4258\times10^{-9}/6.7265\times10^{-10}$ kJ/mol/nm. The template's
projected force max/RMS errors, $0.000565281/0.000237396$, pass the same
criterion against general mixed's $0.00100856/0.000390067$, with bound
$0.00433485148$ kJ/mol/nm. Its relative energy error,
$1.6695968902592555\times10^{-5}$, exceeded the original $10^{-5}$
relative bound. The first execution remains recorded as a numerical
failure. The signed
template energy and general mixed's energy were not printed, so neither
is inferred from the saved absolute relative error.

The separately authorized single instrumentation-only acquisition reports
the CPU oracle's absolute pair sums as $5.3547091502235036$ kJ/mol for LJ
and $3732.1090605247377$ for Coulomb, beside the same net
$5.3014572263468054$. Their sum is about 705 times the net energy.
Signed potential energies on the same recorded frame are
$5.3013687133789062$ (template), $5.3014259338378906$ (general mixed) and
$5.3014572263438957$ kJ/mol (general double). Template minus double is
$-8.85129649894623\times10^{-5}$ kJ/mol; general mixed minus double is
$-3.12925060050873\times10^{-5}$. The template's absolute energy error
is about 2.82857 times general mixed's, and its relative error remains
$1.6695968902592555\times10^{-5}$ against the unchanged $10^{-5}$ bound.
The executable exits 1 at that assertion. Exact energy, oracle-sum,
coordinate and cell bits are recorded. Inputs and paths were unchanged;
only host diagnostic instrumentation was added, with the oracle's existing
net-energy and force expressions retained. No bound changed for that
acquisition. These data led to the maintainer's cancellation-aware ruling
in the validation gates above, applied equally to OPC and TIP3P.

Each physical fixture then passed its single authorized execution with
the new bound on RTX 5090: TIP3P 1/1 and OPC 1/1. TIP3P's absolute energy
error is $8.85129649894623\times10^{-5}$ kJ/mol against
$0.0037374637696749612$ allowed, with $S=3737.4637696749614$ kJ/mol.
OPC's is $8.26266463604952\times10^{-6}$ against
$0.0090322398755513979$, with $S=9032.2398755513987$. All unchanged
force and oracle/general-double checks passed. No additional numerical
retry followed either execution. The prior failures remain recorded
under their original criterion; the approved change resolves that stop
for these two fixtures without completing the larger-system gates.

An additional first-execution fixture includes three TIP3P waters and two
synthetic charged monatomic solutes. Its independent oracle requires all
three class pairs to be present and counts 27 WW, 18 SW and one SS site
interaction. On RTX 5090 it passes 1/1. The oracle energy is
$-160.3300676795815$ kJ/mol against general double's
$-160.33006767913147$, difference $4.5003\times10^{-10}$; their raw force
max/RMS difference is $2.8897\times10^{-9}/9.7022\times10^{-10}$ kJ/mol/nm.
The template's projected force max/RMS errors are
$0.000934964/0.000357931$ against general mixed's
$0.00204032/0.000636082$, with unchanged bound $0.00841704789$ kJ/mol/nm.
Its absolute energy error is $0.000165030206$ kJ/mol against the approved
$0.00678867271$ allowance, with $S=6788.6727145034756$ kJ/mol. This is a
small all-class arithmetic check, not larger-system or refresh-boundary
coverage. No correction, threshold change or numerical retry occurred.

The first physical OPC total-virial check stops at the independent
oracle/general-double comparison: 0/1 passed. The three recorded scalar
virials are $-16.738173485733569$ (template), $-16.737474841997027$
(general mixed) and $-16.738609513560707$ kJ/mol (general double).
The CPU oracle gives pair $145.6249626905543$, site-transfer
$32.434048928111224$ and constraint $-195.00577757087518$ kJ/mol,
total $-16.94676595220966$. The oracle/general-double difference is
$0.2081564386489525$ kJ/mol against $1.694676595220966\times10^{-9}$
allowed, so this gate fails. The pair/site absolute sum is
$9267.282163805563$ and the constraint absolute sum
$230.99318153942767$, total $9498.2753453449914$ kJ/mol. The diagnostic
template/general-double difference, $0.00043602782713847432$, lies below
its $0.0094982753453449909$ envelope; the independent-oracle failure
blocks validation before that assertion. The energy and projected-force
checks preceding the virial check passed. Exact positions, velocities,
cell and virial bits are retained. The subsequent CPU diagnosis found a
convention mismatch in the new fixture: D264's non-Brownian no-step entry
reports the molecular virial. Twice the saved internal kinetic energy is
$0.2081564380316547$ kJ/mol. The independent analytic constraint reaction
and the replay of the velocity-constraint displacement with curvature
agree to $1.14\times10^{-13}$ kJ/mol. The CPU molecular oracle gives
$-16.738609514178023$, within $6.18\times10^{-10}$ of the saved general
double value. The maintainer approved a fixture-only molecular comparison
with both conventions recorded as above; shared code remains unchanged.

The single approved corrected RTX 5090 run passed. The molecular oracle
is $-16.738609514178052$ kJ/mol; general double differs by
$6.1734439782412664\times10^{-10}$ against the unchanged oracle bound
$1.6738609514178053\times10^{-9}$. The template and general mixed
differences from general double are $0.00043602782713847432$ and
$0.0011346715636797455$ kJ/mol. The molecular absolute contribution sum
is $9497.0534530477507$ kJ/mol, giving the approved template envelope
$0.0094970534530477504$ kJ/mol. The atomic diagnostic remains
$-16.94676595220966$, its full constraint reaction
$-195.00577757087518$, and $2K_\mathrm{internal}$ is
$0.20815643803165473$ kJ/mol. This corrects only the new fixture's
convention; it changes neither production arithmetic nor either bound.

The all-class native epoch fixture adds two translating solutes to the
quadratic water fixture. WW, SW and SS contributions are present in all
seven phases: initial build, two small motions at the pruning threshold,
larger motion requiring a build, full winding, cell expansion and cell
restoration. Independent rational all-image sums agree bitwise in both
rectangular and tilted cells (14 energies, bound zero), on the first
RTX 5090 execution. This exercises generated search, pruning and immediate
consumption rather than seeded work records. It checks cell changes but
does not replace the physical barostat and long-trajectory gates.

The same approved molecular-virial fixture also passed its first TIP3P
water-only and TIP3P-plus-solutes executions on RTX 5090. Their CPU
molecular references are $92.427540176839841$ and
$-31.93733989739232$ kJ/mol. General-double differences are
$9.0381035988684744\times10^{-12}$ and
$3.6726532925968058\times10^{-10}$, below the respective oracle bounds
$9.2427540176839847\times10^{-9}$ and
$3.1937339897392322\times10^{-9}$. Template/general-double differences
are $0.00017925847600963607$ and $0.00073529103738678714$, below
$0.0039424069241047795$ and $0.0070071929698018436$ kJ/mol. The
general-mixed differences are $0.00036740053724315658$ and
$0.00039198553221808652$ kJ/mol. All three pair classes occur in the
solute case. Both fixtures retain the atomic diagnostics and exact bits;
neither changes a bound or production arithmetic.

The new snapshot path enforces D107 before packing any real O/H/H or
solute position: NaN, infinity and magnitudes beyond $10^{100}$ are
counted per particle, and preparation of the invalid unit is skipped.
The host consumes that count from the existing snapshot control packet
and calls the existing named stop before spatial ordering. Search later
reuses the same packet word for its flags; the published arena ABI and
ordinary builders are unchanged. All 20 poisoned cold-build/post-writer
cases passed their first RTX 5090 execution, including two bad hydrogens
in one water reporting a count of two. The same 20 device-free lowerings
and the structural ordering checks passed. The new fixture initially
needed explicit tool paths from lit; its single affected rerun passed.
No existing numerical comparison was retried.

The first approved short trajectory stopped at step 10 of OPC leapfrog
NPT on RTX 5090. The template's maximum velocity difference from general
double was $0.077423795223588388$ nm/ps and RMS
$0.029671181280507719$, both above the unchanged short-trajectory bound
$1.9378578682114735\times10^{-5}$. The general-mixed baseline was
$4.45384523073411\times10^{-6}$ maximum and
$1.1859527987351613\times10^{-6}$ RMS. At zero-based particle 2,
x velocity, template reported $0.14639574733540228$ versus general double
$0.22381954255899067$ nm/ps. All three stored velocity offsets were
$-0.5$ step. Position maximum/RMS differences were
$2.5018192850723153\times10^{-8}$/$8.9255992957879745\times10^{-9}$ nm,
below $4.0001617319761775\times10^{-5}$; volume difference
$2.6898305804934353\times10^{-10}$ was below
$0.000640077634487475$ nm$^3$. Exact positions, velocities and cells
were retained. The serial matrix stopped with one failure and seven
skipped cases; it did not reach a fresh same-frame physical evaluation.
No trajectory rerun or production drift run followed before diagnosis.
The established cause was the shared leapfrog/NPT backward-half-kick
storage retaining a constraint-normal force component. The maintainer
required the separate shared fix first: D393 (#802), merged ahead of this
PR, projects that stored array through the mass-weighted constraints.
After rebasing onto that fix, rerun the stopped OPC/leapfrog/NPT case once
with the raw stored arrays and the same bounds. The helper also records
the independently projected full-step reconstruction as a diagnostic;
it does not replace the raw-array gate. On a pass, continue the seven
other short cases, production drift and remaining implementation gates.

Force-bearing H5MD reports evaluate rule (B) with a separate general
no-step program at the current positions, cell and step. Its activation,
runtime records and force buffers belong to that observation; the
template dynamics' carried rule-(A) forces and checkpoint state remain
unchanged. The program is compiled lazily and reused until the physics
version changes. Its precision is the legacy general mixed route's at
every operation, including the bisector placement and transpose; it adds
no new pair arithmetic. No performance conclusion is drawn from this
reporting check.

The reporter isolation fixture passes two force frames against the general
no-step API at relative bound $10^{-5}$, distinguishes rules (A) and (B),
checks carried positions, velocities and forces bitwise before/after
reporting, and resumes dynamics between reports. This is an API isolation
check, not the independent physical oracle gate. Its first run stopped
before execution because the new list-constructor declaration lacked
read-only annotations on its two input tables. Adding those annotations
passed lowering; the affected run then reached the new helper and found
that the public frame API requires tunables. The helper now uses the
existing general no-step state evaluator, without changing that refusal
or any ordinary entry. Its single affected rerun passed. Neither failure
was a numerical comparison or an intermittent failure.

The execution-body emitter retains independent site guards and the actual
scalar/derived expressions. A second test-only outliner checks this
emitter against the same independent 27-image f64 quadratic oracle and
the same predeclared bound as the semantic reduction. This remains an
isolated body test, not physical-system or device-list validation.

The current pair-arithmetic precision comparison is below. It describes
the execution IR, packing/scheduler emitters and native epoch pipeline;
physical-system and end-to-end measurements are still pending. Compile-time folding of
water coefficients is the rounding difference accepted above, not f64
per-pair device work.

| Operation | Legacy mixed pair kernel | Template mixed pieces |
|---|---|---|
| Source positions and lattice shift before packing | f64 | f64 |
| Reference epochs and packing shifts | f64 reference positions and cells | f64 fractions, integer-valued f64 packing cells, class cells and actual reaches |
| Validity deformation and displacement | f64 integrator/reference arithmetic | f64 guarded inverse-cell norm bound, incident-class margins and anchor displacement |
| Spatial-order key arithmetic | f32 on wrapped points | f64 wrapping, then f32 fractional key arithmetic; i32 class/column/z keys |
| Covering-search geometry and radial predicate | f32 | f64 reference fractions, cell geometry and covering predicate; source site cutoffs remain f32 |
| Inner-block pruning predicate | f32 | f64 hybrid reference fractions, explicit images and covering radius; source site cutoffs remain f32 |
| Packed positions and pair displacement | f32 | f32 whole O/H/H; f32 anchor and local offsets |
| Extra point in the pair body | Stored point loaded through packing | Formed from local offsets in f32 |
| Squared distance and site cutoff | f32 | f32 |
| Charge products | f32 from loaded parameters | Water constants may fold in f64 at compile time, then narrow to f32; dynamic solute products are f32 |
| LJ and erfc/exp arithmetic | f32; existing approximation under `fast_math` | f32; the common radial interface enables D94's admitted table and analytic fallback, with the same approximation pass on the three bodies |
| Per-pair and per-thread real-atom forces | f32 | f32 |
| Force destination | f32 | f32 atomics at the mapped original particles |
| Per-pair virial products | f32 | f32 |
| Energy and virial sums | f64 | f64 |

An isolated one-warp device fixture executes the actual derived WW body
and scheduler with 13 valid first waters in a padded cluster of 16,
35 partner entries, exclusion masks and explicit image shifts. Its
independent quadratic oracle uses dyadic positions and coefficients;
all 469 force/energy/virial values must agree exactly. Both the common
orientation and distinct quarter-turn orientations passed their first
RTX 5090 execution. The latter makes each cached site's value depend on
its own partner and checks off-diagonal strain as well.

| Frames | Reference energy | Reference virial xx | Reference virial xy | Largest reference force component | Largest difference |
|---|---|---|---|---|---|
| Common orientation | 37.024810791015625 | -74.04962158203125 | 0 | 9.099609375 | 0 |
| Distinct quarter-turns | 37.713287353515625 | -74.35430908203125 | 0.545654296875 | 9.52734375 | 0 |

The common-orientation fixture also passed memcheck, racecheck and
synccheck: zero errors or hazards. This isolated device execution does
not validate spatial list generation, the physical PME system, dynamics
or performance; their gates above remain required.

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
