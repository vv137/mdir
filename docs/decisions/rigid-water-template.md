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

### RTX 5090 replay reference

Measurement only. Unchanged source SHA256 `67b7f20293c24783e59798eb99bd2afd5ead2cc810c1d5eecedaeeee5bbe5f81`, CUDA 13.2, `-O3 --generate-line-info -arch=sm_120 -DFUSE_MINB=1`, uncapped and without fast math. Native head `f6999f3cc7dc05603841edf55dbdbc49c2b238c3`; main reference `af7a9f47bd6871324f6a4fc158ee1855f7f643c1`. Every cell is independent job A / B.

| System | Prototype evaluation, us/call | Prototype short-range kernel path, us/frame | Composed whole-step ESTIMATE, us/step | ESTIMATE/main |
| --- | --- | --- | --- | --- |
| ubq-opc | 31.128 / 31.120 | 50.131 / 50.077 | 113.902 / 113.933 | 0.93566 / 0.93937 |
| ubq-tip3p | 29.763 / 29.692 | 49.192 / 49.080 | 100.519 / 100.456 | 1.03211 / 1.03250 |
| opc-small | 24.236 / 24.231 | 31.096 / 31.087 | 78.136 / 77.992 | 0.84843 / 0.84792 |
| opc-large | 53.024 / 52.971 | 62.694 / 62.635 | 143.412 / 143.264 | 0.79899 / 0.79971 |
| tip3p-small | 27.808 / 27.757 | 34.712 / 34.649 | 73.866 / 73.765 | 0.97280 / 0.97291 |
| tip3p-large | 54.819 / 54.887 | 64.500 / 64.578 | 128.428 / 128.576 | 0.93799 / 0.93822 |

The short-range path and ESTIMATE are the prototype's, not measured full dynamics. Composition retains main's host gap and other kernels. Replay uses consecutive main frames at 2 fs spacing, absolute steps 53001–55000 for ubiquitin and 1–2000 for boxes; selected prototype windows retain 53101–55000 and 101–2000. Native evolves independently over its unchanged 10,000-step controls. Prototype reads f32 DCD geometry, uses f32 list geometry and archived force rule B; native stores f64 positions, uses f64 coverage/pruning and rule A. Both use f32 pair/force arithmetic and atomics. Prototype force-only evaluation follows separate pruning; native common-force evaluation includes fused pruning and coexists with periodic energy work. Different histories and work volumes limit interpretation.

Both jobs exited zero: 36 successful native/main wall runs, 35 stationary accepted; all 12 prototype kernel plateaus accepted and 36 captures completed. Job A ubiquitin OPC ON was nonstationary, retained/excluded without an accepted ratio, substitution or retry. Prototype evaluation A/B relative spread is at most 0.2403%. No numerical bound or source changed; this measurement does not mark the PR ready.

## Prototype/native mechanisms and correction plan

This table is the performance plan required by the search-first ruling.
The controlled reference uses the unchanged prototype on RTX 5090 and
native `f6999f3c` against `af7a9f47` as the initial baseline. Current search
includes the retained S1, S3 and S4 corrections; rows and isolated results below
record their measured effects. Evaluation changes `33dfacf7` and `c5f4ea57`
are parked for after search. A planned row is
not closed by a smaller instruction count alone: its correction needs
its own validation and two independent timing comparisons. Every row
must be closed or kept with a stated reason before ready.

Times below are GPU microseconds per step unless marked per call or per
refresh. Ranges cover the six systems and both independent captures.
An evaluation subpart without an isolated measurement is marked as such;
the common evaluation time must not be assigned to each subpart and added.
The detailed six-system diagnosis below supplies the per-refresh values,
launch sizes, work counts and measurement qualifications.

| Stage | Prototype: algorithm, precision, work and RTX 5090 time | Native: algorithm, precision, work and RTX 5090 time | Difference and status |
|---|---|---|---|
| Spatial sort | Per-active-species CUB radix sort of column/z keys; f32 geometry, integer keys, stable input order; 0.218–0.796. | One shared joint stable counting-radix order with fixed-ID ties; f64 wrapping then f32 keys; 49 launches per refresh; 0.940–2.643. Rank restoration scans each item's bucket, giving per-digit work $\sum_b n_b^2$. | **Planned S6:** reduce rank/launch work through the shared primitives, in a separate measured correction. The initial replacement of two insertion sorts is closed by `b0f52fad`; stable fixed-ID ordering stays. A change to pre-existing shared code requires its own scope and gates. |
| Grid and groups | Water-density grid and separate solute sort grid; f32 keys, integer gap/span cuts into at most 16 units; 0.169–1.106 for group planning. | Joint-population, arena-bounded grid; shared integer column/gap/span planner and explicit starts/lengths; 0.251–1.832. First replay-frame groups are 458/131 versus 429/152 for ubiquitin OPC. | **Closed `4abdd348`:** joint stable groups and column cuts. **Planned S7:** measure per-species grid choices within the same arena bound. **Kept on purpose:** general lattice-frame/image coverage; no nearest-image shortcut without proof. |
| Boxes | One Cartesian f32 box per group, 16-lane reductions; source requests 24 scalar reduction shuffles; 0.022–0.086. Search reuses its first-group box and skips padded partner lanes before box loads. | After W1: Cartesian f32 and fractional f64 boxes; native f64 Cartesian producer narrows once, shared anchors feed pair/box checks, and distinct first-frame subsets keep the general proof. Cache build costs 0.063–0.153 per step; exact certificate setup is included in candidate cost, not isolated. | **S0 closed, discarded:** `ac4644f9` guarded padded partner lanes; CPU 6/6 and GPU 22/22 passed, but two independent comparisons found no consistent gain (search/refresh -0.49% to +0.58%, unchanged resources). The guard was removed; retain the original padded-lane arithmetic. S0 measurements are recorded below. **W1 closed by `5ea35cdb`, retained:** certified f32 Cartesian boxes and shared anchors; preparation is 3.61–3.69 us per attempt, versus S4 5.31–5.39. Exact CPU 6/6, GPU 22/22 and focused sanitizer 1/1 pass. B1/B2 are not standalone corrections. **Kept on purpose:** f64 fractional enumeration and the general single-image proof until a separate outward proof replaces them. |
| Candidate search | Direct column endpoints for WW, direct group spans for SW/SS; backward WW/SS ranges removed before boxes; shared Cartesian f32 first anchors; 1.0–1.3 in water boxes, 3.1–3.3 in ubiquitin. | After W1: class-specialized direct endpoints with early backward clipping, no z binary searches, immutable Cartesian f32 anchors and certified decisions/original f64 shell. Candidate cost is 7.10–9.02 in water boxes and 45.24–47.79 in ubiquitin; explicit native frame/image enumeration remains. The complete nine-kernel budget is recorded below. | **Standalone S2 stopped by the whole-port ruling:** its source is reverted without timing it. Early backward-range removal belongs to W1 as part of the complete prototype search structure. **S3 closed by `8115bceb`, retained:** direct descriptor-column endpoints reuse dead scratch and preserve retained classes, SS offsets and z searches. CPU 6/6 and GPU 22/22 pass. All 12 net search-stage comparisons improve 6.8–10.8%, including new clear/scan costs; all 11 available matched wall comparisons improve 0.48–3.94%. Detailed accounting follows. **S4 closed by `f9ea87ef`, retained for modest local gain:** immutable f64 anchor/i32 ID caches use 896 application shared bytes per block. CPU 6/6, GPU 22/22 and focused four-mode sanitizer 1/1 pass. All 12 net search/ref comparisons improve 1.13–4.79%; whole-step results are mixed, including two recorded slowdowns. Do not claim a consistent end-to-end gain. **W1 closed by `5ea35cdb`, retained for measured gain:** the complete nine-kernel stage improves all 12 per-attempt comparisons by 13.61–60.25%, and all 11 eligible normalized wall comparisons by 3.59–9.05%. General covering-image proof, entry format/masks/ownership/order, epochs and validity stay fixed. **Kept with reason, not prototype-equivalent:** the ordered general frame/image iterator and conservative fractional ranges preserve existing coverage/order. Candidate time remains 7.00–15.05 times the archived prototype per step. The next whole correction requires a ruling; no standalone search item resumes. |
| Pruning | Separate f32 Cartesian pruning with shared first anchors and CUB selection; 1.397–1.788 in water boxes, 7.436–7.609 in ubiquitin. | Fused pruning in the production evaluation; f64 hybrid-reference geometry, exact class incidence and image accounting. Its cost is not isolated in the current common evaluation. | **Kept on purpose:** fusion; the earlier separate-prune experiment `3eaad4f7` did not establish a total win. **Planned P1:** apply a proved, exact-membership f32/refined predicate separately from search. Keep reference epochs and certified validity reaches f64. |
| Evaluation: loads | Specialized water coefficients and dynamic solute parameters; compact two-word entries; table coefficients and pair values f32. Included in 24.2–54.9 per common call; not isolated. | Actual source parameter bindings; four-word entries with explicit image/mask fields and original-member mapping; two indexed WW LJ coefficient loads remain. Included in 32.2–91.9 per common call; not isolated. | **Planned E3, after search:** invariant WW LJ operands, preserving coefficient bits and mutable bindings. **Kept on purpose:** explicit image domain, bounded-capacity records and original-particle mapping. Do not narrow their supported domain to match the prototype. |
| Evaluation: extra point | Linear forward site geometry; the partner site is formed from its rotated frame. The force convention is rule B. Part of evaluation; no isolated time. | Exact typed source `alpha*(H1+H2)` cached per loaded partner and rotated with local geometry; dynamics uses rule A. Part of evaluation; no isolated time. | **Planned E4:** compare rematerialization with the rotated cache as its own correction after the approved evaluation items. **Kept on purpose:** source placement and the approved force convention. Anchor-cache experiment `8de4edec` was discarded by `d53ff917` for no clear measured gain. |
| Evaluation: radial and LJ | f32 radial tables and scalar LJ arithmetic; one coefficient-vector load per Coulomb site contribution. Part of evaluation; no isolated time. | Same common pair-body interface, D94 criterion and analytic fallback; f32 pair arithmetic. Fixed water coefficient folding retains its existing precision. Part of evaluation; no isolated time. | **Closed `f0151e73` / `8d2cd418` / `f6999f3c`:** common radial route, its oracle and scalar radial forces. **Kept on purpose:** existing table error/admission contract; no approximation change to obtain speed. WW coefficient invariants remain E3. |
| Evaluation: force form | Scalar radial factor times site displacement, site accumulation; WW static footprints are OPC 348 PTX / 431 SASS and TIP3P 321 / 391, excluding SASS NOPs. | Scalar radial factor times displacement; live site accumulators and one post-accumulation rule-A transpose; OPC 707 / 575, TIP3P 621 / 497. | **Closed `d487f859` / `f6999f3c`:** approximately 10–24% normalized improvement over pre-radial native. **Planned E1:** restore the parked accumulator-free norm guard `33dfacf7` before ready; unproved scalar expressions retain general AD. General `Differentiate` stays unchanged. |
| Accumulation and formed-site distribution | f32 site sums/atomics; archived rule-B distribution. External unpack is measured separately; distribution inside evaluation is not isolated. | f32 site sums/atomics; rule A once at the molecular write boundary. Force-bearing observations use the isolated general rule-B evaluator. Distribution inside evaluation is not isolated. | **Kept on purpose:** the approved rule-A dynamics, rule-B observations, and general deterministic route. **Final gate:** all six systems, both OPC boxes included, compare energy/force/molecular virial to f64 brute force; project force differences off constraint directions under the unchanged declared bounds. |
| Launches and sizes | Search uses two group warps per 64-thread block; small-box WW grids 140–143, large 538–547. Common evaluation uses its archived specialized launches. | After W1: three class-specialized searches each use two group warps per 64-thread block and original row/ordinal compaction. The whole search stage has nine launches. Empty classes retain a minimum-grid launch (about 0.4 us each), and inactive rows copy retained entries; these costs are included. Common evaluation remains one combined launch. | **S1 closed by `5652560d`, retained:** two-warp candidate blocks improve all 11 available normalized comparisons by 1.36–6.07% and all 12 search/refresh means by 12.4–25.8%; one nonstationary wall timing is excluded without retry. CPU 6/6 and GPU 22/22 pass. Box and clear launches retain their sizes; calculated resident warps decrease, not increase. Details and row-work diagnosis follow. **W1 launch port closed by `5ea35cdb`:** 64-thread specialized WW/SW/SS launches and fixed global row IDs; empty/inactive native publication semantics remain explicit. **Planned E2, after search:** restore and measure the parked local mask stream `c5f4ea57`. **Kept on purpose:** current evaluation launch until a separate experiment demonstrates a gain. |
| Search row scheduling | One warp owns each first group; 64-thread blocks, f32 search; included in the search time above. | One warp owns each original class row; W1 uses separate 64-thread class launches with certified f32 Cartesian geometry and retained f64 general query proof. Ubiquitin SW costs 618–627 us per attempt and remains the largest candidate component. No new per-row clocks are inferred from launch p99/max. | **D0 completed at S4:** six exact-output-preserving fixed-frame captures show the latest/longest WW row spans 99.93–99.95% of instrumented kernel wall. Instrumentation overhead and all stage distributions follow below. **W1 closed by `5ea35cdb`:** its measured complete-stage gain is retained. The remaining general-iterator mismatch is explicit above; further whole-search design awaits a ruling rather than another single-mechanism experiment. |
| Refresh and prune cadence | Independently aging WW and SL reference epochs; a WW-only rebuild remaps retained SW IDs/images instead of rebuilding SL. | Shared species reference epochs; WW follows W, SW follows W-or-S, SS follows S. Water-box refresh rates nearly match the prototype; ubiquitin class work has different weighting. | **Kept on purpose:** the approved class-list/epoch contract. The 1.85–1.93 weighted-cost alignment factor is not a literal global frequency multiplier. Independent SL water epochs would be a new design, not a timer adjustment. |

S0 was discarded and S1 retained after their isolated comparisons. The
required D0 diagnostic at S4 and **W1 whole search port** are complete.
W1 has a measured gain but leaves the recorded large prototype gap. The later ruling below supersedes the earlier S2/S5/B1/B2
single-mechanism plan. W1 was measured as one correction against S4. Further whole search
work versus resuming evaluation items E1–E3 awaits the next ruling; E4 needs its own concrete proposal before implementation.
Every measured correction updates this table with its head and effect.

### S0 isolated measurement: padded-lane box guard

The tested head was `ac4644f9`, against main `af7a9f47` and the preceding
radial head `f6999f3c`, on RTX 5090. CPU lowering passed 6/6 and the
controlled GPU fixtures passed 22/22 in 108.09 seconds. Both independent
comparisons completed: 48/48 timing rows passed the unchanged stationarity
rule and 12/12 diagnostic profiles completed. No comparison was retried.
Precision, inputs, bounds and launch sizes were unchanged. PTX and SASS
retain the branch before all twelve f64 box loads; 128 registers and a
464-byte stack remain unchanged.

| System | ON/main A / B | Search microseconds/refresh A / B | ON/main divided by radial A / B |
|---|---|---|---|
| ubq OPC | 1.83852 / 1.82597 | 1785.550 / 1782.699 | 1.00490 / 1.01015 |
| ubq TIP3P | 2.23615 / 2.23802 | 1794.135 / 1776.088 | 0.99298 / 1.00141 |
| OPC small | 1.24085 / 1.24274 | 1351.195 / 1348.874 | 1.00429 / 1.00456 |
| OPC large | 1.22164 / 1.22621 | 1379.727 / 1378.939 | 1.00116 / 1.00424 |
| TIP3P small | 1.45997 / 1.45930 | 1350.367 / 1356.143 | 0.99840 / 0.99764 |
| TIP3P large | 1.50814 / 1.50935 | 1395.236 / 1394.360 | 1.00378 / 0.99979 |

Search time per refresh changes by -0.49% to +0.58%, with mixed signs;
per-step costs also reflect different refresh histories. Both comparisons
show small ON/main slowdowns for the three OPC systems. This does not
establish a useful consistent gain, so the source guard is discarded.
The retained implementation is again the radial baseline. S1 next changes
only the candidate-search block and its ceiling division, leaving boxes,
row ownership, arithmetic and output compaction unchanged.

### S1 isolated measurement and row-work diagnosis

Head `5652560d`, base `af7a9f47`, RTX 5090; the retained comparison is
radial `f6999f3c`. Only candidate launch size and its matching grid ceiling
change. CPU lowering passes 6/6 (15.34 seconds), pre-push checks 6/6, and
clean-head GPU fixtures 22/22 (107.27 seconds). Precision is unchanged:
f64 references/search/pruning geometry, f32 packed pair arithmetic and
force accumulation, f64 energy/virial accumulation.

Both comparisons complete with 48/48 exit-zero timing executions and
12/12 complete profiles. Stationarity accepts 47/48 wall timings. The
first comparison's ubiquitin TIP3P ON timing has no stationary trimmed
plateau and is excluded, preserved and never rerun or replaced. Its
independent profile remains usable. Maximum bracketing-main spread is
0.3885%. All eleven available matched ON/main comparisons improve by
1.36–6.07%; the other ratio remains unknown. Every search/refresh mean
improves by 12.4–25.8%. S1 is retained; ON still exceeds main on every
system, so this is not the performance target or a ready claim.

| System | ON/main A / B | Search microseconds/refresh A / B | Search microseconds/step A / B |
|---|---|---|---|
| ubq OPC | 1.71842 / 1.73420 | 1493.058 / 1492.060 | 61.215 / 60.727 |
| ubq TIP3P | excluded / 2.10159 | 1511.897 / 1512.222 | 63.348 / 63.967 |
| OPC small | 1.17254 / 1.17216 | 1003.890 / 1006.315 | 17.668 / 17.611 |
| OPC large | 1.20154 / 1.20232 | 1204.216 / 1206.808 | 23.121 / 23.171 |
| TIP3P small | 1.37834 / 1.37536 | 1006.407 / 1006.783 | 18.719 / 18.625 |
| TIP3P large | 1.48197 / 1.47821 | 1216.871 / 1213.404 | 24.946 / 24.996 |

Search PTX differs only in `.maxntid`, but compiler allocation changes:
168 registers, 168-byte stack and 228/292 spill-store/load bytes replace
128 registers, 464-byte stack and 560/704 bytes. These are static compiler
resources, not measured spill traffic. The calculated register-limited
ceiling falls from 16 to 12 warps/SM. Small-box grids have about 142
blocks: even perfect distribution cannot occupy every SM, and 12 resident
warps/SM must not be confused with achieved occupancy. More block
parallelism and fewer spills accompany the gain; the experiment does not
isolate their individual effects.

The independent CPU work model uses the six initial unrounded f64 input
snapshots. Its initial outer-entry totals match all six recorded totals:
264865, 267364, 113185, 423498, 116491 and 432421 in the table's order.
Startup constraint changes and later refresh history are not replayed;
these are reconstructed work counts, not GPU row clocks. Each metric is
**mean / p99-higher / maximum**, across all class rows. A candidate batch
is one iteration over at most 32 partner-group descriptors; a partner
batch handles at most two surviving groups. Separate metric maxima can
belong to different rows and cannot be added into one longest row.

| System | Rows | Frame subsets | Bin/image enumerations | Candidate-loop batches | Binary-search steps | Partner batches |
|---|---|---|---|---|---|---|
| ubq OPC | 723 | 1.29 / 4 / 4 | 82.51 / 196 / 196 | 76.32 / 196 / 196 | 1421.76 / 4142 / 4478 | 66.13 / 144 / 192 |
| ubq TIP3P | 724 | 1.29 / 4 / 4 | 83.32 / 196 / 196 | 76.92 / 196 / 196 | 1425.21 / 3727 / 4501 | 66.31 / 136 / 193 |
| OPC small | 283 | 1.12 / 2 / 2 | 87.84 / 98 / 98 | 86.96 / 98 / 98 | 1352.70 / 2159 / 2162 | 78.84 / 119 / 121 |
| OPC large | 1079 | 1.08 / 2 / 2 | 74.61 / 98 / 98 | 74.37 / 98 / 98 | 1501.85 / 2582 / 2611 | 73.68 / 117 / 123 |
| TIP3P small | 284 | 1.13 / 2 / 2 | 88.87 / 98 / 98 | 87.99 / 98 / 98 | 1364.81 / 2167 / 2169 | 80.06 / 123 / 125 |
| TIP3P large | 1087 | 1.09 / 2 / 2 | 74.94 / 98 / 98 | 74.49 / 98 / 98 | 1502.54 / 2582 / 2611 | 74.14 / 119 / 124 |

For an explicit serial-time estimate, calibrate one coefficient on the
first comparison's OPC-small mean, 1003.890 microseconds. Multiplying
maximum partner batches by 8.296609 microseconds predicts 1593, 1601,
1004, 1020, 1037 and 1029 microseconds in table order. Actual means are
1493, 1512, 1004, 1204, 1006 and 1217. This is an empirical proxy, not a
measured service time or a conversion of instructions into cycles. Its
six-system mean absolute percentage error is 7.73%, including the
calibration case. A mean-column-search proxy fits almost equally well,
7.99%, so the longest-row explanation is not uniquely established.

Both jobs give essentially the same diagnosis: kernel-mean differences
are 0.02–0.29%. Reusing the first calibration on the second comparison
gives 7.76% error for maximum partner batches and 8.03% for mean column
steps. These differences are descriptive spreads, not confidence intervals.
The agreement does not remove the identifiability problem.

A separately calibrated aggregate model, total candidate batches divided
by the maximum possible occupied SM count, has 51.08% error. Water work
grows 3.24–3.29 times, that aggregate proxy 2.70–2.72 times, while kernel
time grows only 1.20–1.21 times. This rejects the simplest fixed-cost
aggregate model, but not throughput with latency hiding and changing
contention. Warps sharing an SM share issue and memory resources; neither
all theoretical resident warps nor the grid limit measures actual
parallelism. Row timestamps/stage counters would be needed to distinguish
those mechanisms directly.

The S2 counterfactual preserves each row's retained-entry/mask-bit totals
and class/chunk totals. It approximately halves total water-box batches,
but maximum candidate batches stay 98 in every box. Partner maxima change
121 to 110, 123 to 115, 125 to 121 and 124 to 112. Column searches remain
unchanged. Hence S2 does not promise a twofold critical-path reduction.

The working order selected for this earlier record was **S3, S4, S2**, a source-based priority rather
than a predicted cycle count. S3 removes dependent column searches from
every query: maximum column steps are 1490/1932/1496/1930 in the four
boxes and 3224/3220 in ubiquitin; z searches remain. S4 removes repeated
first-anchor broadcasts from each surviving partner batch. S2 then removes
backward ownership work, with its unchanged enumeration ceiling explicitly
recorded. Each is validated and measured alone; an unsuccessful correction
is discarded. This order may change after its own measured evidence.

**Earlier R1 proposal, superseded by D0/W1 below:** keep one warp per row,
because longest-row dominance is not proved. If serial tails remain after
S3/S4, diagnose per-row elapsed clocks and coarse stage work on fixed frames
before a separate split experiment. A possible split partitions disjoint
frame/image/column ranges and compacts by original row, subrange and local
ordinal. Ownership, images, masks and final ordering must remain unchanged;
allocation/merge/launch costs need a separate measured proposal. R1 is
kept with this reason, not silently implemented or claimed unnecessary.

### S3 isolated measurement: direct descriptor-column endpoints

Head `8115bceb`, matched retained S1 `5652560d`, main `af7a9f47`, RTX 5090.
Inputs, protocol and precision remain unchanged. CPU lowering passes 6/6
in 15.58 seconds, pre-push checks 6/6, and the clean-head controlled GPU
fixtures 22/22 in 110.65 seconds. Both comparisons complete: 48/48 wall
executions exit zero and pass stationarity; 12/12 profiles complete.
The matched S1 first ubiquitin TIP3P wall timing remains unknown, without
substitution; S3's accepted timing does not create that missing comparison.

The new index reuses dead spatial scratch, counts every active and retained
descriptor, and scans it without new arena allocation. The SS partition
offset is preserved. Z searches, images, covering predicates, ownership
and final compaction order remain unchanged. Candidate PTX removes the two
column-bound loops; all 511 f64-typed instructions and 259 scalar shuffles
remain. Static candidate resources are 168 registers, a 176-byte stack and
236/284 spill-store/load bytes, versus S1's 168/168/228/292.

All eleven available matched ON/main comparisons improve 0.48–3.94%.
All twelve **net search-stage** costs improve 6.8–10.8%. The net accounts
for seven S3 kernels versus three in S1: histogram clear, atomic boxes,
three scans, flag clear and candidate search. New clear plus scans cost
2.78–2.90 microseconds per refresh; they are never omitted from the gain.
The existing grouping and sort stages remain outside this subtotal.

| System | ON/main A / B | S3/S1 normalized A / B | Column index microseconds/refresh A / B | Net stage microseconds/refresh A / B | S1 net microseconds/refresh A / B |
|---|---|---|---|---|---|
| ubq OPC | 1.70455 / 1.66592 | 0.99193 / 0.96063 | 2.778 / 2.773 | 1352.039 / 1336.583 | 1499.031 / 1498.033 |
| ubq TIP3P | 2.03912 / 2.05066 | unknown / 0.97576 | 2.826 / 2.819 | 1375.645 / 1374.874 | 1517.846 / 1518.176 |
| OPC small | 1.15947 / 1.16093 | 0.98885 / 0.99042 | 2.897 / 2.899 | 933.564 / 935.327 | 1009.849 / 1012.273 |
| OPC large | 1.19267 / 1.19298 | 0.99262 / 0.99223 | 2.834 / 2.833 | 1127.877 / 1126.439 | 1210.233 / 1212.830 |
| TIP3P small | 1.35763 / 1.36228 | 0.98497 / 0.99049 | 2.835 / 2.830 | 941.597 / 939.092 | 1012.364 / 1012.740 |
| TIP3P large | 1.46929 / 1.47114 | 0.99144 / 0.99522 | 2.882 / 2.880 | 1133.077 / 1136.684 | 1222.891 / 1219.425 |

Helper binaries also serve sorting and compaction, so attribution uses the
seven source-ordered roles within each refresh, rather than global function
name counts. An initial new-analysis resource assertion failed once; the
single corrected affected CPU extraction passed, with no recurrence in the
remaining first-pass extractions. No GPU comparison was retried. PTX scan
application shared memory is 36 bytes; ELF total 1060 includes an explicit
1024-byte compiler reserve. The clear's static 8 registers and measured
CUPTI 16 remain separate unresolved observations: no actual-run binary
identity or driver-rounding cause is claimed. Semantic stage selection,
production classification, numerical bounds and refusals never changed.

Both-job accounting checks verify exactly that saved candidate time minus
index, atomic-box and flag-clear deltas equals the net gain. Candidate
savings are 73.68–92.77 microseconds per refresh in water boxes and
145.10–164.30 in ubiquitin. A one-case calibrated maximum-column-step
proxy predicts the savings with 11.95%/10.20% error, versus
95.15%/94.41% for aggregate work per potential SM. This is compatible with
serial sensitivity, without measuring row clocks or a particular stall.
A negative control is instructive: the old mean-column-step proxy still
fits residual S3 runtime at 7.22%/7.02% error even though those searches
are gone. Cross-system correlation therefore cannot identify the remaining
bottleneck or prove longest-row dominance.

S3 is retained. Its modest isolated gain does not establish that column
searches alone dominated the old serial path. The row model still has the
same enumeration, partner batches and z searches, with column-bound steps
removed. S4 subsequently targeted repeated first-anchor broadcasts as a separate
correction with unchanged f64 operands and its own shared-memory gates.

### S4 isolated measurement: immutable first-anchor cache

Head `f9ea87ef`, matched retained S3 `8115bceb`, main `af7a9f47`, RTX 5090.
Precision, controls and protocol remain unchanged. CPU lowering passes
6/6 in 18.84 seconds, pre-push checks 6/6, and controlled clean-head GPU
fixtures 22/22 in 110.46 seconds. The native sanitizer fixture then passes
1/1 in 18.34 seconds: memcheck/initcheck/synccheck report zero errors and
racecheck zero hazards; its oracle checks pass after every mode.

Both timing comparisons complete with 48/48 exit-zero stationary wall
runs and 12/12 complete profiles, with no exclusions or GPU retries.
All twelve net seven-kernel search costs improve 1.13–4.79% per refresh.
S4 is retained for this measured local gain. Whole-step effects are smaller
and mixed: ten normalized comparisons improve 0.029–2.607%, while the
first ubiquitin TIP3P comparison slows 0.618% and the first OPC-large
comparison slows 0.032%. These observations remain in the report; they
are not a consistent end-to-end gain. AUTO still needs its final end-to-end
gate, and ON remains slower than main on every system here.

| System | ON/main A / B | S4/S3 normalized A / B | Net search stage microseconds/refresh A / B | Net stage microseconds/step A / B |
|---|---|---|---|---|
| ubq OPC | 1.66012 / 1.65044 | 0.97393 / 0.99071 | 1304.301 / 1296.071 | 55.302 / 52.620 |
| ubq TIP3P | 2.05172 / 2.03337 | 1.00618 / 0.99157 | 1309.805 / 1320.359 | 55.798 / 54.927 |
| OPC small | 1.15760 / 1.15921 | 0.99839 / 0.99852 | 911.250 / 912.623 | 15.582 / 15.788 |
| OPC large | 1.19305 / 1.19140 | 1.00032 / 0.99867 | 1113.576 / 1111.963 | 21.158 / 21.350 |
| TIP3P small | 1.35723 / 1.35502 | 0.99971 / 0.99467 | 921.705 / 921.362 | 17.144 / 17.229 |
| TIP3P large | 1.46657 / 1.46679 | 0.99815 / 0.99704 | 1120.258 / 1118.512 | 22.853 / 22.818 |

Per-refresh gains do not imply proportional per-step gains. For example,
first-comparison ubiquitin OPC net cost is 55.302372 microseconds per step
versus S3's 55.298404, essentially unchanged despite a 3.53% per-refresh
gain; captured refresh attempts are 424 versus 409. This is accounting
from those histories, not a causal explanation or excuse for wall results.

Static review confirms immutable f64 data, distinct per-warp slots and
one full-mask warp barrier; only lower sixteen lanes store. Cache loads
remain within the selected-partner loop; the original membership ballot
replaces flag shuffles. Original f64 arithmetic, images, ownership and SS
exclusions remain unchanged. Native combined search has 2280 PTX
instructions, 131 scalar shuffles and 4091 non-NOP SASS, versus S3's
2521/259/5058. Resources are 168 registers, 168-byte stack and 232/272
spill-store/load bytes. Application shared memory is 896 bytes, separately
from a 1024-byte compiler reserve (ELF total 1920). Those are static
resources, not executed traffic or proof of shuffle latency.

For the unchanged prototype reference, the following compares search only
with search only, in microseconds per step. Native net seven-kernel cost
above additionally includes boxes, indexing and flag clear and must not be
used as if it were the same component. Both use RTX 5090, but native f64
geometry and integrated histories differ from the prototype's f32 replay
and reference epochs; this is a mechanism comparison, not a new oracle.

| System | Native search A / B | Prototype search A / B | Native/prototype A / B |
|---|---|---|---|
| ubq OPC | 54.928 / 52.262 | 3.061 / 3.055 | 17.945 / 17.108 |
| ubq TIP3P | 55.421 / 54.559 | 3.271 / 3.269 | 16.942 / 16.689 |
| OPC small | 15.430 / 15.634 | 0.980 / 0.979 | 15.744 / 15.968 |
| OPC large | 20.989 / 21.178 | 1.197 / 1.198 | 17.531 / 17.673 |
| TIP3P small | 16.979 / 17.064 | 1.083 / 1.082 | 15.677 / 15.768 |
| TIP3P large | 22.670 / 22.634 | 1.267 / 1.270 | 17.894 / 17.827 |

A large search gap remains. The earlier S2 proposal clipped backward WW/SS
descriptors, preserving SW, self, refusal flags and ordering through its
zero-ownership proof. The superseding ruling stops that standalone item
without timing it; early removal now belongs to W1. The bounded f32/f64-shell
proof likewise supports W1, rather than a separate S5 correction.

### Superseding ruling: one diagnostic, then the whole search structure

The maintainer stopped S2 and further standalone search corrections. The
already-running S2 fixture run completed 22/22; it is not a retained
correction or a timing result. Its source is reverted
so the active search again equals S4. No S2 timing job is submitted.

First, use a separate diagnostic-only build based on S4 `f9ea87ef`, fixed
frames of all six systems, and one device-clock measurement. Record each
row's z search, frame/image/subset enumeration, candidate loop, partner
batches and append elapsed time; report mean, p99 and maximum. Report
inclusive and exclusive intervals clearly, prologue/epilogue/instrumentation
residuals, row timelines, and whether the kernel duration follows the
longest row. This is not the two-job timing protocol. Instrumentation and
its build are not retained; logs, patch identity and resources are kept.

Then implement W1 regardless of the diagnostic's performance result:
class-specialized WW/SW/SS search kernels; prototype direct column/group
endpoints and early backward-range removal; shared first anchors; its
launch distribution; and f32 Cartesian geometry under the declared
certified-accept/reject plus literal-original-f64 uncertainty shell.
Retain the native general covering-image proof where the prototype's
nearest-image assumption is narrower. Entry format, masks, ownership,
output order, reference epochs and validity remain unchanged.

Exact covering, boundary, triclinic, reframed, growth and brute-force
coverage fixtures gate W1. A coverage counterexample stops work and does
not justify widening a bound. Measure the complete port as one correction
with the usual two independent jobs against S4. Close or justify the search,
box and launch rows together from that evidence. Sort/group plan S6/S7 may
follow with the same whole-structure approach if their gap remains large.
No extra single-mechanism search items intervene before W1.

### Completed single S4 row-clock diagnostic

The detached diagnostic uses S4 `f9ea87ef`, with uncommitted diagnostic-only
edits and patch SHA256
`383619218876fdb132e3290cc9f5a36fd28eea0724d034596e53e577a2023ff2`.
It is not a correction and is not kept in the PR implementation. Six
one-step program invocations on RTX 5090 completed; every natural growth
and search attempt was retained. The predefined primary selection is the
first successful initial refresh, not the fastest sample. No standard
warmup/repeat/two-job timing protocol is used for this diagnostic.

All six selected outputs match the source-faithful CPU replay exactly:
264865, 267364, 113185, 423498, 116491 and 432421 ordered entries,
including row/ordinal, partner, image, mask, reserved word, counts and flags.
The replay uses the actual dumped references/order/groups/keys/cells/reaches/
exclusions and the original diagonal multiply plus x/y/z FMA norm graph.
This validates output preservation, not independent physical-force or
full-trajectory/state equality.

Each one-step program naturally builds an initial-force list and a separate
advanced-position list inside the checkpoint segment; reuse is not hoisted
across that segment. Each allocation grows once and then succeeds, producing
24 retained search attempts across six processes. All primary samples were
fixed to serial 1, attempt 1. Serial 2 is retained, not substituted. Serial
is a process-global refresh-call sequence, not a step number; allocation
association is established from source/dataflow because headers omit handle
IDs. Successful serial-1/2 snapshots and payloads differ in every case.

Instrumentation preserves the original 511 static f64 operations and existing
barriers but increases stack/spills materially: 168 registers, 448-byte stack,
468/580 static spill-store/load bytes, versus S4's 168-byte stack and
232/272 bytes. Application shared memory stays 896 bytes; ELF includes a
separate 1024-byte reserve. A 64-byte private depot remains at outer query
boundaries. There is no paired uninstrumented exact-frame run or fabricated
constant overhead subtraction. The production S4 means below provide scale,
not production per-row clocks or a direct overhead estimate.

Times below are instrumented elapsed microseconds. Five exclusive budgets are z; candidate/filter/control excluding partner; partner mask/enumeration excluding append; append; and residual. Candidate-inclusive contains partner-inclusive, which contains append; z is disjoint. Residual=total−z−candidate-inclusive includes setup/cache/frame/image/subset/control/timer/finish overhead and is not a measured pure enumeration stage. P99 uses the higher observed order statistic. Row intervals overlap: sums across rows and sums of stage maxima are not kernel wall time.

| System | Active rows | Kernel W | Row envelope E | Longest L | Latest row | Start delay | W−E | Production S4 search A/B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ubq-opc | 723 | 1395.070 | 1394.432 | 1394.368 | 199 (WW) | 0.064 | 0.638 | 1295.472/1287.249 |
| ubq-tip3p | 724 | 1438.814 | 1438.208 | 1438.144 | 247 (WW) | 0.064 | 0.606 | 1300.962/1311.513 |
| opc-small | 283 | 940.350 | 939.712 | 939.680 | 3 (WW) | 0.032 | 0.638 | 902.321/903.700 |
| opc-large | 1079 | 1132.830 | 1132.224 | 1132.160 | 5 (WW) | 0.064 | 0.606 | 1104.661/1103.046 |
| tip3p-small | 284 | 959.934 | 959.296 | 959.264 | 3 (WW) | 0.032 | 0.638 | 912.839/912.498 |
| tip3p-large | 1087 | 1146.493 | 1145.920 | 1145.856 | 0 (WW) | 0.064 | 0.573 | 1111.281/1109.528 |

In these six instrumented initial launches, the longest row is also the latest finishing row. Its32–64ns start delay plus its total elapsed interval equals E exactly; W−E is0.573–0.638us. Thus the long row envelope accounts for essentially the entire instrumented kernel duration. Production S4 values are separately recorded stationary per-refresh means over evolving geometry, not a paired uninstrumented run on these fixed snapshots. They provide scale, not an instrumentation overhead estimate or production row-clock proof.

| System | Exclusive stage | Mean | P99 | Max |
|---|---|---:|---:|---:|
| ubq-opc | z lookup | 122.294 | 336.992 | 393.248 |
| ubq-opc | candidate/filter/control | 170.904 | 434.720 | 446.656 |
| ubq-opc | partner mask/enumeration | 174.654 | 361.408 | 407.840 |
| ubq-opc | append | 12.051 | 28.224 | 36.672 |
| ubq-opc | setup/enumeration residual | 103.099 | 248.224 | 266.560 |
| ubq-tip3p | z lookup | 123.403 | 335.424 | 403.328 |
| ubq-tip3p | candidate/filter/control | 172.312 | 435.808 | 450.624 |
| ubq-tip3p | partner mask/enumeration | 173.978 | 368.160 | 402.752 |
| ubq-tip3p | append | 12.124 | 27.360 | 36.064 |
| ubq-tip3p | setup/enumeration residual | 104.067 | 231.296 | 275.328 |
| opc-small | z lookup | 127.713 | 205.728 | 208.608 |
| opc-small | candidate/filter/control | 186.438 | 216.128 | 216.352 |
| opc-small | partner mask/enumeration | 212.778 | 395.712 | 402.656 |
| opc-small | append | 13.608 | 25.600 | 26.432 |
| opc-small | setup/enumeration residual | 104.211 | 135.712 | 137.536 |
| opc-large | z lookup | 150.052 | 241.120 | 251.712 |
| opc-large | candidate/filter/control | 187.796 | 284.256 | 288.640 |
| opc-large | partner mask/enumeration | 237.837 | 436.512 | 480.064 |
| opc-large | append | 13.199 | 22.272 | 28.064 |
| opc-large | setup/enumeration residual | 124.309 | 194.496 | 208.928 |
| tip3p-small | z lookup | 129.211 | 204.448 | 208.288 |
| tip3p-small | candidate/filter/control | 189.203 | 215.936 | 216.736 |
| tip3p-small | partner mask/enumeration | 215.848 | 402.176 | 438.240 |
| tip3p-small | append | 13.927 | 25.504 | 29.120 |
| tip3p-small | setup/enumeration residual | 105.590 | 137.184 | 138.112 |
| tip3p-large | z lookup | 149.696 | 239.872 | 250.272 |
| tip3p-large | candidate/filter/control | 188.444 | 286.944 | 297.632 |
| tip3p-large | partner mask/enumeration | 239.431 | 435.904 | 515.904 |
| tip3p-large | append | 13.216 | 22.784 | 26.752 |
| tip3p-large | setup/enumeration residual | 125.580 | 194.784 | 207.264 |

The latest row decomposition is measured on that same row, not assembled from different maxima.

| System | Start delay | z | Candidate exclusive | Partner exclusive | Append | Residual | Envelope E |
|---|---:|---:|---:|---:|---:|---:|---:|
| ubq-opc | 0.064 | 367.776 | 425.728 | 311.808 | 36.672 | 252.384 | 1394.432 |
| ubq-tip3p | 0.064 | 403.328 | 442.272 | 283.936 | 33.280 | 275.328 | 1438.208 |
| opc-small | 0.032 | 208.608 | 215.072 | 352.224 | 26.240 | 137.536 | 939.712 |
| opc-large | 0.064 | 231.616 | 261.536 | 402.016 | 28.064 | 208.928 | 1132.224 |
| tip3p-small | 0.032 | 208.288 | 214.624 | 370.048 | 29.120 | 137.184 | 959.296 |
| tip3p-large | 0.064 | 188.416 | 258.112 | 515.904 | 18.656 | 164.768 | 1145.920 |

No single measured stage explains every long row. Candidate/filter/control, partner mask/enumeration, z lookup and the combined residual all contribute materially; append is smaller. The residual does not isolate frame/image enumeration, and elapsed stage spans include scheduling/memory/timer effects. One representative SM ID is recorded twice after compiler CSE; this is not an independent migration detector. clock64 is auxiliary and was not compared across SMs or converted using a nominal clock. No achieved occupancy, memory throughput, instruction latency or causal speedup is claimed.

The first helper attempt failed before JIT execution because the full host runtime had not been built. Its exported capture has zero CUDA activity/kernel tables and no row dumps. Evidence and cold cache were preserved. One deterministic setup correction built the complete runtime pair and verified both with CPU RTLD_NOW loader checks; the fresh corrected invocation is the only actual measurement. The premeasurement decoder was corrected to ignore uninitialized unused class/inner metadata (affected CPU smoke1/1 passed). During postcapture analysis, the wrapper-integrity parser initially assumed SHA256; unchanged cache source establishes BLAKE3, and one corrected extraction passed all six. Neither analysis correction changed source predicates, classification, refusal, tolerance, nor any GPU run.


The required next step remains W1, the complete prototype search structure,
regardless of this diagnostic result. No row-splitting experiment or further
single-mechanism search correction is inserted before it.

### W1: complete search correction against S4

The whole correction is `5ea35cdb`. The source and build stayed fixed
through the exact-coverage, sanitizer and two independent timing gates.
CPU lowering passed 6/6; the first full build had one new helper API
argument-order error, corrected once before a successful rebuild.
Independent bounded CPU models passed 2,160 pair/singleton-box decisions
and 28 directed threshold conversions. Three host underflow coefficients
were rounded outward during source review, before device comparisons.
All six push checks passed. Production PTX for all three classes has
rounded f64 division and square root, ordinary f32 arithmetic/FMA, and no
FTZ or approximate instructions.

Exact coverage and physical fixtures passed 22/22 in 139.38 seconds on
RTX 5090, including boundary, seam, triclinic, reframed and growth cases,
TIP3P/OPC energy/force and molecular virial. The focused sanitizer passed
1/1 in 22.29 seconds: memcheck, initcheck and synccheck each reported
zero errors; racecheck reported zero hazards. No numerical retry or bound
change occurred.

The subsequent documentation CPU tier reported 18 passed and one failed:
the unchanged output-cleanup checker found nine fixtures added by this PR
without an initial removal of their temporary outputs. Adding only those
removal commands fixed the affected checker in one rerun (1/1 passed).
No production code, checker, refusal or numerical bound changed, and no
device result was rerun. The measured source head remains `5ea35cdb`;
the follow-up commits contain fixture cleanup and this report.

Both normal-protocol timing jobs completed with exit zero. All 48 wall
runs exited zero; 47 were stationary. B ubiquitin TIP3P ON had no stationary
trimmed plateau and is excluded, with no replacement or rerun. All 12
profiles completed and were attributed to the nine ordered source roles.
The unchanged protocol uses 24 main/ON/OFF/main runs and six ON profiles
per job. S4 `f9ea87ef` is the retained corresponding A/B reference,
not a same-job pairing. The main reference remains `af7a9f47`.

| System | A ON ms/step | A ON/main | A normalized W1/S4 | B ON ms/step | B ON/main | B normalized W1/S4 |
|---|---:|---:|---:|---:|---:|---:|
| ubq OPC | 0.19246 | 1.581 | 0.952 | 0.19348 | 1.591 | 0.964 |
| ubq TIP3P | 0.18999 | 1.947 | 0.949 | excluded | excluded | excluded |
| OPC small | 0.09806 | 1.066 | 0.921 | 0.09787 | 1.065 | 0.919 |
| OPC large | 0.20141 | 1.125 | 0.943 | 0.20120 | 1.122 | 0.942 |
| TIP3P small | 0.09361 | 1.235 | 0.910 | 0.09340 | 1.232 | 0.910 |
| TIP3P large | 0.18729 | 1.368 | 0.933 | 0.18737 | 1.368 | 0.932 |

All 11 eligible normalized comparisons improve by 3.59–9.05% against S4;
none has a raw ON-time slowdown. ON remains slower than main in every
eligible comparison. OFF/main is 0.999–1.002. The largest bracketing-main
spread is 0.45%. These are observed differences, not a statistical
significance or automatic-selection claim.

The following microseconds are per captured refresh attempt, including
initial growth. W1 net9 includes histogram clear, preparation/boxes,
three scans, class clear and three candidate kernels. S4 net7 includes
the corresponding indexing/preparation plus its combined candidate kernel.
All overhead, including empty-class launches, is included.

| System | Job | Attempts | Index clear | Prep/boxes | Three scans | Class clear | WW | SW | SS | W1 net9 | S4 net7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ubq OPC | A | 413 | 0.412 | 3.619 | 2.365 | 0.716 | 207.623 | 619.818 | 267.860 | 1102.414 | 1304.301 |
| ubq OPC | B | 421 | 0.412 | 3.617 | 2.362 | 0.715 | 204.835 | 618.174 | 269.042 | 1099.156 | 1296.071 |
| ubq TIP3P | A | 425 | 0.422 | 3.611 | 2.401 | 0.708 | 214.297 | 626.556 | 283.587 | 1131.582 | 1309.805 |
| ubq TIP3P | B | 422 | 0.422 | 3.613 | 2.399 | 0.708 | 215.799 | 626.977 | 276.632 | 1126.551 | 1320.359 |
| OPC small | A | 174 | 0.426 | 3.618 | 2.469 | 0.714 | 407.188 | 0.403 | 0.402 | 415.219 | 911.250 |
| OPC small | B | 176 | 0.425 | 3.622 | 2.468 | 0.712 | 408.087 | 0.399 | 0.403 | 416.116 | 912.623 |
| OPC large | A | 192 | 0.432 | 3.682 | 2.404 | 0.713 | 435.397 | 0.407 | 0.407 | 443.442 | 1113.576 |
| OPC large | B | 193 | 0.429 | 3.679 | 2.407 | 0.712 | 435.609 | 0.406 | 0.404 | 443.645 | 1111.963 |
| TIP3P small | A | 190 | 0.418 | 3.628 | 2.414 | 0.708 | 410.943 | 0.412 | 0.404 | 418.927 | 921.705 |
| TIP3P small | B | 188 | 0.417 | 3.628 | 2.415 | 0.710 | 409.399 | 0.412 | 0.403 | 417.384 | 921.362 |
| TIP3P large | A | 206 | 0.418 | 3.686 | 2.462 | 0.713 | 437.174 | 0.409 | 0.404 | 445.268 | 1120.258 |
| TIP3P large | B | 205 | 0.419 | 3.687 | 2.461 | 0.711 | 436.817 | 0.406 | 0.406 | 444.907 | 1118.512 |

Net per-attempt time improves in all 12 comparisons, by 13.61–60.25%.
Including the actual captured cadence gives 12.06–60.03% improvement in
net search-stage time per step. Attempts are not active per-class rebuilds:
WW follows W, SW follows W-or-S and SS follows S; an inactive class copies
retained rows. Every attempt launches all three classes, including a
minimum-grid empty launch. These empty SW/SS launches cost about 0.4
microseconds each in water boxes and remain in the budget.

The direct class-kernel comparison below gives native/prototype
microseconds per native attempt or prototype class event. It is not
matched-work causality: native attempts include inactive copies, while
the prototype has separate WW and SW/SS schedules and different replay
geometry. Water-only prototype SW/SS launches are absent; native empty
launches are already included above.

| System | WW A native/prototype | SW A native/prototype | SS A native/prototype | WW B native/prototype | SW B native/prototype | SS B native/prototype |
|---|---:|---:|---:|---:|---:|---:|
| ubq OPC | 207.623 / 54.325 | 619.818 / 29.438 | 267.860 / 58.017 | 204.835 / 54.213 | 618.174 / 29.363 | 269.042 / 57.922 |
| ubq TIP3P | 214.297 / 54.292 | 626.556 / 29.853 | 283.587 / 58.917 | 215.799 / 54.033 | 626.977 / 29.829 | 276.632 / 59.053 |
| OPC small | 407.188 / 56.425 | empty / absent | empty / absent | 408.087 / 56.370 | empty / absent | empty / absent |
| OPC large | 435.397 / 63.188 | empty / absent | empty / absent | 435.609 / 63.245 | empty / absent | empty / absent |
| TIP3P small | 410.943 / 57.159 | empty / absent | empty / absent | 409.399 / 57.115 | empty / absent | empty / absent |
| TIP3P large | 437.174 / 63.345 | empty / absent | empty / absent | 436.817 / 63.484 | empty / absent | empty / absent |

| System | Native preparation A/B | Prototype WW boxes A/B | Prototype SW/SS boxes A/B |
|---|---:|---:|---:|
| ubq OPC | 3.619 / 3.617 | 1.290 / 1.288 | 2.358 / 2.351 |
| ubq TIP3P | 3.611 / 3.613 | 1.292 / 1.290 | 2.448 / 2.445 |
| OPC small | 3.618 / 3.622 | 1.271 / 1.272 | absent |
| OPC large | 3.682 / 3.679 | 1.403 / 1.403 | absent |
| TIP3P small | 3.628 / 3.628 | 1.260 / 1.259 | absent |
| TIP3P large | 3.686 / 3.687 | 1.400 / 1.407 | absent |

Native preparation includes all hybrid descriptors, Cartesian-anchor
cache production, fractional boxes and the descriptor histogram. The
prototype box events cover their respective active classes. The scopes
and cadence differ; no combined prototype preparation event is invented.

| System | W1 candidate us/step A/B | Prototype candidate us/step A/B | Ratio A/B |
|---|---:|---:|---:|
| ubq OPC | 45.236 / 45.975 | 3.061 / 3.055 | 14.779 / 15.050 |
| ubq TIP3P | 47.789 / 47.239 | 3.271 / 3.269 | 14.609 / 14.450 |
| OPC small | 7.099 / 7.196 | 0.980 / 0.979 | 7.244 / 7.350 |
| OPC large | 8.375 / 8.423 | 1.197 / 1.198 | 6.995 / 7.029 |
| TIP3P small | 7.823 / 7.712 | 1.083 / 1.082 | 7.224 / 7.126 |
| TIP3P large | 9.023 / 8.971 | 1.267 / 1.270 | 7.122 / 7.066 |

The prototype figures are the retained unchanged RTX 5090 captures, with
their own reference schedules and replay geometry. Native dynamics evolves
independently. The ratios are remaining-gap context, not matched-work
causality, equivalent coverage contracts or whole-step speedups. Summing
one prototype WW event and one SW+SS event gives an algebraic aligned-cost
ratio of 6.894–7.860; it is not a jointly observed event or a literal
refresh-rate multiplier.

The structural difference still present in source is explicit: native
first-occurrence frame subsets, integer-image enumeration and conservative
fractional column/z-image bounds remain, including for SW/SS. The
prototype scans direct class spans with its narrower nearest-image box
assumption. Native z binary searches are gone. Certificate setup and the
literal f64 shell add static code; neither their dynamic contribution nor
the iterator's isolated time is measured by these profiles. SW is the
largest observed candidate component in ubiquitin, 618–627 microseconds
per attempt, versus 205–216 WW and 268–284 SS. Sort and group planning
are only 62 and 42–44 microseconds there. This evidence does not justify
attributing the remaining gap to throughput, spills or one instruction.

| Candidate | Registers | Stack bytes | Application shared bytes | ELF shared bytes | Non-NOP SASS |
|---|---:|---:|---:|---:|---:|
| W1 WW | 168 | 128 | 1152 | 2176 | 4926 |
| W1 SW | 128 | 152 | 1152 | 2176 | 4938 |
| W1 SS | 168 | 136 | 1280 | 2304 | 5590 |
| S4 combined | 168 | 168 | 896 | 1920 | 4091 |
| Prototype WW | 112 | 0 | not separated | 34304 | 1137 |
| Prototype SW | 102 | 0 | not separated | 34304 | 940 |
| Prototype SS | 102 | 0 | not separated | 34432 | 1368 |

W1/S4 ELF shared figures include the explicit 1024-byte compiler reserve;
prototype figures are the archived resource reports, without assuming an
equivalent reserve split. All candidate blocks have 64 threads. Both
native models produce identical role cubins. Preparation uses 46 registers
and zero stack/shared, versus S4's 60-register preparation. Clear/scans
are byte-identical to S4. Static footprints are not executed instruction
counts or spill traffic; explicit spill bytes were not collected. The
clear's static 8/CUPTI 16 register discrepancy remains unresolved and is
not explained by an invented driver-rounding rule. Source order, launch
resources and surrounding context identify roles; generic kernel names
alone do not establish actual binary identity.

The two initial timing wrappers failed before opening their first output
log because preparation omitted group-write permissions. Both records
were retained; neither reached the MDIR spawn or produced numerical or
profile artifacts. One permission correction and actual-account write
preflight preceded the one corrected submission per job. Analyzer setup
also had one deterministic whitespace syntax error, corrected before it
wrote results. These are harness corrections, not numerical retries.

W1 is retained for its measured gain, but the remaining large prototype
gap is not closed by this correction. The subsequent ruling authorizes
the image strategies below. The parked norm proof guard and evaluation
items remain pending; this is not a ready claim.

### Refresh-selected image strategies

The accepted next correction keeps W1's entry ABI, class ownership,
incidence masks, epochs, validity and literal final membership predicate.
It selects separate outlined search functions at refresh scope:

1. One pass image when the greatest actual class reach, including an
   outward numerical margin, is at most half the least cell diagonal.
2. Pass-first conditional neighboring images when that padded reach is
   strictly below the least diagonal. Traversal is c, b, a; b and a are
   re-centered for each higher-axis prefix, as in D241/D242.
3. The unchanged covering-image iterator otherwise. Every such refresh
   logs the general fallback. This is a fallback, not a new refusal.

`ImageEnumerationEmitter` is a common pair-image strategy: it receives
the cell, displacement, padded reach and a visitor. It knows no water
model, class tags, entry layout or membership predicate. Its callers
must establish reduced-cell and numerical preconditions. The template
adapter uses the same emitter for all site counts and parameter sets.
Publication is currently specific to the segmented entry ABI because its
ordering key includes frame-subset and descriptor order. The reusable
publication mechanism is tracked in #841.

The selector evaluates positive error arithmetic outward in f64, taking
the next representable value and a min-normal underflow floor. It covers
the original absolute-frame displacement graph, its rounded squared norm,
and the new Cartesian c/b/a prefix graph. If $M$ bounds the absolute floor
of every hybrid reference coordinate and $h$ is the outward sum of cell
magnitudes, the prefix envelope includes
$\gamma_{64}h(128+24M)+\beta_{256}$, where
$\gamma_n=nu/(1-nu)$ and $\beta_n=n\nu/(1-nu)$,
$u=2^{-53}$ and $\nu=2^{-1022}$. This is a counted rounding bound, not
physical skin or an observed-error fit. The selected margin also covers
the ordinary rounded addition of reach and margin. Finite arithmetic,
reduced-cell and normal-core volume/face-width guards must all hold;
otherwise the general strategy runs. The entry-image ABI further requires
$M\le251$: with each provisional image component bounded by eight,
$2M+8<512$ prevents provisional overflow before the original query-domain
filter. Larger reference frames use the general iterator.

| Operation | Retained W1 | Image strategies |
|---|---|---|
| References, winding, cell, reaches and validity | f64/i64 | Unchanged |
| Refresh dispatch and margin | General iterator | Outward f64; separate host-selected functions |
| Pair-local image prefixes | General f64 iterator | f64, correctly rounded reciprocal/multiply and round-to-even |
| Cartesian boxes and radial certificate | Certified f32, original f64 uncertainty shell | Unchanged |
| Final domain and membership | Original rounded f64 predicates | Same predicates, including the original query domain |
| Entry publication | Row/ordinal compaction | Same records; stable canonical ordering restores original frame/x/y/z/partner order |

No image-policy branch is added inside a candidate kernel. The temporary
canonical-order key uses dead planning/staging storage; sorting, gathering
all rows and final copying are separate phases so permutation scratch
cannot alias another row's retained entries. The public reserved word is
zero again before consumption. The general strategy keeps its original
publication path.

Gates include exact ordered comparisons against a forced general iterator
for nearest, conditional, coupled tilted-cell, boundary and refresh cases;
an independent all-images oracle; and a predeclared native barostat run
crossing from nearest to conditional. The new outline fixture initially
failed during construction because a detached callback had no parent
module. Passing the module explicitly removed the crash. Its next run
completed lowering and exposed a separate `func.call` versus `call`
FileCheck spelling error; one correction and rerun passed 1/1. The six
existing search/refresh lowering fixtures and independent image reference
then passed 7/7. The first five-fixture device job stopped during CPU
generation of the new conditional input, before either MDIR route ran:
the chosen dyadic reach coincided with a lattice distance and violated
the existing nonboundary-input guard (one failed, four skipped). Only the
new inputs were corrected to odd-over-128 base reaches, 119/128 for
conditional/coupled and 97/128 for ordering. All three generators passed
once with minimum squared-distance gaps 0.00103759765625,
0.0007934603490866721 and 0.00006103515625. The guard, oracle and comparison
criteria are unchanged. Device correctness and performance remain
unmeasured for this correction.

### Target-driven remaining batches and reporting

The subsequent target ruling supersedes a full timing round after every
single edit. Each batch must pass exact coverage and physical/virial
gates, with sanitizer coverage for device-code changes. Short profiles and
static footprints steer implementation. Revert a batch without measured
stage improvement. Run the full two-job timing protocol when a batch is
expected to reach a target, or after a few batches to check progress.

| RTX 5090 target | Required GPU time or ratio |
|---|---|
| Common evaluation, large OPC/TIP3P boxes | At most 55 us/step each |
| Common evaluation, small OPC/TIP3P boxes | At most 25/28 us/step |
| Common evaluation, ubiquitin OPC/TIP3P | At most 32/31 us/step |
| Ubiquitin search, sort and group planning | At most main neighbor build and prune |
| Water-box search, sort and group planning | At most retained W1 |
| Other kernels | At most main, with kernel-level attribution |
| Whole step | ON/main below one on all six systems |

From the image-path measurement onward wall timings use **100,000 steps**
and a rate every 1,000 steps, with the unchanged plateau rule, trims and
two-job interleaving. Each job measures main, retained W1 and the new head
at that length. Per-kernel profiles may use 10,000 steps and are labeled
separately; old 10,000-step wall ratios are not a same-length baseline.
Every result comment includes both ON/main tables, supporting stage/kernel
numbers, gate counts, exclusions and spread. Only process narrative moves
to supporting records.

After image paths, inventory the remaining kernels from existing
captures, then resume evaluation. E1 is the parked norm proof guard;
E2 streams masks; E3 exposes fixed pair constants. E5 specializes only
classes present, derived from host descriptor counts. E6 selects separate
prune and nonprune functions from the invalidation flags already read on
the host, without another device readback. All specialization comes from
program properties with checked preconditions and shared emitters; no
handwritten kernel for a named water model is admitted. Any template-only
mechanism requires a stated reason and a general-form issue.

### Precision contract for the whole search port and future pruning

The first corrections are control/launch changes: every f64 predicate,
image, reach, source admission and accuracy bound remains unchanged.
The following argument is declared before a possible f32 correction,
following [How bounds are set](../validation.md#2-how-bounds-are-set).

Normalize reference fractions in f64 first. Evaluate local wrapped
differences and a small exact integer image in f32; retain winding,
reference epochs, cells, fractional ranges, image proofs and validity in
f64/i64. This avoids making the f32 error scale depend on an arbitrary
unwrapped origin. Conversion, subtraction and cell-matvec bounds use the
component magnitudes, $u_{32}=2^{-24}$, an absolute underflow term, and
$\gamma_n=n u/(1-n u)$; they are not a fitted multiple of the measured error.

Let $\delta_p$ bound the Cartesian-vector error of precision path $p$
against a common exact-real displacement, and let $A$ bound its norm.
The f64 bound includes the **original** wrapping, absolute-image
subtraction and cancellation graph. A conservative squared-norm bound is

$B_p=\gamma_{5,p}(A+\delta_p)^2+2A\delta_p+\delta_p^2+\beta_{5,p}$.

Here $\beta_{5,p}$ is the propagated absolute underflow floor for the
squared-norm computation, including flush-to-zero if applicable.

With outward $E=B_{32}+B_{64}$, promote the f32 squared norm $Q_{32}$
exactly to f64. The interval
$[\mathrm{down}(Q_{32}-E),\mathrm{up}(Q_{32}+E)]$ encloses the original
f64 result. Accept only if its upper bound is at most the **literal
original rounded** f64 squared reach; reject only if its lower bound is
greater. In the uncertainty shell execute the original f64 predicate
with the same operation graph. Nonclosing, nonfinite, overflow or image
representation cases use f64, without a new source refusal.

For Cartesian boxes, endpoint and image-shift error bounds propagate
through the 1-Lipschitz max-gap operation. Add the resulting length error
to the numerical box margin and round the squared threshold outward.
Extra internal prefilter survivors are safe; **extra final pair/image
membership is not**. Existing search and prune checkers remain exact.
Do not publish numerical padding as additional `classReaches`: that would
incorrectly credit an incompletely covered shell as validity skin.

The whole-search implementation uses cached Cartesian anchors, rather
than a fractional subtraction and cell multiplication for every pair.
The cache producer retains the original f64 wrap and Cartesian transform
before narrowing to f32. Three class-specialized searches use two warps
per block, direct descriptor-column endpoints, early backward-range
removal and shared first anchors. Direct column scans replace dependent
z binary searches. The native general image/subset iterator remains:
the prototype's nearest-image restriction cannot replace its covering
proof or change within-row output order. The sparse append, compaction,
entry masks and ownership are unchanged.

The cache uses dead group-planning storage only after group publication.
Every search attempt rebuilds Cartesian data for active and retained
hybrid references under the current cell; it does not retain an old-cell
Cartesian cache. The auxiliary Cartesian box slot has a separate f32
view, while reference and fractional-box views retain their f64 ABI.

| Operation | S4 | Whole search port |
|---|---|---|
| Reference fractions, wrap and cell | f64 | Same f64 |
| Cartesian anchor producer | f64 for boxes | Same f64 expression, then f32 cache |
| Query-image Cartesian shift | f64 | f32, with exact-integer and rounding certificate |
| Pair displacement and norm | f64 fractional subtraction, cell transform and norm | f32 Cartesian subtraction and norm; original f64 graph in uncertainty shell |
| Cartesian group boxes | f64 | f32 extrema with a certified conservative rejection threshold |
| Fractional ranges and image uniqueness | f64 | Same general f64 proof |
| Images, epochs, validity and certified reaches | f64/i64 | Unchanged |

Certificate setup evaluates positive error bounds outward in f64, one
representable step after each rounded operation. Thresholds narrow
outward to f32. Ordinary gradual-underflow arithmetic is required; PTX
must contain no flush-to-zero or approximate arithmetic on this path
before device validation. NVIDIA's [PTX ISA floating-point
instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/)
specify subnormal support for ordinary f32 arithmetic without `.ftz`.
Unsupported representation, nonfinite or overflow cases disable the
certificate and retain the literal f64 decision; no new source refusal
is introduced. Numerical envelopes are never physical skin.

This implementation has no performance conclusion before its gates and
the two-job measurement against S4. The exact all-images, boundary, reframed, triclinic,
incidence and growth fixtures remain gates. A coverage counterexample
stops work; no bound is fitted or widened after a failure.

The implemented certificate uses the equivalent reach-based implication
to avoid charging a global coordinate norm to each squared-norm threshold.
For the computed displacement $d_p$ and norm $Q_p$, the counted dot graph
satisfies $(1-g_p)\lVert d_p\rVert^2-b_p\le Q_p\le
(1+g_p)\lVert d_p\rVert^2+b_p$, where $g_p=\gamma_{3,p}$ and
$b_p=\beta_{5,p}$. The Cartesian producer, cast, image shift and original
absolute-frame graph give an outward
$\Delta=\delta_{32}+\delta_{64}\ge\lVert d_{32}-d_{64}\rVert$.
With $T$ the literal original rounded squared reach, form
$r_+=\mathrm{up}\sqrt{(T+b_{64})/(1-g_{64})}$ and
$r_-=\mathrm{down}\sqrt{\max(0,T-b_{64})/(1+g_{64})}$.
The outward thresholds are
$U=(1+g_{32})(r_++\Delta)^2+b_{32}$ and
$L=(1-g_{32})(r_--\Delta)^2-b_{32}$.
An original accepted pair has $Q_{32}\le U$, proving strict rejection
above its upward f32 conversion. Acceptance below the downward conversion
of $L$ implies $\lVert d_{64}\rVert\le r_-$ and hence $Q_{64}\le T$.
Fast acceptance requires both $r_--\Delta$ and $L$ strictly positive.
All other cases execute the original f64 graph, including zero when
positivity cannot be proved.

For boxes, let $\alpha$ bound the shared native f64 Cartesian producer's
error and $\delta_G$ bound the f32 max-gap error after endpoint narrowing
and image-shift rounding. Min/max is 1-Lipschitz per component; the bound
accounts for extrema coming from different points. A pair accepted by the
old predicate proves the conservative box threshold
$U_B=(1+g_{32})(r_++\delta_{64}+2\alpha+\delta_G)^2+b_{32}$.
Only a finite box norm strictly above its upward f32 conversion rejects.
This changes neither the physical reach nor the fractional image proof.


## Gates of the implementation

The final-kernel short-range oracle gate covers all six measured systems,
including both OPC water boxes. On the same exact frame, compare energy,
forces and molecular virial with the independent f64 brute force; project
force differences off the constraint-gradient span when comparing rules
(A) and (B). Retain raw force diagnostics. The existing declared physical
bounds remain unchanged; projection does not loosen energy or virial
criteria. This additional gate runs with the final implementation gates,
not as a prerequisite to the search diagnosis.


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
| Covering-search geometry and radial predicate | f32 | Cached Cartesian f32 with certified decisions and the literal f64 uncertainty shell; reference fractions and general image proof remain f64; source site cutoffs remain f32 |
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

### Radial force construction and site accumulation

Following the isolated anchor-cache experiment (discarded for no clear
measured gain), the derivative-window constructor recognizes energy that
depends on its site displacement only through the existing squared norm.
It differentiates that scalar expression and forms
$\mathbf F=-2wU'(r^2)\mathbf d$. General differentiation is unchanged;
unrecognized window expressions retain their existing derivative path.
The common pair-kernel-body interface supplies D94 radial extraction for
both ordinary pairs and template windows.

A proved site set with no formed-site contribution keeps three force
slots per water, including three-site models. Otherwise the constructed
force layout carries O/H1/H2/E interaction-site forces
through accumulation, then applies rule A once at each molecular write:
$\mathbf F_O \mathrel{+}= (1-2\alpha)\mathbf F_E$ and
$\mathbf F_{H_i} \mathrel{+}= \alpha\mathbf F_E$.
Site displacements, per-site cutoffs and the strain chain rule retain the
candidate image. The transpose does not use the normalized bisector.
Mixed pair arithmetic, site accumulators and the transpose are f32;
energy and virial accumulators remain f64 and coverage/pruning geometry
remains f64. Reassociation may change mixed bits; every existing physical
bound is unchanged. Deterministic mode retains the general path.
