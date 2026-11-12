# D[rigid-groups]

Recorded at its merge.

**The molecules of a topology that move as rigid bodies are found by two rules stated on the constraints and the site definitions alone, R1 (a molecule is a rigid group when a constraint fixes the distance of every pair of its massive atoms, its frame) and R2 (each massless site of a rigid group is a combination of the frame atoms with constant weights that sum to 1), and sorted into kinds by exact equality of their numbers; `mdir check` reports the kinds and counts the molecules left out by reason, and no run reads the result.** Issue #855, the foundation of #795: a consumer that specializes a kind (a pair kernel for the waters of a system) asks this layer for the kinds and states its own conditions on one, instead of recognizing a chemistry. The derivations are in [constraints.md](../constraints.md), Section 3.

## The rules

A molecule is a connected component of the graph whose edges are the bonds, the constrained distances (the three of a group of SETTLE; the center and each hydrogen of a group of SHAKE), and the pairs of a site with the three atoms that place it. Its frame is the set of its particles with a mass. No atomic number, atom name, residue name, or order of elements is read; a new molecule kind needs no line of C++.

**R1.** A molecule is a rigid group when every pair of atoms of its frame has a constrained distance. All pair distances fixed means that two configurations that satisfy the constraints are congruent, so the condition is sufficient for any frame (for four or more atoms out of a plane up to a reflection, which a continuous motion does not reach). It is not necessary in general: $3m - 6$ distances can hold $m \ge 3$ atoms. For the records that `validateTopology` accepts today it is exact. Groups share no particle; a group of SETTLE has all 3 distances of 3 atoms; a group of SHAKE has the $h$ distances of a center to $h = 1, 2, 3$ hydrogens, which leaves the angles free unless $h = 1$. The rigid groups of today are therefore single atoms (nothing to hold), a group of SHAKE with one hydrogen that is the whole molecule, and a group of SETTLE that is the whole molecule. A molecule that fails is counted with `flexible_frame`, "a pair of massive atoms without a constrained distance". A constraint record that holds a frame with fewer than all its distances, or a frame of four atoms out of a plane, belongs to the decision that adds it.

Single atoms are rigid groups: the rule does not single them out, and a consumer that wants molecules of three atoms asks for the size.

**R2.** A linear site $\mathbf x_s = \mathbf x_i + a\,\mathbf d_{ji} + b\,\mathbf d_{ki}$ has the weights $(1 - a - b,\ a,\ b)$ on any frame. The extra point of Amber, $\mathbf x_s = \mathbf x_i + a\,(\hat{\mathbf u} + \hat{\mathbf v})/\lVert\hat{\mathbf u} + \hat{\mathbf v}\rVert$, is not linear in the positions; on a frame that holds $d_{ij}$, $d_{ik}$, and $d_{jk}$ the law of cosines gives $d_{ij} d_{ik}\lVert\hat{\mathbf u} + \hat{\mathbf v}\rVert^2 = (d_{ij} + d_{ik})^2 - d_{jk}^2 \equiv Q$ and

$$w_j = \frac{a}{\sqrt Q}\sqrt{\frac{d_{ik}}{d_{ij}}}, \qquad w_k = \frac{a}{\sqrt Q}\sqrt{\frac{d_{ij}}{d_{ik}}}, \qquad w_i = 1 - w_j - w_k ,$$

exactly, under the condition that the frame is rigid (R1 holds, so the three distances are constraints). For $d_{ij} = d_{ik} = d$ it is $a/(2\sqrt{d^2 - d_{jk}^2/4})$. Off the rigid frame, with the lengths held within a relative $\tau$, the combination and the bisector differ in first order, by at most $a\tau(2/S + 2(4 - S^2)/S^2)$ for two equal lengths, $S = \lVert\hat{\mathbf u} + \hat{\mathbf v}\rVert$: $4.85\,a\tau$ for OPC. A consumer that places sites by the weights agrees with the engine's placement to the tolerance of the constraints, not bitwise.

A site of any other form, or a bisector without a direction ($Q \le 0$), is not assumed to be a combination: its molecule is counted with `site_not_constant_weight`, "a site that is no constant-weight combination of the frame".

## Kinds

Two rigid groups are one kind when, with the particles of each in the order of their numbers, they have the same size, the same places of frame atoms, the same type and charge at each place, the same frame distances, and the same site weights. Masses are not compared.

**No tolerance.** The values are compared as f64. Copies of a molecule take their parameters from one text through the same arithmetic, so equal inputs give equal values; a tolerance is not transitive, which would make the kinds depend on the order of the molecules; and a consumer that replaces per-particle values by those of the kind changes no number of a run only under equality. Two descriptions of one site are one kind only where their weights are equal numbers; in the last place apart they are two kinds, each correct. The alternative, a relative tolerance of a few units in the last place, was not taken for the second and third reasons.

**The record** (`RigidKind`, `include/mdir/Driver/RigidGroups.h`): the particles of a copy; the places of the frame; charges and types by place; the frame distances; each site's place and weights over the frame; the particles of all copies, in the order of their first particle. Kinds are ordered by count, then by first particle.

**Facts, not refusals.** What a consumer of #795's kind asked of a water (that its particles are consecutive, that all its own pairs are excluded, that no exclusion leaves the molecule) are counts of copies on the kind: `scattered`, `incompleteExclusions`, `outsideExclusions`. They decide neither rigidity nor the kind. Where Lennard-Jones sits is read from the types. The conditions of a run (LJ-PME, custom parameters, a free-energy selection) are not properties of the topology's molecules and stay with the consumer.

## What is not reachable from a file today

No reader gives a site outside the two forms (the frames of lone pairs and of TIP5P of Amber, the other virtual sites of GROMACS, and the lone pairs of CHARMM are refused when the topology is read; `validateTopology` refuses another kind), and a bisector without a direction is refused as lengths of SETTLE. A water of five sites with its sites out of the plane can therefore not be given to `mdir check`; the reason `site_not_constant_weight` is tested on a topology made in the unit test, with a site of an unknown form on a rigid frame, and is there for the reader that adds such sites. Those sites are constant combinations of the frame atoms and the cross product of two bonds, which the record of this decision does not hold. A water of five sites whose sites are linear, in the plane, is a rigid group under R2.

## The report

`mdir check` prints, after the degrees of freedom, the line `rigid groups:` with the number of rigid groups, of molecules, and of kinds, then for each of the 8 most populous kinds its count, sizes, types, charges, frame distances in Ang, site weights, and the facts that are not zero, and the molecules left out by reason. `--json` has all kinds under `system.topology.rigid_groups` ([driver.md](../driver.md), Section 1.4). No key of the control file is added, and no input runs differently.

## Validation

`test/Driver/rigid-groups.test` (made topologies, in f64) and `test/Driver/check-rigid-groups.test` (files of GROMACS and Amber); the table is in [constraints.md](../constraints.md), Section 3.4. The numeric check of R2 compares $\sum_a w_a\mathbf x_a$ with the placement of the engine's form on 20,000 random frames of three different lengths and on 20,000 rotated and shifted copies of a kind, each coordinate within a bound derived from the rounding of f64; the largest errors are 0.08 to 0.23 of the bounds. Exchanging $w_j$ and $w_k$, inverting the ratio of the lengths, changing a sign of $w_j$, of $w_i$, or in $Q$, putting the weights on exchanged frame atoms, and exchanging $a$ and $b$ of a linear site were each tried once, and each fails the test.
