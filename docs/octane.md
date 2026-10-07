# Forces Without a Neighbor List, by a Search in an Octree: Evaluation

Status: evaluated (2026-10-08), D[octane]; not adopted. Issue #201.
The keys in brackets are those of [references.md](references.md).

This document evaluates the method of [Toutouni2026] (OCTANE) for MDIR:
what it is, the theorems that an adoption would rest on, with proofs, the
analysis of the paper checked and where needed corrected, and the times
of a prototype against the neighbor structures of MDIR
([neighbors-m0.md](neighbors-m0.md), [groups-m1.md](groups-m1.md)).
The notation is that of Section 2 of the white paper
(`docs/paper/02-notation.md`); the symbols added here are in the table
below.

| Symbol | Meaning |
|---|---|
| $\ell$, $w_a = L_a / 2^\ell$ | The level of the leaves of the tree (the root is level 0), and the width of a leaf along axis $a$ |
| $u_a = 2 r_c / w_a$ | The width of a query box in widths of a leaf |
| $Q_i = \prod_a [x_{ia} - r_c,\ x_{ia} + r_c]$ | The query box of particle $i$ |
| $\ell_\text{c}(i)$, $c = \ell - \ell_\text{c}$ | The deepest level at which one node holds $Q_i$, and the levels that a search climbs from the leaf |
| $\Delta\mathbf x$ | The displacement of a particle in one step |
| $J$ | The levels between the leaf of a particle before a step and the deepest node that also holds its leaf after it |
| $\mu$, $M$, $b$ | The mean number of particles in a leaf, the capacity of a leaf, and the factor $M = \lceil b\mu\rceil$ |

## 1. The method

OCTANE stores no neighbor structure. The periodic cell is the root of a
region octree: a node of level $l$ is one of the $8^l$ boxes that halving
the cell $l$ times along each axis gives, and a leaf, at level $\ell$,
holds the numbers of its particles in a block of fixed capacity $M$.

- **Search and forces** (Algorithm 1 of the paper). One thread per
  particle $i$ starts at the leaf of $i$, climbs to the first ancestor
  whose box holds $Q_i$, and from there descends with a stack held in
  registers, skipping the nodes whose boxes do not meet $Q_i$. At a leaf it
  tests each particle $j \ne i$ against the cutoff and adds the force, the
  energy, and the virial of the pair to sums of its own. Each pair is
  computed twice, once for each particle, and no thread writes what
  another reads.
- **Update** (Algorithm 2). After the positions move, a particle that has
  left the box of its leaf is taken out of that leaf and put into the leaf
  of its new position. The tree is built once and never rebuilt.
- **Order.** The particles are sorted along a Morton or Hilbert curve
  every step, so that the threads of a block work on neighbors in space.

The paper analyzes how far a search climbs (its Section 3.2.1) and how
far an update climbs (3.3.1), under uniform density, and reports times
against the GPU package of LAMMPS for Lennard-Jones argon.

## 2. Completeness of the search

The paper states no theorem of correctness. MDIR requires one of every
structure that decides which pairs a loop sees (Section 4.1 of the white
paper): no pair within the cutoff may be left out. The method has no
skin and no test of validity, since it searches at the present positions
every step; what must hold instead is that the tree is consistent and
that the search is complete over it.

**Definitions.** The *box* of a node is the region it stands for, half
open: for the node $(l, \mathbf k)$, $B = \prod_a [k_a L_a/2^l,\ (k_a+1)L_a/2^l)$.
A tree is *consistent* with a configuration if every particle is stored
in exactly one place: the leaf whose box holds its position wrapped into
the cell, or a list that every search reads in full (the prototype keeps
there the particles that find their leaf full; the theorems below hold
with it, since a particle of the list is met by every search). The boxes of the leaves partition the cell, and the box of a
node is the union of the boxes of the leaves below it.

**Theorem 1 (descent).** Let the tree be consistent, $A$ a node, and
$Q$ a closed box. A descent from $A$ that enters exactly the children
whose boxes meet $Q$ reaches every leaf below $A$ that stores a particle
whose wrapped position is in $Q$.

*Proof.* By induction on the number of levels between $A$ and the
leaves. If $A$ is a leaf there is nothing to show. Otherwise let $j$ be
stored below $A$ with $\mathbf x_j \in Q$. By consistency $\mathbf x_j$
is in the box of its leaf, which is inside the box of exactly one child
$C$ of $A$; that box therefore meets $Q$, the descent enters $C$, and the
induction applies to $C$. $\blacksquare$

**Theorem 2 (climb).** Let the tree be consistent and let the box of $A$
contain $Q$. Then every particle whose wrapped position is in $Q$ is
stored below $A$, and the search of Theorem 1 from $A$ reaches it.

*Proof.* Such a particle is in the box of $A$, hence in the box of one
leaf below $A$, where consistency stores it. $\blacksquare$

**Theorem 3 (periodic cell).** Let $r_c < \min_a L_a / 2$ and let the
tree be consistent. For a lattice vector $\mathbf n \in \{-1, 0, 1\}^3$
write $Q_i^{\mathbf n} = (Q_i \cap (\text{cell} + \mathbf n\odot\mathbf L))
- \mathbf n\odot\mathbf L$ for the part of the query box that lies in the
image $\mathbf n$ of the cell, moved back into the cell; at most eight of
them are not empty. Searching each from a node whose box contains it
(the root always does) and testing every particle met with the distance
of the minimum image finds every $j \ne i$ with $r_{ij} \le r_c$, each
exactly once.

*Proof.* Let $r_{ij} \le r_c$. The image $\mathbf x_j + \mathbf
n\odot\mathbf L$ nearest to $\mathbf x_i$ is within $r_c$ of it along
every axis, so it lies in $Q_i$ and in the image $\mathbf n$ of the cell
with $n_a \in \{-1, 0, 1\}$, because $\mathbf x_i$ is in the cell and
$r_c < L_a$. The wrapped position $\mathbf x_j$ is then in
$Q_i^{\mathbf n}$, and Theorem 2 finds $j$ there. For uniqueness, the
parts $Q_i \cap (\text{cell} + \mathbf n\odot\mathbf L)$ are disjoint,
and $j$ is met in the part $\mathbf n$ only if the image $\mathbf x_j +
\mathbf n\odot\mathbf L$ is in $Q_i$. Two images of $j$ differ by at
least $L_a$ along some axis, and $Q_i$ is $2r_c < L_a$ wide, so $Q_i$
holds at most one image of $j$. $\blacksquare$

A box $Q_i$ that crosses a face of the cell is inside no node, not even
the root, without the images. Algorithm 1 says only "adjust query bounds
for PBC"; the theorem says what that adjustment must be. A search that
climbs to the root and tests the intersections of the boxes modulo the
cell is the same search written differently.

**Theorem 4 (updates).** If the tree is consistent before a step, and
after the positions move every particle whose wrapped position has left
the box of its leaf is removed from that leaf and stored in the leaf
whose box holds its new position, the tree is consistent after the step.

*Proof.* A particle that stays in the box of its leaf satisfies the
definition as before; one that was moved satisfies it by construction;
no particle is stored twice. $\blacksquare$

**Where completeness fails.** Two points of the paper's description
decide whether these theorems apply to it.

1. *The boxes that the descent tests.* The paper says that internal
   nodes keep "axis-aligned bounding boxes that enclose all atoms
   contained in their subtree" and that an update changes "only the old
   and new leaf nodes, while the rest of the tree remains unchanged".
   Theorem 1 uses that the box tested for a child contains every particle
   stored below it. The box of the region does, at all times, by
   consistency. A tight bounding box computed when the tree is built does
   not: a particle may move outside it without leaving the region of its
   leaf, the update then does nothing, and a later search prunes the node
   and misses the pair. Tight boxes are complete only if every move
   refits the boxes on the path to the root, which the update as
   described does not do.
2. *The capacity of a leaf.* Theorem 4 assumes that the insertion
   succeeds. The paper sets $M = \lceil b\,N / 8^\ell\rceil$ with $b$
   between 1.1 and 1.5 and does not say what happens to a particle that
   finds its leaf full (Section 6).

The prototype (Section 7) tests the boxes of the regions, handles the
images as in Theorem 3, and keeps the particles of a full leaf in the
list of the definition, counted. The theorems are in exact arithmetic;
the prototype computes in f32 and takes the query box wider by $10^{-5}$
of its half-width, for the reason that the build of the matrix widens
its reach ([neighbors-m0.md](neighbors-m0.md), Section 2.3): rounding
may then add a candidate beyond the box, which the test of the cutoff
drops, and cannot lose one within it.

## 3. How far a search climbs

**Theorem 5.** Let $\mathbf x_i$ be uniform in the periodic cell and
$\ell_\text{c}$ the deepest level at which one node holds $Q_i$, with
$\ell_\text{c} = -1$ if $Q_i$ crosses a face of the cell. Then for
$l \ge 0$

$$
P(\ell_\text{c} \ge l) = \prod_a \Bigl(1 - \frac{2^{l+1} r_c}{L_a}\Bigr)_+ ,
$$

where $(y)_+ = \max(y, 0)$.

*Proof.* Along axis $a$ the nodes of level $l$ are the intervals
$[k w, (k+1) w)$ with $w = L_a / 2^l$. The interval $[x - r_c, x + r_c]$
is inside one of them if and only if $t = x \bmod w$ lies in $[r_c,\ w -
r_c)$. For $x$ uniform on $[0, L_a)$, $t$ is uniform on $[0, w)$, so the
probability is $(1 - 2r_c/w)_+$. The three coordinates are independent.
A node of level $l$ lies inside one node of each lower level, so $Q_i$ is
held at level $l$ if and only if $\ell_\text{c} \ge l$. $\blacksquare$

**Corollary 6 (the climb is bounded).** With leaves of level $\ell$ and
$u_a = 2r_c/w_a \le 1$, the levels climbed, $c = \ell - \ell_\text{c}$,
satisfy $P(c > j) = 1 - \prod_a (1 - u_a 2^{-j})$ for $0 \le j \le \ell$,
and in a cubic cell

$$
E[c] = \sum_{j=0}^{\ell}\bigl[1 - (1 - u\,2^{-j})^3\bigr]
\;<\; 6u - 4u^2 + \tfrac87 u^3 \;\le\; \tfrac{22}{7} .
$$

*Proof.* $P(c > j) = 1 - P(\ell_\text{c} \ge \ell - j)$ and $2^{\ell - j
+ 1} r_c / L = u\,2^{-j}$. Expanding the cube and summing the three
geometric series over all $j \ge 0$ gives $3u\cdot 2 - 3u^2\cdot\tfrac43 +
u^3\cdot\tfrac87$, which increases in $u$ on $[0, 1]$. $\blacksquare$

With leaves at least $2r_c$ wide the mean climb is therefore below 3.15
levels whatever the number of particles: the paper's claim of a constant
expected climb is correct. For $u = 1$, $\tfrac12$, $\tfrac14$ the bound
is 3.14, 2.14, and 1.27 levels. Narrower leaves, $u > 1$, cannot hold a
query box: the first $\lceil\log_2 u\rceil$ levels are climbed by every
search, and the corollary applies from the level above them, so that
$E[c] < \lceil\log_2 u\rceil + \tfrac{22}{7}$. The leaves that timed
best in Section 7 have $u$ from 1.3 to 4.1, where each halving of the
leaves adds one level to the climb, as measured (2.17, 3.17, 4.17
levels for leaves of 32.3, 16.1, and 8.07 Å in the vapor, with the faces
of the cell left out).

**The analysis of the paper** (its Eq. 2 to 7, Tables 1 and 2).

- *Confirmed.* The paper counts the $2^l - 1$ planes of level $l$
  inside the cell and takes $P_\text{safe}(l) = (1 - 2r(2^l - 1)/D)^3$,
  the critical level $l_m = \lfloor\log_2(D/2r + 1)\rfloor$, and
  $E[l^*] = 1 + \sum_{k=1}^{l_m} P_\text{safe}(k)$ for the lowest level
  $l^*$ of a plane that the query crosses. Its Table 2 follows from these
  formulas to the digits printed ($E[l_m - l^*]$ from 1.95 to 2.11 for
  $D/r$ from 520 to 18640; `octane_analysis.py`).
- *The periodic boundary.* The count $2^l - 1$ leaves out the faces of
  the cell. That is right for a cell with walls. In a periodic cell,
  which is what the paper simulates, a query that crosses a face needs
  the images (Theorem 3) and is inside no node; the planes of level $l$
  are then $2^l$ along an axis, and Theorem 5 replaces Eq. 3. The
  difference is the term $l = 0$: a fraction $1 - \prod_a(1 - 2r_c/L_a)$
  of the particles search from the root, 12% in a cell of 48 cutoffs
  (the smallest system of the paper) and 2.7% in one of 220.
- *The overlap of the strips.* Eq. 2 subtracts a strip of $2r$ for each
  plane, which is exact only while the strips do not overlap, $D/2^l \ge
  2r$. The critical level admits $D/2^{l_m}$ down to $2r(1 - 2^{-l_m})$;
  in that window Eq. 2 is off by less than $2r/D$.
- *What the level means.* The smallest node that holds the query is at
  level $l^* - 1$, so "two levels above the critical level" for $l^*$ is
  three for the node where the descent starts, as the paper says in the
  next sentence, and as Corollary 6 gives for $u$ near 1.
- *The sizes.* Tables 1 and 2 are computed for $D/r$ from 520 to 18640.
  The systems of the paper's Table 3 have $D/r$ from 47 to 220 (boxes of
  480 to 2230 Å, $r = 3\sigma = 10.1$ Å): $l_m$ is 4 to 6 there, and the
  share of searches from the root is the one above.

## 4. What a search costs

**Proposition 7.** Let the density be uniform, $\rho = N/V$, and
$\mathbf x_i$ uniform. The descent of particle $i$ reaches on average
$\prod_a (1 + u_a)$ leaves, which hold $\rho\prod_a (w_a + 2r_c)$
particles, and enters $\prod_a(1 + u_a 2^{-j})$ nodes $j$ levels above
the leaves, below the node where it starts.

*Proof.* Along axis $a$ the leaves that an interval of length $2r_c$
meets are one more than the planes of spacing $w_a$ that it crosses, and
an interval placed uniformly crosses $2r_c/w_a = u_a$ of them on average,
for any $u_a$. The coordinates are independent, so the mean of the
product over the axes is the product of the means. The leaves reached
are disjoint boxes of volume $\prod_a w_a$. The same count with the
spacing $2^j w_a$ gives the nodes of the levels above. $\blacksquare$

The pairs within the cutoff of a particle number $\tfrac{4\pi}{3}\rho
r_c^3$. In a cubic cell a search therefore tests

$$
\kappa(w) = \frac{3}{4\pi}\Bigl(2 + \frac{w}{r_c}\Bigr)^3
$$

candidates for each pair it computes: at least $6/\pi = 1.91$ (the cube
around the sphere, for leaves that hold less than one particle), 3.7 for
$w = r_c/2$, 6.4 for $w = r_c$, and 15.3 for $w = 2r_c$, the leaves of
the critical level that the paper's analysis assumes. A loop of MDIR
over a list of reach $R = r_c + s$ tests $(R/r_c)^3$ entries for each
pair within the cutoff, 1.42 at 9 Å over 8 Å, twice for each pair in the
matrix and once in the groups; it pays a build, which is one search of
this kind, once in the steps between builds. The method thus exchanges a
build every 4 to 20 steps for a search every step, and a loop over a
sphere for one over a box of leaves. Section 7 measures the exchange.

## 5. How far an update climbs

**Theorem 8.** Let the position $\mathbf x$ of a particle be uniform in
the periodic cell and independent of its displacement $\Delta\mathbf x$
in a step, as it is in a homogeneous system at equilibrium. Let $J$ be
the number of levels between the leaf of $\mathbf x$ and the deepest
node that holds both that leaf and the leaf of $\mathbf x + \Delta\mathbf
x$ ($J = 0$ if the leaf is the same). Then for $0 \le k < \ell$

$$
P(J \le k) = E\Bigl[\prod_a \Bigl(1 - \frac{\lvert\Delta x_a\rvert}{2^k w_a}\Bigr)_+\Bigr],
$$

and $P(J \le \ell) = 1$.

*Proof.* $J \le k$ if and only if both positions are in one node of level
$\ell - k$, that is, if the particle crosses no plane of spacing $h_a =
2^k w_a$ along any axis; the faces of the cell are among these planes
for every $k < \ell$. Given $\Delta\mathbf x$, the particle crosses such a
plane along $a$ if and only if $x_a \bmod h_a$ lies within $\lvert\Delta
x_a\rvert$ of the plane ahead, an event of probability
$\min(\lvert\Delta x_a\rvert/h_a, 1)$ for $x_a$ uniform; the three
coordinates of $\mathbf x$ are independent of one another and of
$\Delta\mathbf x$. $\blacksquare$

**Corollary 9.** If $\lvert\Delta x_a\rvert \le w$ in a cubic cell and
the law of $\Delta\mathbf x$ is isotropic, with $m_1 = E\lvert\Delta
x_a\rvert$, $m_2 = E\lvert\Delta x_a\,\Delta x_b\rvert$, and $m_3 =
E\lvert\Delta x_a\,\Delta x_b\,\Delta x_c\rvert$ for distinct axes,

$$
P(J > k) = \frac{3 m_1}{2^k w} - \frac{3 m_2}{4^k w^2} + \frac{m_3}{8^k w^3},
\qquad
E[J] < \frac{6 m_1}{w} - \frac{4 m_2}{w^2} + \frac{8 m_3}{7 w^3} .
$$

In particular $P(J = k+1)/P(J = k) \to \tfrac12$, the ratio that the
paper assumes.

**The analysis of the paper** (its Eq. 9 to 13). The paper takes the
displacement uniform in a ball of radius $p$ and a leaf of width $d =
2p$. For that law $m_1 = 3p/8$, $m_2 = 2p^2/(5\pi)$, $m_3 = p^3/(8\pi)$
in three dimensions, and $m_1 = 4p/(3\pi)$, $m_2 = p^2/(2\pi)$ in two.

| Quantity | Paper | Corollary 9 | Monte Carlo ($4\times10^6$) |
|---|---|---|---|
| $P_\text{exit} = P(J > 0)$, two dimensions | $29/(24\pi) = 0.3846$ | $\tfrac{4}{3\pi} - \tfrac{1}{8\pi} = \tfrac{29}{24\pi} = 0.38462$ | 0.38436 |
| $P_\text{exit}$, three dimensions | 0.4237, "no closed form" | $\tfrac{9}{16} - \tfrac{3}{10\pi} + \tfrac{1}{64\pi} = 0.47198$ | 0.47192 |
| $P(J = 1)$, to a sibling | 0.1937 | 0.21398 | 0.21394 |
| $P(J > 1)$ | 0.2300 | 0.25800 | 0.25798 |
| $E[J]$ | $4 \times 0.1937 \to 0.782$, "less than one level" | 1.00336 | 1.00304 |

- The value in two dimensions is confirmed. In three dimensions there is
  a closed form, and the paper's numbers do not agree with it or with a
  direct simulation of the paper's own model: the probability of leaving
  the leaf is 0.472, not 0.424, and the mean climb is 1.003 levels, not
  0.78.
- The paper's numbers do not agree with one another either: a geometric
  law with $P(J = 1) = 0.1937$ and the ratio $\tfrac12$ gives $P(J \ge 1)
  = 0.387$, not the 0.4237 of its Eq. 10, and $P(J > 1) = 0.194$, not the
  0.2300 of its Eq. 12.
- The model itself is far from a step of dynamics. With $d = 2p$ a
  particle moves up to half a leaf in a step. In a step $E\lvert\Delta
  x_a\rvert = \Delta t\sqrt{2k_BT/(\pi m)}$: 0.0013 Å for argon at 120 K
  and 1 fs, 0.006 Å for an oxygen and 0.025 Å for a hydrogen at 300 K and
  2 fs. With leaves of 8 to 10 Å, $P_\text{exit} \approx 3m_1/w$ is $4
  \times 10^{-4}$ to $10^{-2}$ and $E[J] \approx 6m_1/w$ is $8 \times
  10^{-4}$ to $2 \times 10^{-2}$. The conclusion of the paper, that
  updates are cheap, holds, by three orders of magnitude more than its
  analysis says; the numbers of the analysis do not.

## 6. The capacity of a leaf

**Proposition 10.** If the positions are independent and uniform, the
number of particles in a leaf is binomial with mean $\mu = N/8^\ell$,
and for $b > 1$

$$
P(\text{a leaf holds} \ge b\mu) \le \exp\bigl(-\mu\,(b\ln b - b + 1)\bigr) .
$$

*Proof.* The Chernoff bound for a binomial variable, which is the bound
of the Poisson law of the same mean. $\blacksquare$

The exponent is $0.108\,\mu$ for $b = 1.5$ and $0.0048\,\mu$ for $b =
1.1$. For no leaf of $10^6$ particles to overflow with a probability of
0.999, the bound asks for $\mu \ge 146$ at $b = 1.5$ and $\mu \ge 2653$
at $b = 1.1$. The bound is not tight; the exact Poisson tail gives, for $b = 1.5$, 6.4% of the
leaves over capacity at $\mu = 8$, 0.31% at $\mu = 32$, and $5 \times
10^{-8}$ at $\mu = 128$, and for $b = 1.1$ more than 10% at every $\mu$
up to 128 (`scripts/experiments/neighbor-structures/octane_analysis.py`). Independent positions are the ideal
gas, which the vapor of the paper's benchmark (Section 7) is close to. A
liquid fluctuates less, by the factor $\rho k_BT\kappa_T$ in a large
volume, but a leaf is not a large volume. With the factors the paper
names, either the leaves are large, and then a search tests many times
the pairs it computes (Proposition 7), or leaves overflow and a
particle is lost, which the theorems of Section 2 exclude. A structure
for MDIR would have to grow, as the matrix and the groups do.

## 7. Measurements

A standalone prototype, `scripts/experiments/neighbor-structures/octree.cu`,
implements the method (T) on one RTX 3090 with the pair kernels of the
other prototypes there, in f32, and beside it the same idea without the
tree (G): a counting sort into cells every step and a search of the cells
that computes the forces, as a build of the matrix searches. Both are set
against the loops and the builds of the matrix (M) and of the groups (Gr)
of MDIR at their reach, run in the same session. The README of that
directory has every table, the sweeps, and how the ambiguities of the
paper were resolved; the numbers are in
`results/2026-10-08-octree-NVIDIA-GeForce-RTX-3090.csv`.

**Systems.** JAC (23,558 atoms) and Cellulose (408,609), at the density
of water, with a cutoff of 8 Å; M and Gr at a reach of 9 Å with the
builds every 4.4 and 4.0 steps that MDIR measures. Argon, $10^6$ atoms,
cutoff $3\sigma = 10.1$ Å, as a liquid (0.021 atoms/Å³) and at the
density of the paper, $9.07 \times 10^{-4}$ atoms/Å³, which its Table 3
gives (a box of 1033 Å): a vapor, with 3.7 pairs within the cutoff of an
atom; M and Gr at a reach of 11.1 Å with a build every 58 steps, counted
in the dynamics of the prototype.

**Exactness.** Against all pairs in f64, on JAC and on argon in a small
cell where 76% of the query boxes cross a face: T at every leaf level
and G at every cell size give the same set of pairs, none missing and
none added, with leaves over their capacity and after 25 partial updates
with 16,000 to 64,000 changes of leaf; the forces agree to $7.5 \times
10^{-7}$ relative. Theorems 1 to 4 hold for the prototype, which tests
the boxes of the regions, takes the images of Theorem 3, and keeps the
particles of a full leaf in a list that every search scans.

**Time a step** (µs; the best leaf level and cell size of each).

| | JAC | Cellulose | Argon, liquid | Argon, density of the paper |
|---|---|---|---|---|
| Pairs within the cutoff of an atom | 209 | 221 | 90 | 3.7 |
| T: search and forces | 754 | 7032 | 7033 | 2187 |
| T: update of the tree | 18 | 78 | 132 | 141 |
| T: sort along the Morton curve | 81 | 294 | 446 | 446 |
| **T, a step** | **853** | **7404** | **7611** | **2774** |
| G: sort into cells, search and forces | 31 + 235 | 65 + 1997 | 120 + 1512 | 145 + 260 |
| **G, a step** | **266** | **2062** | **1632** | **405** |
| M: loop + build over the interval | 76 + 81 | 1235 + 1650 | 1051 + 176 | 262 + 140 |
| **M, a step** | **157** | **2885** | **1226** | **402** |
| Gr: loop + build over the interval | 70 + 31 | 1094 + 392 | 1139 + 42 | 201 + 65 |
| **Gr, a step** | **101** | **1486** | **1181** | **266** |
| T over Gr | 8.4 | 5.0 | 6.4 | 10.4 |
| T with the test of the excluded pairs | +24% | +23% | | |

**What the numbers show.**

1. *A step of the method takes 5 to 10 times a step of the groups of
   MDIR and 2.6 to 6.9 times a step of the matrix*, on every system,
   the vapor of the paper included, where the programs of MDIR were not
   tuned (a build of the matrix there takes 8 ms with rows of 736
   entries for 5 neighbors).
2. *The tree costs more than the absence of a list.* The search of the
   cells, G, is 3.2 to 6.8 times faster than T: a candidate of T is an
   index loaded from a block and a position loaded through it, and its
   threads diverge in the traversal, where G reads positions in runs. G
   is still 1.4 to 2.6 times a step of Gr.
3. *The count of candidates is that of Proposition 7.* On JAC (a cube
   of 62.23 Å, 0.0978 atoms/Å³) with leaves of 3.89 Å the descent meets
   768 candidates an atom, and $\rho\prod_a(w_a + 2r_c)$ gives 769; on
   Cellulose ($259.2 \times 124.6 \times 123.5$ Å, 0.1025 atoms/Å³) with
   leaves of $8.10 \times 7.79 \times 7.72$ Å, 1403 against 1393. Leaves of half the cutoff thus test 3.7 candidates for
   each pair (2.5 with a test of the sphere against each node, which the
   prototype adds); a list of 9 Å tests 1.42.
4. *The levels are those of Theorem 5.* For uniform positions in the
   cell of the vapor ($L/r_c = 102.1$) the measured shares of the levels
   $-1$ and 0 together, 1, 2, 3, 4, 5 are 0.113, 0.104, 0.184, 0.275,
   0.271, 0.052, and the theorem gives 0.113, 0.104, 0.183, 0.276, 0.272,
   0.052. With the faces left out, as in the paper, the mean level is
   2.83, which is its Eq. 7 at this size; but $l_m - E[l^*]$ is then
   1.17, not the "about 2" of its Table 2, whose sizes all have $D/2r +
   1$ just above a power of two. Where the cell is a few cutoffs wide
   most searches start at the root: 89% on JAC, 32% on Cellulose.
5. *The updates are those of Corollary 9.* In the dynamics of argon
   0.033% of the atoms of the liquid leave their leaf of 11.3 Å in a
   step and 0.024% of the vapor theirs of 16.1 Å; $3m_1/w$ with $m_1 =
   \Delta t\sqrt{2k_BT/(\pi m)}$ gives 0.033% and 0.023%. Of those that
   leave, 0.50, 0.25, 0.125, ... climb 1, 2, 3, ... levels, the ratio
   $\tfrac12$ of the corollary. The update takes 2% of the step.
6. *The capacity of the paper overflows* (Proposition 10): with $b =
   1.5$, 27,000 of $10^6$ atoms of the vapor find their leaf of 16 Å
   full (the fullest holds 13, the capacity is 6), and 137 to 1300 on
   the other systems at the leaf levels timed. The times above are with
   the capacity of the fullest leaf.
7. *The memory is the gain.* T holds 0.8 MB on JAC and 9.4 MB on
   Cellulose beyond the particles, where the groups hold 4.0 and 71 MB
   and the matrix 28 and 516 MB.

The prototype was not profiled, and its kernels are a first version; the
programs of MDIR it is set against were tuned over several days. The
ratio of 3 to 7 between T and G is the part most likely to shrink with
work; that G, which shares nothing of the tree, stays behind the groups
on every system is the part that work on the tree cannot change.

**Against the times of the paper.** The paper reports speedups of 5.5
to 16 over the GPU package of LAMMPS on another device, on the vapor
alone. They are not comparable with the table above, and say nothing of
a liquid: at 3.7 pairs an atom a step is bound by the traversal and by
the launches, not by pairs. On that system the prototype of the method
takes 2.8 ms a step and the groups of MDIR, untuned, 0.27 ms.

## 8. Verdict

Not worth adopting.

- **Rate.** The method is 5 to 10 times slower a step than the groups
  of MDIR on a liquid, on water with a solute, and on the vapor of the
  paper itself. The reason is structural (Proposition 7, confirmed by
  the counts): a search every step tests 2.5 to 15 candidates for each
  pair in a box of leaves, where a list tests 1.4 in a sphere and pays
  for its search once in 4 to 58 steps; and it computes each pair twice.
  The exact test of validity of MDIR (Section 4.1 of the white paper)
  already gives what the absence of a list promises, no pair left out
  and no estimate, without the search every step.
- **What MDIR would add to it.** The excluded pairs, which the method
  must test for every candidate every step (+23%), the images for more
  than half of the atoms of a cell a few cutoffs wide, a capacity that
  grows, and boxes that stay valid under updates: each is needed for
  the theorems of Section 2 and none is in the paper.
- **The analysis of the paper.** Its claim of a constant expected climb
  is right and is proved here with a bound; its numbers for the update
  in three dimensions are wrong, its model of the update is three
  orders of magnitude from a step of dynamics, and it leaves out the
  periodic boundary, the overflow of a leaf, and any statement of
  correctness.
- **What is worth keeping.** The memory: 5 to 55 times less than the
  groups and the matrix. No system of the suite is bound by memory on
  24 GB (the groups of Cellulose take 71 MB), so this buys nothing now;
  it would matter for $10^8$ particles on one device, where the search
  of the cells (G), not the tree, would be the form to take. And the
  measured cost of the excluded pairs in a search, 8 to 24%, is a number
  that the build of MDIR can use.

The prototype and its results stay in
`scripts/experiments/neighbor-structures` as the record, as those of the
tiles (D82) do.
