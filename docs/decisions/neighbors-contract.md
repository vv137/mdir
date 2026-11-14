# D[neighbors-contract]

Recorded at its merge.

**The layout contract of a neighbor structure is a parameter of `!mdrt.neighbors`. It states what the build guarantees about the multiplicity of the lists (the traversal, the images, the encoding of the excluded pairs) and their organization (`none`, `grouped_by(G)`, `sorted_within(G, K)`, `stable(G, K)`, the values of D409). Each consumer states its requirement from its policy and its algebra, and its verifier refuses a structure whose guarantee does not meet it. A build may state no more than its kind and its images guarantee. Cost choices stay attributes of the build. `md_exec.weaken_neighbors` weakens a guarantee, and nothing emits it yet. No program changes.** Issue #881, step 1 of #877 (the maintainer's rulings of 2026-11-14), part of #558.

## Why

The facts that decide whether a sort of a neighbor build can be removed
are known in `md_exec`, where `md-exec-choose-neighbors` picks the
structure and `pair_for` states its policy and its algebra (D301). The
lowering drops them: a consumer of `md_alg` sees raw reads, and D409's
analysis therefore finds every requirement to be `layout` (#869, #877).
Before a consumer can move into `md_alg` and state its requirement there
(#877, step 2), the requirement and the guarantee need a home in the IR.

A structure does not reach its consumers from its build directly. It flows
through `refresh_neighbors`, the values that a loop carries, and the
results of an `scf.if`. With the guarantee in the type, the type rules of
those ops keep it, and each consumer compares its requirement with the
type of its operand (the maintainer's ruling 1 on #877). The alternative,
an attribute on the build, would need a dataflow analysis at every
consumer.

The design follows the required and delivered physical properties of query
optimizers, where a sort is an enforcer placed only where the delivered
property does not meet the required one
[[Graefe1993]](../references.md#graefe1993). Grouping is a property apart
from ordering, and weaker [[Neumann2004]](../references.md#neumann2004).
The sparse tensor types of MLIR state properties of a collection's levels
in the type and change them only through explicit conversion ops
[[Bik2022]](../references.md#bik2022). Cabana makes full and half lists
and their layout parameters of a list's type
[[Slattery2022]](../references.md#slattery2022).

## What each structure guarantees

| Structure | Traversal | Images | Exclusions | Organization | Where |
|---|---|---|---|---|---|
| MATRIX, host | `directed` | `minimum` | removed: `emitExclusionFilter` compacts each row in its order | `stable(owner, search)`: the cells in the order of the search, each cell by index | `lib/Runtime/Templates/NeighborsMatrix.mlir` (its header); `MDExecToLoops.cpp`; `MDExecKernels.cpp`, `emitExclusionFilter` |
| MATRIX, device | `directed` | `minimum` | masked: an excluded pair is entered as the particle itself, which every loop over a matrix skips | `stable(owner, search)`: the rows by place in the order of the cells (D86), a ballot giving each neighbor its slot in the order a thread would find it, each cell sorted by index after binning | `NeighborsMatrixGPU.mlir` (its header); `MDExecToGPU.cpp` (the search's exclusions); `MDExecKernels.cpp`, `emitPairKernel`; `HostPairLanes.cpp` |
| GROUPS | `unique` | `minimum`: each entry carries the shift into the frame of its group (I4) | removed: no bit is set for an excluded pair (I1) | `stable(group, search)`: a warp builds the list of a group from its cells in a fixed order (I2); which blocks of the pool a list takes is free, its order is not | [neighbor-groups.md](../neighbor-groups.md), Section 1.3; `GroupsBuild.cpp` |
| Dual list (D114) | as GROUPS | as GROUPS | as GROUPS | as GROUPS: the pruning keeps the order of the outer list and adds none (I5, I6) | [neighbor-groups.md](../neighbor-groups.md), Section 6.2, "The order" |
| Explicit images (D290) | `directed` | `explicit`: every pair within the reach in the plain difference, and possibly more, which the loop's cutoff removes | none: `build_neighbors` refuses `exclude` with them | as MATRIX on its target | `BuildNeighborsOp::verify`; `MDExecToLoops.cpp` (the refresh with `policy(always)`) |
| Ragged rows (D314) | | | | `stable(row, item)`: each row sorted ascending | `RaggedRows.h`, `emitDeviceRaggedRows`; this is `!mdrt.incidence`, not `!mdrt.neighbors`, and is outside this step |

The type is made in `convert-md-to-md-exec`, before the target is known.
The matrix's exclusions are removed on the host and marked on a device,
and every consumer of a matrix skips an entry equal to its center
(`other != central` in `emitPairKernel` and in the lanes of the host,
D402). So a matrix states the weaker value, `masked`, on both targets, and
`masked` is below `removed`. Every structure of `main` guarantees `stable`
on its own grouping. What separates them is the multiplicity and the
grouping key.

## The type

```mlir
!mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
!mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>
!mdrt.neighbors<@dd.d0.local, directed, explicit, masked, stable(owner, search)>
```

The four words are always spelled, and always printed (the maintainer's
ruling 1 on #882). A bare type that meant the matrix's guarantee would be
an unchecked claim: a new build that emitted it would get that guarantee
silently. The tests, the op descriptions, and the documents spelled the
bare type 378 times; a commit of its own rewrites each to the matrix's
guarantee, the type it parsed to, and changes nothing else.

The four words are positional. The axes are ordered by strength, and a
guarantee $g$ meets a requirement $r$ when $r \sqsubseteq g$ on every axis:

- **traversal**: `directed` or `unique`. Two values are comparable only
  when they are equal.
- **images**: `minimum` or `explicit`. The same rule holds.
- **exclusions**: `masked` $\sqsubset$ `removed`. A consumer that skips
  marked entries reads a list without them correctly. A consumer that
  skips nothing needs `removed`.
- **organization**: `none` $\sqsubset$ `grouped_by(G)` $\sqsubset$
  `sorted_within(G, K)` $\sqsubset$ `stable(G, K)`. Two values with keys
  are comparable only when their keys agree, as in D409. `G` is `owner`
  (the row of one center) or `group` (a group of places of a build of
  groups). `K` is `item` (the partner) or `search` (the order in which the
  build's search finds the entries). Both keys are total, so
  `sorted_within(G, K)` is normalized to `stable(G, K)` on parsing, as
  D409 normalizes `sorted_within(G, item)`. The type's `none` is D409's
  `membership`: a list always holds its pairs.

`NeighborsContract` (`include/mdir/Dialect/MDRT/NeighborsContract.h`)
holds the value, its order, and the join of two organizations with the
same keys.

The kind, the width, the order of the places of a build of groups, the
group size, and `cell_width` stay attributes of the build. They are cost
choices. In the type, each change of the cost model would change types and
make the branches of a program disagree (#877, correction 5).

## Who sets it

- `convert-md-to-md-exec` makes every structure with the matrix's
  guarantee. The executor of `md_dist` makes the rows of a domain with
  `explicit` images (`md_exec::getBuildType`).
- `md-exec-choose-neighbors` gives the groups' guarantee to each structure
  it turns into groups. It retypes every value of the structure: the
  build or `empty_neighbors`, the refreshes, and the initial values,
  iteration arguments, yielded values, and results of the `scf.for` ops
  that carry it. A structure keeps the matrix, with all its beginnings,
  in two cases:
  - its values reach an op whose types the pass does not follow, such as
    an `scf.if`, a call, or a return;
  - one of its beginnings has no loop over pairs.

  No program of `main` has such a structure: the lowered programs are
  identical (below).
- The other passes copy the type: `md-exec-reuse-neighbors` (the empty
  structure and the refresh that replace a build) and
  `md-exec-assign-storage` (the storage and its refreshes).

**The builds' own check.** `build_neighbors` and `empty_neighbors` refuse
a result type that states more than their kind and images guarantee.
`refresh_neighbors` refuses a type whose images are not those it builds
with. So a stronger type cannot be made, and a weaker one only through the
cast.

## What each consumer requires

The corrected table of #877:

| Consumer | Traversal, images | Exclusions | Organization |
|---|---|---|---|
| `pair_for (directed, owner_only)` | the loop's own | `masked` | `grouped_by(owner)`: each center adds its partners in its registers and stores once, so its partners are its own row |
| `pair_for (directed, atomic)` | the loop's own | `masked` | `none` |
| `pair_for (unique, atomic)` | the loop's own | `removed`: the loop reads the bits of its entries and skips nothing | `none` |
| any of these with a `fixed_order` value in `md_exec.algebra` | | | joined with `stable(owner, search)` (directed) or `stable(group, search)` (unique) |
| `build_triplets`, `build_edges` | `directed`, `minimum` | `masked` | `stable(owner, search)`: the rows hold every neighbor of a particle (D160, D277), and the order of the tuples they make is the order of later sums |

`md_exec::getRequirement` derives the requirement from the policy and the
algebra. It never infers one from the region: inference from code is the
conservative guess that #877 set out to remove.

A loop that states no algebra yet, before `md-exec-assign-algebra`, takes
nothing from the algebra. `(unique, atomic)` with a `fixed_order` value
passes the type check, because `stable(group, search)` is met. The
lowering still refuses it, because its atomics fix no order. A stable list
is necessary for `fixed_order`, not sufficient: the lowering must also fix
its reduction tree (#877, correction 4).

The verifier of each consumer refuses $r \not\sqsubseteq g$ and names the
axis:

```text
'md_exec.pair_for' op takes a neighbor structure that does not meet its requirement:
the organization 'stable(owner, search)' is required, and the structure guarantees 'grouped_by(owner)'
```

Two checks of the lowering now have a check in the verifier: the
refusal of `lowerPairFor` of a `unique` loop over a matrix and of a
`directed` loop over groups. D357's provenance check of the images also
has one. These checks stay in place; the type check runs before them.
Coverage (D357) keeps the cutoff and the exact centers. Those are SSA
facts and values that a type cannot hold, which is why D357 chose
provenance over a contract in the type. The layout contract is different:
it is a closed lattice that the build fixes and that has no SSA subjects.

## The cast

```mlir
%w = md_exec.weaken_neighbors %nl
       : !mdrt.neighbors<@atoms, directed, minimum, masked, stable(owner, search)>
       to !mdrt.neighbors<@atoms, directed, minimum, masked, grouped_by(owner)>
```

The op changes only the type, to a weaker guarantee on the same particle
set. It is `Pure`. It folds to its operand when the two types are equal.
Its verifier refuses a result that the operand's guarantee does not meet:
a stronger one, or one with another key or another traversal.

A pass that finds that no consumer of a structure needs an order will
weaken the type. The sort that gave the order then has no reader, and the
dead-collection rule of #876 can remove it. A sort becomes an enforcer of a
property [[Graefe1993]](../references.md#graefe1993).

Nothing emits the cast yet. D357's coverage query looks through it,
because it forwards the same structure. `findBeginnings` of
`md-exec-choose-neighbors` and the chains of `md-exec-reuse-neighbors` do
not look through it. A structure behind a cast therefore stays a matrix and
keeps its own storage. This limitation is to be removed when a pass first
emits the cast.

## The same programs

The lowered programs of the branch were compared with those of the base
(125b1f94), by the method of D406. The command was `mdir emit
--stage=lowered` with `MDIR_GPU_BINARY=ptx MDIR_GPU_ARCH=sm_89
MDIR_COMPILE_CACHE=off MDIR_SKIN_CACHE=off MDIR_SKIN_SEARCH=never`, which
gives the PTX of every kernel and the host's code, through
`scripts/builder-equivalence/check.py compare --stages lowered`. The
programs are identical, byte for byte:

| Program | Target | Mixed | Double |
|---|---|---:|---:|
| the rhombic dodecahedron of `test/Driver/triclinic.test` (flexible water: bonds and angles in ragged rows), groups | GPU | 154 | 153 |
| the same, groups with a dual list (`pruned_distance = 8.5`) | GPU | 164 | 163 |
| the same, matrix | GPU | 125 | 124 |
| the same, matrix, deterministic | GPU | 135 | 134 |
| JAC NVE of the Amber suite, groups | GPU | 308 | 304 |
| the same, groups with a dual list | GPU | 338 | 334 |
| the same, matrix | GPU | 275 | 271 |
| the same, matrix, deterministic | GPU | 297 | 293 |
| the four matrix programs above | CPU | identical | identical |

The numbers are the kernels of each program. The control files with groups
stop on the CPU with the same error in both builds. That is 24 programs
and 8 equal errors.

`md_dist` is lowered by `mdir-opt`, not by `mdir run`. Two programs of the
tests were lowered through `convert-md-dist-to-md-exec` and the device
pipeline, then through `--gpu-lower-to-nvvm-pipeline`:

- `test/Integration/dist-eam-exec.mlir`: 3×2×1 domains, the two-ended
  loops of D290;
- `test/Integration/dist-lj-exec.mlir`.

Each was lowered in double and mixed precision, with and without
`md-exec-choose-neighbors=kind=groups`. The 8 outputs are identical to
those of the base, apart from the compile time that MLIR records in each
binary (`LLVMIRToISATimeInMs`). They have 85 to 315 kernels each.

## Tests

- `test/Dialect/MDRT/neighbors-type.mlir` and `neighbors-type-invalid.mlir`
  cover the spellings, the refusal of the bare type, the normalization of `sorted_within`,
  and one case for each refusal of the parser.
- `test/Dialect/MDExec/neighbors-contract-invalid.mlir` has one case for
  each refusal. Each case is a realistic bug:
  - a `unique` loop over a directed list, and a directed loop over groups;
  - `owner_only` over a list that is not grouped, over groups, and over
    a list grouped by another key;
  - `fixed_order` over a list that is only grouped, and over one stable
    by another key;
  - a mismatch of the images;
  - a loop over groups over a list typed `masked`;
  - `build_triplets` over groups;
  - a build of a matrix typed as groups, and the reverse;
  - an empty structure typed as more than its kind;
  - a build with explicit images typed as minimum;
  - a refresh whose images are not its type's;
  - the cast, which refuses to strengthen the organization or the
    exclusions, another key, another traversal, and another particle
    set;
  - branches of an `scf.if`, and the yield of an `scf.for`, whose types
    differ;
  - a refresh whose result is not its operand's type.
- `test/Dialect/MDExec/neighbors-contract.mlir`:
  - the type through `refresh_neighbors`, the values of an `scf.for`, and
    the results of an `scf.if`;
  - every consumer of a matrix (including the deterministic mode's
    `fixed_order` sum), a weakened list for what needs less, and the
    rows of a domain;
  - the fold of the cast (`--canonicalize`).
- `test/Dialect/MDExec/Transforms/choose-neighbors.mlir` checks the type on
  every value of a structure that becomes groups (with a dual list too),
  and a structure that reaches an `scf.if`, which keeps the matrix.
- `test/Dialect/MDExec/Transforms/reuse-neighbors.mlir` checks that a
  structure of groups keeps its type on the empty structure, the value the
  loop carries, and the refresh.
- Nine existing tests of groups and of explicit images now spell their
  types. One of them, `choose-scatter-pairs.mlir`, had a `(directed,
  owner_only)` loop over a structure of groups. The verifier now refuses
  that loop, and the test takes a matrix for it.

**A mutation that must fail.** The organization's order was changed so
that `grouped_by(G)` meets `stable(G, K)`. Under that mutation three cases
of `neighbors-contract-invalid.mlir` fail:
- `fixed_order` over a list that is only grouped;
- `fixed_order` over a list stable by another key, whose requirement the
  mutated join weakens to `grouped_by`;
- the cast from `grouped_by` to `stable`.

The mutation was then reverted.

## Open

- **Ruled on #882** (the maintainer, 2026-11-14): the parameter is
  mandatory; the keys are the total ones above; the cast is
  `md_exec.weaken_neighbors`; `!mdrt.incidence` takes its contract later.
- **`!mdrt.incidence`**, the ragged rows, takes a contract of its own when
  its consumer moves to `md_alg` (#877, item 3).
- **A key with ties for sort elision** comes with the first elision
  (ruling 2). `sorted_within(owner, cell)` is what the device matrix would
  guarantee without its sort of each cell by index. It needs one rule of
  refinement: `stable(G, search)` meets `sorted_within(G, cell)`.
- **`centers`** stays an SSA fact of D357's query, outside the type.
