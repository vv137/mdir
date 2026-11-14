# D[md-alg-traverse]

Recorded at its merge.

**A loop over pairs over groups becomes one target-neutral op, `md_alg.traverse_pairs`. A new pass, `convert-md-exec-to-md-alg`, makes it from each `md_exec.pair_for` over a structure of groups. The op keeps the loop's region and declares what the loop needs: the visit order, the combination of each destination, and its algebra (D301). Its verifier checks two things: the region writes nothing, so every write goes through the declared combination; and the requirement that follows from the declarations is met by the contract of the list (D413). The serial lowering is the plain loop over the entries, and it is the oracle. The SIMT lowering is today's schedule of `lowerGroupPairFor` and `emitGroupPairKernel`, moved into the `md_alg` lowering library as a pattern. For the bytes to stay those of `main`, that pattern is called where the op stands, in program order, until four couplings to the big pass are removed (Section 7; open question 1). The old path for groups is removed in the same PR. The design leaves room for the rigid-water template of #795: lists of units, segments of class pairs, constants of a template, and sites formed in the kernel (Section 9).** Issue #886, step 2 of #877 (the maintainer's rulings of 2026-11-14), part of #558.

Status: the design, for review on the pull request; nothing is implemented yet.

## 1. What the groups path reads today

`Lowering::lowerPairFor` (`lib/Conversion/MDExecToGPU/MDExecToGPU.cpp:3651`)
sends a `unique` loop to `lowerGroupPairFor` (`:3787`–`:4212`). That
function writes the host code and the kernels. The kernels' inner part is
`kernels::emitGroupPairKernel` (`lib/Conversion/MDExecKernels/MDExecKernels.cpp:558`–`:946`).
The table lists everything this path reads, and where each fact goes in the
new form:
- **operand**: a value of the op;
- **attribute**: a declaration of the op's meaning, which both lowerings honor;
- **type**: a fact of the list's contract (D413), or a fact of the layout the build guarantees;
- **SIMT**: an execution parameter (D406). The serial lowering ignores it,
  and the SIMT lowering requires it;
- **host**: state of `convert-md-exec-to-gpu`, which the SIMT lowering
  reaches through an interface (Section 7).

### From the op `md_exec.pair_for` (`include/mdir/Dialect/MDExec/MDExecOps.td:1097`)

| Fact | Read at | Goes to |
|---|---|---|
| the structure `%nl` | `getNeighbors`, `:3657`; the dispatch on `unique`, `:3660`–`:3668` | operand (the list, Section 3) |
| `positions` | `:3825`; the gather into places, `:3891`–`:3950` | operand |
| `cell` | `:3827`; lowered to 3 or 6 numbers by the module's flag `triclinic` (`:233`–`:236`) | operand. Triclinic is a fact of the lowered cell's width (`isTriclinic(box)`, `MDExecKernels.cpp:710`), not an attribute |
| `ins` | gathered into places, `:3872`–`:3950`; loaded per lane and per entry, `MDExecKernels.cpp:618`–`:621`, `:729`–`:730` | operands |
| `outs` | cleared where overwritten, `:3814`–`:3827`; atomics, `MDExecKernels.cpp:854`–`:868`, `:901`–`:915` | operands, written only through the combination |
| `reduce` (the initial values) and the results | `finishSums`, `:1391`, called at `:4211` | operands, and the op's results |
| `scratch` | the partial sums of the warps, `:4064`–`:4073` | SIMT (the execution group) |
| `cutoff` | `MDExecKernels.cpp:571`, `:799`–`:801`, `:811`–`:812` | attribute |
| `weights` | `MDExecKernels.cpp:924`–`:927` (times 2: each pair once) | attribute |
| `overwrite` | `:3814`–`:3827` | attribute |
| `traversal` (`unique`) | `:3660` | attribute |
| `conflict` (`atomic`) | the verifier, `MDExecOps.cpp:1125`–`:1135` | attribute `combine` |
| `exchange` | `MDExecKernels.cpp:824`–`:833` (the partner's sign) | attribute |
| `centers`, `center_set`, `ghostFree`, `images(explicit)` | refused with `unique`, `MDExecOps.cpp:1136`–`:1140` | none: the op has no domain rows in this step |
| `scatter(hash(S, P, L))` (D291) | `:3991`–`:4002`; the table of a block, `:4026`–`:4032`; the choice by `L`, `:4084`–`:4102` | SIMT |
| `hostLanes` (D402) | the host only | none |
| `md_exec.algebra` (D301) | `everythingAllowsReorder`, `:3797` and `:520` | attribute `algebra`, one entry per value |
| the region | cloned per step, `MDExecKernels.cpp:803`–`:810` | the op's region, unchanged; its argument types give the type the pairs are computed in (`:3841`) |

### From the structure, as the lowering holds it (`Neighbors`, `:122`–`:180`)

| Fact | Read at | Goes to |
|---|---|---|
| `handle`: the buffers `order`, `counts`, `entries`, `masks`, `units`, `ordinals`, `shifts` | `getGroupsBuffers`, `:6627`, called at `:3830` | operands of the buffer form (Section 3); host in the structure form |
| `sizes[0]` (places), `sizes[2]` (units, the blocks) | `:3842`–`:3845`, `:4107`–`:4110` | operands of the buffer form (index values) |
| `sizes[1]` (the longest list) | `getGroupsShape`, `:6694`, for the fused pruning | host |
| `dual`: the inner list (D114) | set by the refresh with `prune_skin`, `:7814`; read at `:3971`–`:3981` | attribute `lists(inner)`, which the converter reads from the refresh |
| the inner list's `entries`, `masks`, `counts`, and the active units with their number (D268) | `getInnerLists`, `:6670`; `getPruneBuffers`, `:6677` | operands of the buffer form |
| `prunedNow`, the frames that the pruning left (`getGroupsPlaced`, `:6688`) | `:3847`–`:3864`, `:4150`–`:4152` | host (SIMT state) |
| the empty place (`order[p] = -1`) | `:3891`–`:3897`; `MDExecKernels.cpp:883`–`:887`, `:903`–`:912` | layout, shared with the build |
| the shift of a place, 10 bits an axis from −512 | `:3901`–`:3924` | layout, shared with the build |
| the bits of an entry's mask: 16 pairs, then the shift of the entry in 4 bits an axis, or 5 for a triclinic cell (D95, D115) | `MDExecKernels.cpp:701`–`:726`, `:732`–`:762`; `getEntryShiftBits`, `include/mdir/Conversion/MDExecKernels.h:75` | layout, shared with the build; the type states only what it means, `minimum` images by the entry's shift (I4 of D413) |
| groups of 16 places, blocks of 64 entries (D89) | `MDExecKernels.cpp:582`–`:616`, `:660`; `GroupsBuild.cpp` | layout, shared with the build: constants of D89 that the build and the schedule both write out today. They become one header, read by the build and both lowerings. They are not a type parameter, since D413 keeps the group size a cost of the build; the buffer form states them in its layout (`groups(16, 64)`) for the serial lowering |

### From the pass

| Fact | Read at | Goes to |
|---|---|---|
| `findSharedFrames` (D268): the loop follows its refresh at the same positions and in the same cell, and nothing between them writes the positions | `:2043`–`:2075`; read at `:3858` | SIMT attribute `frames(shared)`, set by the converter |
| `findPrunedInLoop` (D284), its reach, and `fuse-pruning`/`fusePruningGroups` | `:2077`–`:2102`; read at `:4142`–`:4198`; **also read by the refresh**, `:7783` | SIMT attribute `prune(fused, groups(G), reach(R))`, set by the converter. The refresh finds it on the op among its result's users |
| `check-pruning` (D268) | `:4195`–`:4197`; the refresh, `:7811` | SIMT attribute `check_pruning` |
| `block-size` | `:3801`–`:3803`, `:4022` | SIMT `block(B)` |
| the deterministic refusal | `:3797` | the verifier (Section 5, rule 4) |
| the arithmetic finishing: contraction, explicit FMAs, and the approximate reciprocal (D98, #102) | `forEachKernelRegion`, `:9025`–`:9045` (it already covers the regions of `md_alg` ops) | host (Section 7) |
| the names of the kernels, `<function>_pair_for_<ordinal>` (D354) | `nameKernel`, `:846`; `bringIn`, `:855` | host. The schedule's kernels keep the purpose `pair_for` |
| `lowerLookups`, `expandPowers`, `keepReadbacksOnDevice`, `deferReadbacks`, `releaseStack`, run on the function after its ops | `:8891`–`:8901` | host (Section 7) |

The batched stages (D293, `lowerStagedPairFor`, `:3338`) need the slots of
`build_edges`, and a matrix alone has them (`:3349`). No loop over groups
takes that path, so it stays where it is.

## 2. The op

```mlir
%u, %w = md_alg.traverse_pairs %nl lists(inner), %x, %cell
    ins(%q : memref<?xf32, 1>)
    outs(%f : memref<?x3xf32, 1>) overwrite [true]
    reduce(%u0, %w0 : f64, vector<6xf64>)
    unique visit unordered combine atomic
    algebra [any_order, any_order, any_order]
    exchange [antisymmetric, symmetric, symmetric]
    cutoff(0.9) weights [0.5, 0.5]
    simt(block(128) scratch(%s0, %s1, %s2, %s3 : ...)
         scatter(hash(512, 4, 2000)) frames(shared)
         prune(fused, groups(4096), reach(1.0)) check_pruning)
    : !mdrt.neighbors<@atoms, unique, minimum, removed, stable(group, search)>,
      memref<?x3xf64, 1> {
^bb0(%r2: f32, %d: vector<3xf32>, %qi: f32, %qj: f32):
  ...
  md_alg.yield %kf, %ku, %kw : vector<3xf32>, f32, vector<6xf32>
} -> f64, vector<6xf64>
```

**Meaning.** For each entry $(i, j)$ of the list with
$r_{ij}^2 < r_c^2$, the region gets $r^2$, $\mathbf d = \mathbf x_i - \mathbf x_j$,
and the values of `ins` at $i$ and at $j$. It yields one value $k_a$ for
each destination and one value $k_S$ for each sum. Then:

$$a_i \mathrel{+}= k_a(i,j),\qquad a_j \mathrel{+}= \sigma_a\,k_a(i,j),\qquad S \mathrel{+}= 2\,w_S\,k_S(i,j)$$

Here $\sigma_a = +1$ for a `symmetric` destination and $-1$ for an
`antisymmetric` one (`unique`: each unordered pair once). A destination
with `overwrite` starts from zero. The results are the initial values plus
the sums.

**Where the precision changes** is part of the meaning, so that the two
lowerings differ only in the order of the sums:
- a position is moved into the frame of its group, by its place's shift,
  in the type it is stored in (f64 in mixed precision);
- that sum is converted to the type of the region's first argument (D95,
  `:3921`–`:3933`);
- the entry's shift is added in that type (`MDExecKernels.cpp:706`–`:726`).

Both lowerings do exactly this. The serial loop does not convert before
it shifts.

**Operands:** the list (Section 3), the positions, the cell, `ins`, `outs`,
and the initial values of `reduce`. In the execution group: `scratch`.

**Attributes of the meaning:**
- `unique` or `directed`: the multiplicity of the visit. A `unique` visit
  of a directed list would count every pair twice (#877, correction 3).
- `visit`: `unordered`, or `stable` by the list's key. It is derived from
  the algebra: `stable` exactly where a value is `fixed_order`. That
  condition is necessary for `fixed_order`, not sufficient; the lowering
  must also fix its reduction tree (#877, correction 4).
- `combine`: how the destinations receive their values. The values are
  those of the md_exec conflict:
  - `atomic`: additions at both ends, in any order;
  - `owner`: each owner adds its row in registers and stores once;
  - `privatized`: is reserved.

  The block table of D291 does not change the meaning: it is an execution
  strategy of `atomic`, and it adds the same terms.
- `algebra`: one entry for each destination and then each sum, as
  `md_exec.algebra` (D301).
- `exchange`, `cutoff`, `weights`, `overwrite`: as on `pair_for`.

**The execution group `simt(...)`:** `block`, `scratch`, `scatter`,
`frames`, `prune`, and `check_pruning`. These are D406's optional group:
the result does not depend on them. The converter fills them, as the
callers of D406's ops do, until a policy pass exists (#433).

**Results:** the sums, as values. The lowering alone knows about the
partial sums of the warps.

## 3. The list: two forms of one op

Two constraints meet here. The verifier compares the requirement with the
contract that a *type* carries (ruling 1 on #877). The serial lowering has
to run on the host over plain buffers (D406). Groups exist only on a
device, and the runtime hands their buffers out by a handle. So the op
has two forms, as `md_exec` has a value form and a storage form:

- **The structure form**, which `convert-md-exec-to-md-alg` makes: the
  operand is `%nl : !mdrt.neighbors<...>`, with `lists(outer)` or
  `lists(inner)`. The verifier reads the guarantee from the type, as
  `pair_for` does (`md_exec::verifyRequirement`).
- **The buffer form:** the operands are the buffers of the layout, with
  the counts of places and of blocks:
  - `groups(entries, masks, counts, units, ordinals, places, shifts)`;
  - and `inner(entries, masks, counts, active, active_count)` for the inner list.

  The contract is an attribute,
  `contract(unique, minimum, removed, stable(group, search))`, with the
  parser and the order of D413's type parameter. It is a stated
  precondition on the buffers, as the order of `items` is a precondition
  of `bin_members` (D409). The verifier checks the requirement against it.
  The serial lowering, the tests, and the SIMT pattern on its own
  (`convert-md-alg-to-gpu`) take this form.

The SIMT lowering called by `convert-md-exec-to-gpu` takes the structure
form and binds the buffers itself: it calls the runtime's getters through
the host interface at the places where `main` calls them. In this step,
nothing turns one form into the other as a pass. That conversion is item
(b) of Section 7.

**Rejected:** a new list type with a pack op. It would need a 1:N type
conversion from one value to the seven buffers in the big pass. It would
also give no more checking than the contract attribute does: a pack from
raw buffers states its contract too.

## 4. The region

The region is that of `pair_for`, unchanged: it has the same arguments,
and it ends in `md_alg.yield` with one value for each destination and then
each sum. The converter moves it; it does not rewrite it. The finishing of
the arithmetic already reaches the regions of `md_alg` ops
(`forEachKernelRegion`), so the region's arithmetic is finished as it is in
`main`.

## 5. The verifier

Each rule has one negative test, and each test is a realistic bug
(Section 8):

1. **No write in the region.** The region has no `memref.store`, no
   atomic, and no call or op with a write effect. Every write is the op's,
   through `combine`. This rule is stricter than D409's for
   `bin_members`, which admits slots of the item: a loop over pairs writes
   at both ends, and the op owns those writes.
2. **No read of a destination.** The region reads no buffer of `outs`,
   and no view of one. No buffer of `outs` is, or views, an operand that
   the op reads: `positions`, `ins`, or a list buffer. These are the views
   that D409 follows, through the helper shared with `bin_members` and
   `sort_by_key` (#871, #874).
3. **The requirement is met:** requirement $\sqsubseteq$ guarantee. The
   requirement follows from the declarations, as `md_exec::getRequirement`
   computes it:
   - the traversal, and the images `minimum`;
   - `removed` exclusions for `unique`, `masked` for `directed`;
   - `grouped_by(owner)` for `combine owner`, `none` for `atomic`;
   - joined with `stable(group, search)` (unique) or
     `stable(owner, search)` (directed) under `visit stable`.

   The guarantee is the type of the structure form, or the contract
   attribute of the buffer form.
4. **The visit meets the algebra.** A `fixed_order` value needs
   `visit stable`. `combine atomic` needs every destination to allow
   reordering, and `unique` needs every sum to allow it too (the partial
   sums of units). This moves the refusal of `:3797` from the lowering
   into the verifier.
5. **Exchange.** Under `unique` or `atomic`, every destination is
   `symmetric` or `antisymmetric` and every sum is `symmetric`, as on
   `pair_for` (`MDExecOps.cpp:1141`–`:1154`).
6. **`combine owner` takes `directed`.** With `unique`, the partner's end
   is not the owner's row.
7. **Coverage.** The structure form runs D357's query on its operand, as
   `pair_for` does: the cutoff is within the reach that the structure was
   built for. `verifyNeighborCoverage` becomes a query over an interface
   of the consumers, which both ops implement. The buffer form states its
   reach with its contract, `reach(R)`, as a precondition, and the
   verifier refuses a cutoff beyond it. Without this, a hand-written op
   could claim any cutoff over any list, which is P2 of #877 one layer
   down.
8. **Shapes:** one `algebra` and one `exchange` entry for each value, one
   `weights` entry for each sum, and one `overwrite` flag for each
   destination. In the buffer form: the buffers of the layout, i32 and
   rank 1, and the contract.

**The SIMT lowering refuses** what its schedule cannot keep:
- a `visit stable` value: the rotation and the atomics fix no tree;
- `combine owner` over groups;
- a missing execution group;
- `block` not a multiple of 32 (`:3801`).

## 6. The two lowerings

**Serial** (`convert-md-alg-to-scf`), the oracle. It takes the buffer form,
and it refuses the structure form, because groups do not exist on the host.
For each unit $b$ (block $b$ of group `units[b]`), for each place $p$ of
the group and each entry $e$ of the block whose bit $p$ is set:
1. take $\mathbf x_i$ in the frame of the group, the position plus the
   place's shift;
2. take $\mathbf x_j$, plus the entry's shift;
3. test $r^2 < r_c^2$;
4. run the region;
5. add $k_a$ at $i$, and $\sigma_a k_a$ at $j$, in the order of the visit.

Under its existing option `reverse-unordered-visits` (D409), it visits the
units and the entries of an `unordered` traversal from the last to the
first. With `lists(inner)` it reads the inner list. The serial code is
written apart from the schedule, in the plain order of the definition.

**SIMT** (`convert-md-alg-to-gpu`, the pattern `TraversePairsGroupsSIMT`):
the code of `lowerGroupPairFor` and `emitGroupPairKernel`, moved into
`lib/Conversion/MDExecKernels/TraversePairs.cpp`. It does what it does
today:
- the clears, and the gather into the places with the frames;
- a warp for each unit: lanes $u$ and $u+16$ hold place $16g+u$, and the
  entries turn through the half-warps for 16 steps;
- the ballots of the masks, the hash table, the active units, and the
  fused pruning (D284);
- the partial sums of the warps, and the frees.

Its sources change only in the names they take:
- the op `traverse_pairs` instead of `pair_for`;
- the host interface instead of members of `Lowering`.

## 7. Where the SIMT lowering runs (open question 1)

A pattern that runs *after* `convert-md-exec-to-gpu` cannot give `main`'s
bytes today. Four things tie the schedule to the big pass:

- **(a) The names of the kernels** are ordinals in the order of the
  lowering (D354). The schedule's kernels, and the reduction kernels of
  `finishSums` after them, are numbered in turn with every kernel around
  them.
- **(b) The order of the host code.** The getters of the runtime
  (`getGroupsBuffers`, `getInnerLists`, `getPruneBuffers`,
  `getGroupsPlaced`) are called between the launches and the `scf.if`s of
  the schedule (`:3830`, `:3859`, `:3972`–`:3976`, `:4163`). The host code
  is part of D406's comparison.
- **(c) The work after the ops.** After its ops, the big pass runs over
  the function:
  - `lowerLookups`, `expandPowers`, `keepReadbacksOnDevice`, and
    `deferReadbacks`, which numbers a slot for each sum of the function;
  - `releaseStack`;
  - the finishing of the module's arithmetic.

  The finishing reaches the region, but not the schedule's own arithmetic
  in a kernel: for example, the frame shift `position + shift`
  (`MDExecKernels.cpp:726`), which `main` contracts.
- **(d) The analyses of the pass** (D268, D284). The refresh's own
  lowering reads their results (`:7783`).

The converter takes (d): it runs the two analyses, which are facts of
`md_exec`, and records them on the op. The other three remain.

**Option A (proposed): call the pattern in place.**
- `convert-md-exec-to-md-alg` makes the op. The SIMT pattern lives in the
  `md_alg` lowering library and is registered in `convert-md-alg-to-gpu`,
  where the tests run it on the buffer form.
- `convert-md-exec-to-gpu` calls the same pattern where it meets the op,
  in program order. It passes the pattern an interface, `TraverseHost`,
  for:
  - the names of the kernels (purpose `pair_for`), and `bringIn`;
  - the device buffers, and the getters of the structure by its handle;
  - `prunedNow`, and the frames;
  - `finishSums`.
- The bytes are then those of `main` by construction, and the comparison
  of D406 checks it.
- `convert-md-alg-to-gpu` supplies a default `TraverseHost` for the op on
  its own, in the buffer form:
  - names by its own counter;
  - device buffers by `gpu.alloc`;
  - the partial sums reduced by the existing block reduction
    (`ReductionEmitter.h`).

  The contract test of Section 10 runs the pattern through this default.

This departs from the issue's words: the pattern of
`convert-md-alg-to-gpu` is the code, but the big pass calls it. The list
of the ops that are lowered in place (D408; `docs/device-primitives.md`,
"Pass order") gains this op, with what would free it:
- (a) names of kernels that do not depend on the order of the lowering:
  a one-time rename of every kernel, which changes every program's bytes;
- the sums of a loop as an `md_alg` reduction op, the reordered
  floating-point reduction of D406, which (c) needs for its slots;
- the finishing, the lookups, and the powers as passes after
  `convert-md-alg-to-gpu`;
- (b) the binding of the buffers as an op that the big pass lowers before
  the traversal, after a reorder of `main`'s host calls.

Until then the op does not reach `md-alg-transform-collections`. For groups
that loses nothing today. The lists reach the loop through the runtime's
handle: D409's analysis sees them as escaping (rule 3), so the slot could
not use the consumer. The requirement that would allow an elision is in the
type (D413), where `md_exec.weaken_neighbors` acts.

**Option B: a pattern after the big pass, with preparatory steps.**
Items (a) to (d) come first, each as a PR of its own that changes `main`'s
bytes once and is checked by its own method:
- a rename with the kernels' code equal;
- the reduction op;
- the finishing as a pass;
- the reorder of the host calls, with every kernel equal and the host
  code reordered.

Then step 2 ports onto that `main`, byte for byte. The cost is 3 or 4 PRs
before this one. Every kernel's name would change for profiles, traces,
and the sanitizer's reports (D354).

## 8. The pipeline and the removal of the old path

```
... md-exec-assign-storage, md-exec-assign-streams, [stage bodies],
convert-md-exec-to-md-alg{fuse-pruning=... fuse-pruning-groups=... check-pruning=... block-size=...}
convert-md-exec-to-gpu{...}         (calls the pattern in place: option A)
md-alg-transform-collections, convert-md-alg-to-gpu, ...
```

- **The new pass.** `convert-md-exec-to-md-alg` turns each
  `pair_for (unique, atomic)` into the structure form. It runs
  `findSharedFrames` and `findPrunedInLoop` (moved from the big pass) and
  fills the execution group. The options of the fused pruning move to it
  from `convert-md-exec-to-gpu` (`getFusedPruningOptions` in
  `lib/Compiler/Compile.cpp`). The ops of the loop's region become the
  region of the new op.
- **The op's effects.** The op implements `MemoryEffectOpInterface`
  exactly: it reads the list, the positions, `ins`, and the cell, and it
  writes `outs` and `scratch`. The analyses of the big pass that walk ops
  then see what they saw before: `BufferAliases`, the hoisting of a
  reciprocal sum across a refresh (D113, `:8771`), `getEffectsRecursively`,
  and `md_exec::verifyTransfers`.
- **Removed:**
  - `lowerGroupPairFor`;
  - the `unique` branch of `lowerPairFor`, which becomes a refusal: "a
    loop over groups is converted by `convert-md-exec-to-md-alg`";
  - `findSharedFrames` and `findPrunedInLoop` in the big pass;
  - `emitGroupPairKernel` in `MDExecKernels.cpp`, which moves.
- **Stays:** MATRIX, the ragged rows, PME's accessors, and the batched
  stages.
- **The documents** that change with the implementation: `docs/device-primitives.md`
  ("Pass order", and the ops of `md_alg`), `docs/neighbor-groups.md` (the
  loop over pairs), the dialect's reference, the roadmap, and the paper's
  section on the layers.
- **`mdir-opt`.** The tests that lower groups by `mdir-opt` (in
  `test/Conversion/MDExecToGPU/` and the `md_dist` programs with
  `md-exec-choose-neighbors=kind=groups`) put the new pass before the big
  one.

## 9. What the template of #795 needs

The maintainer's comment on #886 (2026-11-14) asks for four things. Each
one fits after the change described below. None of it is implemented in
this step: the verifier of step 2 admits only the degenerate case, which is
the groups.

**(1) Lists of units, whose entries are a superset filtered by the cutoff
of site pairs.**
- The superset needs nothing new. The op owns the cutoff test, so every
  list may hold more entries than the cutoff selects. The matrix and the
  groups already do: they are built to the reach $r_c + s$.
- The f32 search's extra entries in the shell, which the ruling on #795 of
  2026-11-11 allows, are more of the same. They contribute exact zeros.
- What changes is that the test moves from the entry to each pair of
  sites. A unit's entry stands for its site pairs, and the op tests
  $r_{ab}^2 < r_c^2$ for each pair $(a, b)$.
- The list of a unit set is a structure over a set of units, `@atoms.units`,
  whose first type parameter is that set. Its coverage is D357's query,
  with the covering reach of each segment that the build states:
  $R + 2r_\mathrm m$, $R + r_\mathrm m + \rho$, and $R + 2\rho$ (#795,
  Section 2).

**(2) Three classes of cluster pairs in one buffer.**
- The list's type becomes segmented: one buffer, with one contract per
  segment (`ww`, `sw`, `ss`), each a D413 contract. The list is then
  `grouped_by(class)` across the segments, which is a value of D409's
  lattice. That is #795's `!mdrt.segmented_pairs`, with its contract in
  the type.
- The op takes one region per segment (a variadic region list, keyed by
  the segments of the type) and one visit and requirement per segment. The
  verifier checks each.
- The groups are the case of one segment. #795's `md_exec.template_for`,
  with its three tagged bodies, converts to this form one body to one
  region.
- That a warp stays inside one segment is a choice of the SIMT schedule.

**(3) The template's parameters as constants of the schedule, with zero
terms not emitted.**
- Each segment's unit shape carries a table of its site pairs. For each
  pair it gives the constants (the product of the charges and the
  Lennard-Jones coefficients) and a mask of the terms that are not zero.
- The region yields one value per term and destination. The lowering
  binds the constants to `arith.constant` and clones, for each site pair,
  only the slices of the terms in its mask.
- A zero term is therefore not emitted. Folding is not relied on, because
  $x \cdot 0$ does not fold in IEEE arithmetic.
- For the groups the table is one site pair with every term, and the
  values come from `ins`.

**(4) Sites formed in the kernel, with the forces written through the
transpose.**
- A unit shape has $k$ members and $s$ sites, with constant weights
  $\mathsf W \in \mathbb R^{s\times k}$ and rows that sum to one. A site
  is
  $\mathbf x_a = \mathbf x_o + \sum_m W_{am}(\mathbf x_m - \mathbf x_o)$,
  in the frame of the anchor $o$.
- A force $\mathbf f_a$ on a site reaches the members as
  $\mathsf W^{\mathsf T}\mathbf f$, which is rule (A) of #795. For OPC:
  $k = 3$, $s = 4$, the fourth row $(1-2\alpha, \alpha, \alpha)$.
- The written forces take rule (B) (ruling 6 on #795). That is a second
  form of the shape, a `spread` region: from the members' offsets and
  $\mathbf f$, it yields the members' forces. The region is pure, and the
  combination still makes every write.
- The groups are $k = s = 1$ and $\mathsf W = (1)$.
- Forming each site once per lane, and rotating it with its partner, is
  #795's SIMT schedule of the same op.

**What this changes in the design:**
- the list type gains a segmented form;
- the op's region becomes a variadic list with one region per segment;
- the cutoff applies to site pairs.

Step 2 fixes the syntax with the degenerate defaults: one segment, `units(particle)`.
The full syntax comes with #795.

## 10. Tests and measures

- **The same programs** (D406's method): `mdir emit --stage=lowered` with
  `MDIR_GPU_BINARY=ptx MDIR_GPU_ARCH=sm_89 MDIR_COMPILE_CACHE=off
  MDIR_SKIN_CACHE=off MDIR_SKIN_SEARCH=never`, through
  `scripts/builder-equivalence/check.py compare --stages lowered`.
  - The programs: every groups program of the corpus and the nine systems
    of the Amber suite with groups, in mixed and double precision.
  - The variants: the dual list; the pruning fused (D284) and in the
    refresh (`fuse-pruning=false` through `MDIR_PIPELINE`); the
    single-pass scan; the hash table on and off; `MDIR_CHECK_PRUNING`.
  - Also the 8 lowerings of `md_dist` of D413 by `mdir-opt`, and the
    module after `convert-md-exec-to-gpu` in the generic form.
- **The serial lowering against the device**
  (`test/Integration/md-alg-traverse-contract-gpu.test`):
  - The lists are those of the device's build of the dodecahedron of
    `test/Driver/triclinic.test` and of JAC, with groups: the outer list,
    and the inner list after a pruning in the refresh and after a fused
    one.
  - They are read back into the buffer form on the host. The serial
    lowering runs there, and the SIMT pattern runs on the device on the
    same lists.
  - Energies, forces, and virials are compared. Each component $T$ of $n$
    terms $t_k$ in the type of its sum is allowed
    $\lvert\Delta\rvert \le 2\gamma_{n-1}\sum_k \lvert t_k\rvert + \epsilon_\mathrm{abs}$,
    with $\gamma_m = m u/(1 - m u)$ and $u$ the unit roundoff of that
    type. The bound for a reordered sum is that of D406.
  - $\sum_k\lvert t_k\rvert$ comes from a second serial run that adds the
    magnitudes, a test option of the serial lowering. $\epsilon_\mathrm{abs}$
    is the floor for a sum that cancels to zero.
- **A reversed visit.** The serial lowering with
  `reverse-unordered-visits`, against itself in order, under the same
  bound. The sums are floats under `any_order`, so the comparison is by
  the error model and not by bytes. The comparison of `bin_members` is by
  bytes because its `unordered` visit admits integers alone.
- **Mutations that must fail** (each applied alone, then reverted):
  - the partner's sign ignored;
  - the weight's factor 2 dropped;
  - the entry's shift dropped in the serial loop;
  - the inner list's per-block count ignored.
- **Negative tests**, one per refusal (`test/Dialect/MDAlg/traverse-invalid.mlir`;
  for the SIMT lowering, `test/Conversion/MDAlg/simt-invalid.mlir`):
  - the region stores into a destination;
  - the region reads a destination through a subview;
  - a destination that views `ins`;
  - `combine owner` over a list that is not grouped by owner;
  - `combine owner` with `unique`;
  - a `unique` visit of a directed list;
  - a `masked` list for a `unique` visit;
  - a `fixed_order` destination with `visit unordered`;
  - a `fixed_order` destination with `combine atomic`;
  - an `antisymmetric` sum;
  - a missing algebra entry;
  - the buffer form without its contract;
  - the serial lowering on the structure form;
  - the SIMT lowering: a `visit stable` value on the groups schedule,
    which fixes no tree; and a missing execution group.
- **The sanitizer tier,** since device code moves: compute-sanitizer
  memcheck and racecheck on the groups tests (`*-gpu.test`). The PR
  states how long it held the lock.
- **Timing** by the timing protocol: JAC and FactorIX with groups, on one
  kind of device, against `main`, in at least two interleaved jobs. The
  target is no slowdown beyond the spread. The PTX is identical, so this
  is a guard.

## Open

1. **Where the SIMT lowering runs:** option A or option B (Section 7).
2. **The list's two forms:** a contract attribute on the buffer form,
   against a list type with a pack op (Section 3).
3. **The name** `md_alg.traverse_pairs`.
4. **The direction for #795** (Section 9): a segmented list type, one
   region per segment, site-pair tables with masks of the terms, and unit
   shapes with constant weights and a `spread` region.
