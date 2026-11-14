# D[md-alg-traverse]

Recorded at its merge.

**A loop over pairs over groups is one target-neutral op, `md_alg.traverse_pairs`. A new pass, `convert-md-exec-to-md-alg`, makes it from each `md_exec.pair_for` over a structure of groups, before `convert-md-exec-to-gpu`. The op keeps the loop's region and declares what the loop needs: its visit, the combination of its destinations, and the algebra of each value (D301). Its verifier checks two things. The region writes nothing, so every write goes through the declared combination. The requirement that follows from the declarations is met by the contract of the list (D413). The serial definition is the plain loop over the entries, and it is the oracle. The SIMT lowering is the schedule that `lowerGroupPairFor` and `emitGroupPairKernel` wrote, moved into the `md_alg` lowering library. `convert-md-exec-to-gpu` calls it where the op stands, through an interface, so that every program keeps its bytes. The maintainer ruled this departure from #886's words on #887; #888 removes it before any fusion of a build into its consumer. The old path for groups is removed. The program of every groups case of the corpus is identical to `main`'s, by D406's method. The schedule and the serial definition agree within the error model of a reordered sum on the dodecahedron and on JAC. The design leaves room for the rigid-water template of #795.** Issue #886, step 2 of #877 (the maintainer's rulings of 2026-11-14), part of #558.

## The rulings (the maintainer, 2026-11-14, on #887)

1. **The SIMT lowering is called in place.** The big pass calls the pattern through `TraverseHost`. Option B's four steps are #888: names of kernels that do not depend on the order of the lowering, a reduction op for the sums, the finishing of the kernels' arithmetic as a pass, and the binding of the buffers as an op. They must be done before any fusion of a build into its consumer, or any removal of a materialized intermediate. Until then, sort elision goes through the type (D413).
2. **The list has two forms, and the pipeline emits one.** `convert-md-exec-to-md-alg` always emits the structure form, whose contract is in the type. The buffer form is for the serial oracle and the tests. Its `contract(...)` is a precondition that its writer asserts, not a guarantee that the IR checks. `convert-md-exec-to-gpu` refuses it.
3. **The name is `md_alg.traverse_pairs`.**
4. **The direction for #795 is agreed** (Section 8). Step 2 parses only the degenerate form.

## 1. What the groups path reads today

The line numbers are those of `main` at 1c11a780, the base of this change.
`Lowering::lowerPairFor` (`lib/Conversion/MDExecToGPU/MDExecToGPU.cpp:3651`)
sent a `unique` loop to `lowerGroupPairFor` (`:3787`–`:4212`). That
function wrote the host code and the kernels. The kernels' inner part was
`kernels::emitGroupPairKernel` (`lib/Conversion/MDExecKernels/MDExecKernels.cpp:558`–`:946`).
The table lists everything that path read, and where each fact went:
- **operand**: a value of the op;
- **attribute**: a declaration of the op's meaning, which both lowerings honor;
- **type**: a fact of the list's contract (D413), or a fact of the layout the build guarantees;
- **SIMT**: an execution parameter (D406). The serial lowering ignores it,
  and the SIMT lowering requires it;
- **host**: state of `convert-md-exec-to-gpu`, which the SIMT lowering
  reaches through an interface (Section 6).

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
| `handle`: the buffers `order`, `counts`, `entries`, `masks`, `units`, `ordinals`, `shifts` | `getGroupsBuffers`, `:6627`, called at `:3830` | operands of the buffer form (Section 3); in the structure form, the host binds them |
| `sizes[0]` (places), `sizes[2]` (units, the blocks) | `:3842`–`:3845`, `:4107`–`:4110` | operands of the buffer form (index values) |
| `sizes[1]` (the longest list) | `getGroupsShape`, `:6694`, for the fused pruning | host |
| `dual`: the inner list (D114) | set by the refresh with `prune_skin`, `:7814`; read at `:3971`–`:3981` | attribute `lists(inner)`, which the converter reads from the refresh |
| the inner list's `entries`, `masks`, `counts`, and the active units with their number (D268) | `getInnerLists`, `:6670`; `getPruneBuffers`, `:6677` | operands of the buffer form |
| `prunedNow`, the frames that the pruning left (`getGroupsPlaced`, `:6688`) | `:3847`–`:3864`, `:4150`–`:4152` | host (SIMT state) |
| the empty place (`order[p] = -1`) | `:3891`–`:3897`; `MDExecKernels.cpp:883`–`:887`, `:903`–`:912` | layout, shared with the build |
| the shift of a place, 10 bits an axis from −512 | `:3901`–`:3924` | layout, shared with the build |
| the bits of an entry's mask: 16 pairs, then the shift of the entry in 4 bits an axis, or 5 for a triclinic cell (D95, D115) | `MDExecKernels.cpp:701`–`:726`, `:732`–`:762`; `getEntryShiftBits`, `include/mdir/Conversion/MDExecKernels.h:75` | layout, shared with the build; the type states only what it means, `minimum` images by the entry's shift (I4 of D413) |
| groups of 16 places, blocks of 64 entries (D89) | `MDExecKernels.cpp:582`–`:616`, `:660`; `GroupsBuild.cpp` | layout, shared with the build: constants of D89 that the build, the schedule, and the serial definition write out. They are not a type parameter, since D413 keeps the group size a cost of the build |

### From the pass

| Fact | Read at | Goes to |
|---|---|---|
| `findSharedFrames` (D268): the loop follows its refresh at the same positions and in the same cell, and nothing between them writes the positions | `:2043`–`:2075`; read at `:3858` | SIMT attribute `frames`, which `convert-md-exec-to-gpu` sets where it ran the analysis, after it moves the reciprocal sums (D113) |
| `findPrunedInLoop` (D284), its reach, and `fuse-pruning`/`fusePruningGroups` | `:2077`–`:2102`; read at `:4142`–`:4198`; **also read by the refresh**, `:7783` | SIMT attribute `prune(G, R)`, set by `convert-md-exec-to-gpu` with the same analysis and its options `fuse-pruning` and `fuse-pruning-groups`. The refresh finds it on the op among its result's users (`getFusedPruning`) |
| `check-pruning` (D268) | `:4195`–`:4197`; the refresh, `:7811` | host: an option of `convert-md-exec-to-gpu`, which the refresh reads too |
| `block-size` | `:3801`–`:3803`, `:4022` | host: an option of `convert-md-exec-to-gpu` |
| the deterministic refusal | `:3797` | the verifier (Section 4) |
| the arithmetic finishing: contraction, explicit FMAs, and the approximate reciprocal (D98, #102) | `forEachKernelRegion`, `:9025`–`:9045` (it already covers the regions of `md_alg` ops) | host (Section 6) |
| the names of the kernels, `<function>_pair_for_<ordinal>` (D354) | `nameKernel`, `:846`; `bringIn`, `:855` | host. The schedule's kernels keep the purpose `pair_for` |
| `lowerLookups`, `expandPowers`, `keepReadbacksOnDevice`, `deferReadbacks`, `releaseStack`, run on the function after its ops | `:8891`–`:8901` | host (Section 6) |

The batched stages (D293, `lowerStagedPairFor`, `:3338`) need the slots of
`build_edges`, and a matrix alone has them (`:3349`). No loop over groups
takes that path, so it stays where it is.

## 2. The op

```mlir
%u = md_alg.traverse_pairs structure %nl : !mdrt.neighbors<@atoms, unique,
         minimum, removed, stable(group, search)> inner
     positions(%x : memref<?x3xf64, 1>) cell(%cell : !md.cell)
     ins(%q : memref<?xf32, 1>) outs(%f : memref<?x3xf32, 1>)
     reduce(%u0 : f64) scratch(%s0, %s1 : memref<?xf64, 1>, memref<?xf64, 1>)
     unique visit unordered combine atomic
     algebra ["any_order", "any_order"] exchange ["antisymmetric", "symmetric"]
     cutoff(9.0) weights [0.5] overwrite [true]
     scatter(#md_exec.scatter<hash(512, 4, 2000)>) frames prune(4096, 10.0) {
^bb0(%r2: f32, %d: vector<3xf32>, %qi: f32, %qj: f32):
  ...
  md_alg.yield %kf, %ku : vector<3xf32>, f64
} -> f64
```

**Meaning.** For each entry $(i, j)$ of the list with $r_{ij}^2 < r_c^2$,
the region gets $r^2$, the displacement $\mathbf d = \mathbf x_i - \mathbf x_j$,
and the values of `ins` at $i$ and at $j$. It yields one value $k_a$ for
each destination and one value $k_S$ for each sum. Then

$$a_i \mathrel{+}= k_a(i,j),\qquad a_j \mathrel{+}= \sigma_a\,k_a(i,j),\qquad S \mathrel{+}= 2\,w_S\,k_S(i,j),$$

where $\sigma_a = +1$ for a `symmetric` destination and $-1$ for an
`antisymmetric` one (`unique`: each unordered pair once). A destination
with `overwrite` starts from zero. The results are the initial values plus
the sums.

**Where the precision changes** is part of the meaning, so that the two
lowerings differ only in the order of the sums:
- a position is moved into the frame of its group, by its place's shift,
  in the type it is stored in (f64 in mixed precision);
- that sum is converted to the type of the region's first argument (D95);
- the entry's shift is added in that type.

**Attributes of the meaning:**
- `unique` or `directed`: the multiplicity of the visit. A `unique` visit
  of a directed list would count every pair twice (#877, correction 3).
- `visit`: `unordered`, or `stable`. The converter sets `stable` exactly
  where a value is `fixed_order`. A stable visit is necessary for
  `fixed_order`, not sufficient: the lowering must also fix its reduction
  tree (#877, correction 4).
- `combine`: how the destinations receive their values:
  - `atomic`: additions at both ends, in any order;
  - `owner`: each owner adds its row and stores once;
  - `privatized`: reserved.

  The block table of D291 does not change the meaning: it is an execution
  strategy of `atomic`, and it adds the same terms.
- `algebra`: one entry for each destination and then each sum, from
  `md_exec.algebra` (D301).
- `exchange`, `cutoff`, `weights`, `overwrite`: as on `pair_for`.

**The execution group,** which the result does not depend on (D406):
- `scratch`;
- `scatter`, the table of a block (D291);
- `frames`: the loop takes the frames that its refresh's pruning left
  (D268);
- `prune(G, R)`: the loop prunes for its refresh from $G$ groups, with the
  reach $R$ (D284).

The converter sets `scatter`, which `md-exec-choose-scatter` chose.
`convert-md-exec-to-gpu` sets `frames` and `prune` by the analysis it ran
before, which now walks the traversals. It runs after the pass moves the
reciprocal sums before the refreshes (D113). Run in the converter, before
that move, the analysis met a reciprocal sum between a refresh and its
loop. On FactorIX one loop lost its pruning in the loop, and the step was
3.5% slower; the bytes showed it. The block size, `check-pruning`, and
the options of the pruning in the loop stay options of
`convert-md-exec-to-gpu`, whose refreshes read them too.

The big pass's analyses that name the loops over pairs know the traversal
too:
- the aliases of a structure's buffers, among the readers of a structure
  (`Independence.cpp`);
- `overwritesAll`, by which a fused integration leaves a destination that
  the next loop overwrites;
- the writes that the fusion of a term over centers must not cross.

Without them, FactorIX's fused integration kernel also wrote the forces
that the loop overwrites.

**Effects.** The op declares its effects exactly, as `pair_for` does:
- it reads the list, the positions, and `ins`;
- it writes `outs` and `scratch`;
- it reads `outs` too, except where it overwrites them.

The analyses of the big pass that walk ops therefore see what they saw
before.

## 3. The two forms of the list

- **The structure form** (`structure %nl : <type>`, with `inner` for the
  inner list of a dual list, D114) is what the pipeline emits. Its
  guarantee is the type of the structure.
- **The buffer form** (`lists(...) sizes(%places, %units) contract(<type>)
  reach(R)`) takes the buffers of the layout:
  - `entries`, `masks`, `counts`, `units`, `ordinals`, `places`, `shifts`;
  - with `inner`, also the inner list's `entries`, `masks`, `counts`, the
    active units, and their number.

  Its contract and its reach are preconditions that its writer asserts
  (ruling 2). The serial definition and the tests take this form.
  `convert-md-exec-to-gpu` refuses it.

## 4. The verifier

Each rule has a negative case in `test/Dialect/MDAlg/traverse-invalid.mlir`.
Each case is a realistic bug:

1. **No write in the region.** The region has no op with a write, free,
   or allocate effect, and no op whose effects it cannot see. Every write
   is the op's, through its combination (the store into another buffer).
2. **No use of a destination in the region,** nor of a view of one,
   wherever the view is made (the store into a destination, the load
   through a subview).
3. **A destination is no buffer that the op reads,** nor a view of one,
   nor another destination (a destination that is a cast of `ins`).
4. **The requirement is met:** requirement $\sqsubseteq$ guarantee, as
   `md_exec::getRequirement` computes it for `pair_for`:
   - owner registers over the groups;
   - a `unique` visit of a directed list;
   - a `unique` visit of a list whose exclusions are only marked.
5. **The visit meets the algebra.**
   - A `fixed_order` value needs `visit stable`.
   - `combine atomic` refuses a `fixed_order` destination.
   - A `unique` visit refuses a `fixed_order` sum: the partial sums of
     its units have no fixed order.

   This moves the refusal of the deterministic mode from the lowering
   into the verifier.
6. **Exchange.** Under `unique` or `atomic`, every destination is
   `symmetric` or `antisymmetric` and every sum is `symmetric` (an
   antisymmetric sum).
7. **`owner` takes `directed`;** `privatized` is reserved; there is no
   `sorted` visit.
8. **Forms and shapes:**
   - one form at a time;
   - 7 buffers of i32, or 12 with `inner`, and two sizes;
   - the contract and the reach of the buffer form, and a cutoff within
     that reach;
   - one algebra and one exchange entry for each value, one weight for
     each sum, and one overwrite flag for each destination;
   - the region's arguments, its terminator, and its yields;
   - the known names of the algebras and of the exchange contracts, and a
     positive cutoff;
   - two destinations that are one buffer, which would lose one sum.

Each case is a valid op with one change, and checks the message.

`convert-md-exec-to-gpu` refuses what its context decides, each with a case
in `test/Conversion/MDExecToGPU/traverse-invalid.test`:
- the buffer form;
- a loop over groups left in `md_exec`, which a pipeline without the
  converter would give;
- a cutoff beyond the reach that the structure was built for. This is
  D357's query (`verifyNeighborCoverage`), now over any consumer, so that
  a hand-written op cannot claim any cutoff over any list;
- the outer list of a dual list;
- a traversal that visits each pair in both orders, written by hand over
  a matrix: the loops over a matrix stay `md_exec.pair_for` in this step.

Two refusals that the design listed cannot be reached, and are now
assertions:
- the schedule's refusal of a combination other than `atomic`. `owner`
  takes `directed`, which the pass refuses before it calls the schedule;
- its refusal of a `fixed_order` value. The verifier refuses one with
  `unique` or `atomic` (D290, D301), so the deterministic mode stops
  there, in `convert-md-exec-to-md-alg`'s output
  (`test/Conversion/MDExecToMDAlg/convert.mlir`).

The pass's check that the structure is of groups is an assertion too.
A `unique` type is a structure of groups: no other build may state it
(D413), and the verifier compared the op with the type.

The schedule refuses:
- a stable visit;
- a block that is not whole warps.

The serial oracle refuses the pruning in the loop.
`convert-md-alg-to-gpu` refuses the op. `convert-md-alg-to-scf` refuses the
structure form, and a buffer form whose cell is not a vector of 3 or 6.

## 5. The two lowerings

**The serial definition** (`emitTraversePairsSerial`,
`lib/Conversion/MDExecKernels/TraversePairs.cpp`) is written apart from
the schedule, in the order of the definition:
- for each unit (each active unit of an inner list), each entry of its
  block, and each of the 16 places of its group whose bit is set, the
  frames, the cutoff, and the region;
- the destinations added to at both ends, the sums in the type of the
  region's yield.

`convert-md-alg-to-scf` lowers the buffer form by it, with
`reverse-unordered-visits` (D409) taking every loop of an unordered visit
from the last step.

`convert-md-exec-to-gpu{traverse-oracle=...}` (`MDIR_TRAVERSE_ORACLE`) runs
it in place of the schedule, on the host, over copies of the lists, the
positions, the fields, the tables that the region reads, and the
destinations. The modes are:
- `serial`, the definition;
- `serial-reversed`, its visit reversed;
- `serial-magnitudes`, $\lvert t\rvert$ in place of each term $t$;
- `omitted`, no term.

**The SIMT lowering** (`lowerTraversePairsSIMT`, the same file) is the
code of `lowerGroupPairFor` and `emitGroupPairKernel`, which this change
removes from `MDExecToGPU.cpp` and `MDExecKernels.cpp`. Its sources change
only in what they name:
- the op `traverse_pairs` instead of `pair_for`;
- the algebra and the exchange from the op's attributes;
- the methods of `TraverseHost` instead of the members of `Lowering`;
- `frames` and `prune` instead of the pass's sets.

## 6. Where the SIMT lowering runs

A pattern that runs after `convert-md-exec-to-gpu` could not give `main`'s
bytes. Four things tie the schedule to the big pass:
- (a) the kernels are named by ordinals in the order of the lowering
  (D354);
- (b) the runtime's getters are called between the schedule's launches,
  and D406 compares the host code;
- (c) the work after the ops (the lookups, the powers, the slots of the
  readbacks, `releaseStack`) and the finishing of the kernels' own
  arithmetic would miss it;
- (d) the pass's analyses, which the refresh reads too.

(d) stays with the pass (Section 2). By ruling 1, `convert-md-exec-to-gpu` calls the
pattern where it meets the op, in program order, through `TraverseHost`
(`include/mdir/Conversion/TraversePairs.h`). The interface gives:
- the block size, the launches, `bringIn`, and the device's buffers;
- the structure's sizes, buffers, inner list, prune buffers, and frames;
- the pruning in the loop and its check;
- `finishSums`.

The kernels keep the purpose `pair_for`, so they keep their names. Items
(a) to (c), and the binding of the buffers as an op, are #888. The list of
the ops lowered in place (D408; `docs/device-primitives.md`, "Pass order")
gains this op.

The op therefore does not reach `md-alg-transform-collections`. For the
groups that loses nothing today. The lists reach the loop through the
runtime's handle, which D409's analysis sees as an escape (its rule 3).
The requirement that would allow an elision is in the type (D413).

## 7. The pipeline and the old path

```
... md-exec-assign-storage, md-exec-assign-streams, [stage bodies],
convert-md-exec-to-md-alg,
convert-md-exec-to-gpu{... fuse-pruning=true fuse-pruning-groups=G ...},
md-alg-transform-collections, convert-md-alg-to-gpu, ...
```

- **The new pass** takes no option.
- **Removed:**
  - `lowerGroupPairFor`;
  - the `unique` branch of `lowerPairFor`, which now refuses;
  - the sets of `findSharedFrames` and `findPrunedInLoop`, which became
    one analysis over the traversals that sets their attributes;
  - `emitGroupPairKernel` in `MDExecKernels.cpp`.
- **Stays in the big pass:** MATRIX, the ragged rows, PME's accessors, and
  the batched stages, which take a matrix alone.
- **The lit substitution** `%md_exec_gpu_passes` and
  `test/Conversion/MDExecToGPU/group-scatter.mlir` put the converter
  before the big pass.

## 8. What the template of #795 needs

The maintainer agreed the direction (ruling 4). Step 2 parses only the
degenerate form: one segment, `units(particle)`. The full syntax comes with
#795.

**(1) Lists of units, whose entries are a superset filtered by the cutoff
of site pairs.**
- The superset needs nothing new. The op owns the cutoff test, and every
  list already holds more entries than the cutoff selects. The f32
  search's extra entries in the shell are more of the same (the ruling on
  #795 of 2026-11-11).
- The test moves from the entry to each pair of sites.
- The list is over a unit set. Its coverage is D357's query, now over any
  consumer, with the covering reach of each segment: $R + 2r_\mathrm m$,
  $R + r_\mathrm m + \rho$, and $R + 2\rho$.

**(2) Three classes of cluster pairs in one buffer.**
- A segmented list type has one D413 contract per segment
  (`grouped_by(class)` across them, a value of D409's lattice). This is
  #795's `!mdrt.segmented_pairs`, with its contract in the type.
- The op takes one region per segment, with a visit and a requirement each.
- `md_exec.template_for` converts one body to one region.

**(3) A template's parameters as constants, with zero terms not
emitted.** A segment's table of site pairs gives each pair its constants
and a mask of its nonzero terms. The lowering clones, for each site pair,
only the slices of the yields in its mask. Folding is not relied on, since
$x \cdot 0$ does not fold in IEEE arithmetic.

**(4) Sites formed in the kernel.**
- A unit shape has $k$ members, $s$ sites, and constant weights
  $\mathsf W \in \mathbb R^{s\times k}$ whose rows sum to one:
  $\mathbf x_a = \mathbf x_o + \sum_m W_{am}(\mathbf x_m - \mathbf x_o)$.
- Forces reach the members as $\mathsf W^{\mathsf T}\mathbf f$ (rule A).
  For OPC: $k = 3$, $s = 4$, the fourth row $(1-2\alpha, \alpha, \alpha)$.
- Rule (B), for the written forces, is a pure `spread` region.
- The groups are $k = s = 1$, $\mathsf W = (1)$.

## 9. The same programs

The lowered programs of the branch were compared with those of its base by
the method of D406, once on 1c11a780 and again after the rebase on
9c10a859, with the same counts: `mdir emit --stage=lowered` with
`MDIR_GPU_BINARY=ptx MDIR_GPU_ARCH=sm_89 MDIR_COMPILE_CACHE=off
MDIR_SKIN_CACHE=off MDIR_SKIN_SEARCH=never`, through
`scripts/builder-equivalence/check.py compare --stages lowered`. That
gives the PTX of every kernel and the host's code.

| Corpus | Variants | Cases | Identical | Of which the same error |
|---|---|---:|---:|---:|
| the examples, the tests, the Amber suite, and the cases of the corpus | GPU; the run and the segments; mixed and double; as written, and with `neighbor_structure = "GROUPS"` (`--execution`) | 2,170 | 2,170 | 892 |
| the Amber suite, the cases, and the dodecahedron of `test/Driver/triclinic.test` with groups and with a dual list (`pruned_distance = 8.5`) | GPU; the run and the segments; mixed and double; as written, and with `MDIR_FUSED_PRUNE=off` (the pruning in the refresh), `always` (in the loop), and `MDIR_CHECK_PRUNING=1` | 888 | 888 | 64 |

Notes on the table:
- The cases written in Python run with the interpreter of the build. With
  the system's interpreter, which cannot import the package, the first
  comparison found their two errors different only in the path of the
  build.
- With `neighbor_structure = "GROUPS"`, 627 of the 1,085 cases lower a
  program with the groups. The others stop with the same error in both
  builds: their inputs are named in RUN lines that the corpus does not
  run, or the groups are refused there, as for the CPU or the
  deterministic mode.
- The Amber suite's controls take the groups and the dual list as
  written.

`md_dist` is lowered by `mdir-opt`, not by `mdir run`. The two programs of
D413 were lowered by `convert-md-dist-to-md-exec` and the device pipeline,
then by `--gpu-lower-to-nvvm-pipeline`:
- `test/Integration/dist-eam-exec.mlir`;
- `test/Integration/dist-lj-exec.mlir`.

Each ran on 1×2×3 domains, in double and mixed precision, with the matrix
and with `md-exec-choose-neighbors=kind=groups`. The branch's pipeline puts
`convert-md-exec-to-md-alg` before the device pass. The 8 outputs are
identical to those of the base, apart from the compile time that MLIR
records in each binary. They have 630 to 814 launches each.

**What the comparison caught.** In an earlier commit of the branch the
converter ran the analyses of the frames and of the pruning in the loop
(D268, D284). It ran them before the big pass moves the reciprocal sums
before the refreshes (D113), and the analyses of the big pass did not know
the traversal. On FactorIX two kernels changed:
- a loop over pairs lost its pruning in the loop;
- the fused integration kernel wrote the forces that the next loop
  overwrites.

The step was 3.5% slower. Section 2 records the fix. The programs of the
dodecahedron and of JAC had been identical.

## 10. The serial definition against the device

`test/Integration/Inputs/traverse_contract.py` evaluates the energy, the
virial, and the forces at the start of a run on an RTX 4090. The runs are
in five modes:
- the schedule;
- the serial definition;
- the serial definition with its visit reversed;
- the magnitudes of the terms;
- no term.

Coulomb is by a plain cutoff, so that no reciprocal sum spreads with
atomics. The loops over tuples take a row a particle (the oracle sets
`tuples-once=false`). What differs between two runs is then the
traversal's alone.

For each component $X$, the magnitude is
$M = X_\mathrm{magnitudes} - X_\mathrm{omitted} = \sum_k\lvert t_k\rvert$.
Two orders of a sum of $n$ terms differ by at most
$2\gamma_{n-1}M$, with $\gamma_m = mu/(1-mu)$ (D406). The test holds the
schedule and the reversed visit to the serial order within
$2\gamma_{n-1}M + 4u\max M$, with these values:
- $n = 1024$. The particles within the reach of one, at most 0.11 per
  cubic angstrom over a sphere of 11 Å, are 613, and they bound the terms
  of a force. A lane's sums over its units bound those of an energy.
- $u = 2^{-24}$ in mixed precision and $2^{-53}$ in double.

The test fails if the traversal contributes nothing, or if the magnitudes
do not bound the sums. The largest difference, as a fraction of its
bound:

| System | List | Precision | Schedule | Reversed visit |
|---|---|---|---:|---:|
| dodecahedron (1,209 atoms) | groups | mixed | 0.0087 | 0.0038 |
| | | double | 0.0045 | 0.0050 |
| | dual, pruned in the refresh | mixed | 0.0081 | 0.0038 |
| | | double | 0.0046 | 0.0050 |
| | dual, pruned in the loop (D284) | mixed | 0.0087 | 0.0038 |
| | | double | 0.0045 | 0.0050 |
| JAC (23,558 atoms) | groups | mixed | 8.6e-5 | 6.6e-5 |
| | | double | 6.1e-4 | 1.2e-3 |
| | dual, pruned in the refresh | mixed | 8.7e-5 | 7.4e-5 |
| | | double | 6.6e-4 | 1.3e-3 |
| | dual, pruned in the loop | mixed | 8.5e-5 | 7.4e-5 |
| | | double | 7.2e-4 | 1.4e-3 |

The worst component is a force in mixed precision, and in double on JAC
the virial.

**A mutation that must fail.** The serial definition was changed to ignore
the sign of an antisymmetric destination, and the case of the dodecahedron
was run. The schedule then differed from the serial definition by
$1.3\times10^4$ times the bound (force 960, $3.3\times10^5$ against 25).
The reversed visit, which has the same mutation, stayed within its bound,
at 0.0033 of it. The mutation was reverted.

## 11. The sanitizer and the timing

**The sanitizer tier** (`lit -Dsanitize=1 test/Sanitizer`, RTX 4090):
23 passed and 2 were unsupported (JAX and the frame batches). The tests
include `fused-prune-gpu.test` (the pruning in the loop, under memcheck,
racecheck, and initcheck), `group-scatter-gpu.test` (the table of a block
on groups), `groups-empty-gpu.test`, and `triclinic-gpu.test`. The run held
the device for 181 s.

**Timing** by the timing protocol, on RTX 4090s:
- two jobs, each with `--idle`;
- in each job, three rounds of the base (1c11a780) and the branch,
  interleaved;
- JAC NVE (100,000 steps) and FactorIX NVE (50,000 steps), with groups and
  the dual list, as the Amber suite sets them;
- the rate of the second half of each run, as `mdir run` reports it.

The rates are in ns/day. For each system, job, and build, the table gives
the arithmetic mean of the three rounds' second-half rates, with no
trimming:

| System | Job | Base | Branch | Branch / base |
|---|---|---:|---:|---:|
| JAC | 1 | 1268.0 | 1266.8 | 0.999 |
| JAC | 2 | 1255.9 | 1256.1 | 1.000 |
| FactorIX | 1 | 563.8 | 563.7 | 1.000 |
| FactorIX | 2 | 558.2 | 558.4 | 1.000 |

Within a job the rounds spread by up to 0.6%, and the two jobs differ by
1%. The clocks that `gpu-run` sampled under load were:
- job 1: SM 210–2,775 MHz, at most 64 °C;
- job 2: SM 210–2,745 MHz, at most 61 °C.

No slowdown is beyond the spread. The programs are identical, so this is
a guard. The sanitizer tier and the timing ran on the branch before its
rebase onto 9c10a859. The lowered programs were compared again after the
rebase, and they are identical (Section 9).

## Tests

- `test/Dialect/MDAlg/traverse.mlir`: both forms, in the custom and the
  generic form.
- `test/Dialect/MDAlg/traverse-invalid.mlir`: one case for each refusal of
  the verifier (33).
- `test/Conversion/MDExecToMDAlg/convert.mlir`: the converter on a dual
  list with two loops, and the refusal of the deterministic mode's
  `fixed_order`.
- `test/Conversion/MDExecToGPU/traverse-invalid.test`: the refusals of
  the lowerings and of the pipeline (12).
- `test/Integration/md-alg-traverse-serial.test`: the serial definition on
  the host on a list written by hand, the outer and the inner list, in
  order and reversed, with the values worked out.
- `test/Integration/md-alg-traverse-contract-gpu.test` and
  `md-alg-traverse-contract-jac-gpu.test` (`amber-suite`): the contract of
  Section 10.
- `scripts/builder-equivalence/check.py --execution KEY=VALUE`: each case
  of the corpus with keys of `[execution]` set as well, such as
  `neighbor_structure=GROUPS`.

## Lines

The code, without the tests and the documents, gained 2,463 lines and
lost 971:
- the op, its verifier, and its effects (`MDAlgOps.td`, 178;
  `MDAlgDialect.cpp`, 244);
- the converter (`MDExecToMDAlg.cpp`, 118);
- the interface (`TraversePairs.h`, 169);
- the lowerings (`TraversePairs.cpp`, 1,254). About 830 of these are the
  schedule and the kernel, moved; about 330 are the serial definition;
- `MDExecToGPU.cpp` (+319 −536: the old path removed, the host interface,
  the oracle in place, and the analysis over the traversals);
- `MDExecKernels.cpp` (−386, the kernel moved);
- D357's query over any consumer (`NeighborCoverage.cpp`).
