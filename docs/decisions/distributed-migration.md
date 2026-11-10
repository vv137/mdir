# D[distributed-migration]

Recorded at its merge.

**A step loop on domains carries one linear value, `!md_dist.state`, that holds an ownership epoch, its halo, the halo's reference positions, and every live per-particle field; a state is formed only by `md_dist.pack` under a proof that its halo covers its positions, and the in-process executor runs `md_dist` ops inside `scf.for` and `scf.if` through it.** #38, after D313 (migration primitives) and D332 (the collective skin predicate).

*The problem.* The verifier proves ownership epochs (V5), copy versions
(V2), and coverage by following defining ops, and the in-process executor
ran the ops of `md_dist` in one block only. A loop that carried a layout, a
halo, and fields as separate block arguments could be neither verified nor
lowered: a block argument has no defining op, and a field could be carried
apart from its layout.

*The state.* `!md_dist.state<halo-type, cutoff, fields...>` names the
halo type (decomposition and reach), the cutoff the halo is proved to cover,
and the types of the carried fields; the first field is the positions.

- `md_dist.pack %l, %h, %cell fields(%x, %v, ...)` forms a state. Every field
  is a complete owned field of `%l`, the halo is of `%l`, and the
  positions `%x` are covered by `%h` at the state's cutoff by one of three
  witnesses: `%h` was built at `%x` (`md_dist.halo`); `%h` and `%x` come
  from one `md_dist.unpack` of a state with that cutoff; or the pack is in
  the then-region of an `scf.if` whose condition is a `md_dist.covered` of
  `%h`, `%x`, and a cutoff at least the state's.
- `md_dist.unpack %s, %cell` gives the layout, the halo, the halo's
  reference positions, and the fields, with their provenance: `layoutOf`
  and `haloOf` of the results are the unpack's layout and halo, and
  `md_dist.covered` and `md_dist.neighborhood` accept the unpacked halo
  with the unpacked reference and cell.
- A state is linear: it has exactly one use, in the block that defines it
  (an `unpack`, an `scf.yield`, or the init of an `scf.for`). It cannot be
  read twice or dropped on one path.
- Only states cross control flow: any other `md_dist` value as an
  iteration argument or a result of `scf.for` or `scf.if` is refused with
  the reason.

*Coverage of a neighborhood (V10).* `md_dist.neighborhood` at positions
other than the halo's build positions needs the same witness as `pack`.
Before, the verifier only compared the cutoff with the reach.

*The step.* Integrate on owned rows (`map_particles`), test
`md_dist.covered`, and branch on the agreed bit with `scf.if`: if covered,
repack the old layout and halo with the new fields (the candidate map is
reused, and each force evaluation still copies the current positions
through it with `halo_copy`: payload freshness is independent of map
reuse); else `migrate`, `redistribute` every carried field (positions with
`image`), build a new halo, and pack the new epoch. Since `pack` requires
every field of the state type in one layout, a migration that leaves a
field behind is rejected (V5); there is no partially migrated state.
Positions are not wrapped between rebuilds, because the halo's images are
explicit and `covered` treats a wrap as a move.

*Rank-uniform choices.* The grid, the reach, the cutoff, the carried
fields, and the traversal are in the types and attributes of one program
text, so every domain makes the same choices. The rebuild decision is the
one value of `covered`, reduced over every domain including empty ones.
The capacity of ghost buffers is the largest count of any domain
(`mdrtDistHaloCapacity`), so every domain allocates alike. A runtime
failure (a non-finite position, a cell that differs from the halo's, an
identity that is not a bijection) is detected before any domain writes
the new epoch, and stops the team. These are the contracts a transport of
#828 must keep: a reduced bit for the branch, a reduced maximum for
capacities, and an agreed failure.

*The in-process lowering.* A state lowers to a tuple: the layout handle,
the halo handle, the reference positions, and one buffer per field;
`scf.for` and `scf.if` carry the tuple through MLIR's structural type
conversion. `pack` copies the buffers into the tuple and retains the two
handles (the runtime counts references); `unpack` copies them out into
fresh allocations of its block and releases the tuple. Every other buffer
of a step is freed at the end of its own block. The copies keep each
buffer of a step a `memref.alloc` of its block, which the halo events of
D310 require, so that a halo start and its wait inside a loop body are
proved as before: no buffer or handle of a transfer is packed, released,
or deallocated before its wait. Moving buffers into the state instead of
copying is an optimization for a later strategy, with this as its
baseline. On the route through `md_exec` (D290) the particle sets of a
domain are named by epoch, and the body of the loop becomes a function of
its own called by the loop, because a set of `md_exec` has one size in a
function and its storage is assigned for the whole function; without this
the storage of the first evaluation was reused after a migration and
overflowed.

*Alternatives.* Separate block arguments for the layout, the halo, and
each field (rejected: the verifier loses the epoch at the block argument,
and a loop could carry a field of another epoch). One opaque runtime
handle for the state (rejected: lifetimes and buffers move out of the IR,
where the D310 proof of completion before retirement lives). One op that
tests, migrates, and rebuilds inside the runtime (rejected as the
baseline: the branch and the migrated fields would not be visible to the
verifier; it can be a later expansion, like `ExposeValidity` for
`md_exec.refresh_neighbors`).

*Not in this decision.* The control-file table `[distribute]` (Section 13
of [distributed-ml2.md](../distributed-ml2.md)) needs the driver to emit
this loop, which is the boundary recorded in that document; MPI and GPU
transports (#828); asynchronous overlap (#481, #536); bonded terms across
domains (#41); a neighbor structure of `md_exec` kept across steps on
domains.

*Validation* ([distributed-ml2.md](../distributed-ml2.md), Section 8.1).
A 200-step Lennard-Jones trajectory of 64 particles on 2 x 2 x 2, 16 x 1 x 1,
and 3 x 2 x 1 grids, with 77 crossings of faces, 6 of edges, and 8 of
corners and 11 rebuilds of the halo, against the same trajectory on one
domain: in double the positions agree to 8.1e-15 of the edge (bound 1e-12),
the velocities to 1.1e-13 of the largest component (1e-11), and the
energies to 1.4e-14 of the kinetic energy (1e-12); in mixed to 9.9e-7,
1.2e-5, and 1.2e-6 (bounds 1e-5, 1e-4, 1e-5), the same as on one domain of
`md_dist`. Over 2000 steps the total energy changes by 1.0e-5 of the
kinetic energy on domains and 1.2e-4 on one domain (bound 2e-3).
