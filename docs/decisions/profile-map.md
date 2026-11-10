# D[profile-map]

Recorded at its merge.

**The compiler records the source of every parallel region of the host in a
profile map, written as JSON by `mdir emit --profile-map=FILE` and by
`[output] profile_map`, and the CPU sampler's tools read it instead of
classifying regions by the functions they call.** Issue #840 (the CPU part;
the GPU part waits for #795).

*Why.* `scripts/benchmarks/cpu-breakdown/prof.py` gave each region
`mdir_run..omp_par.N` a part from what it calls: radial tables, `powi`,
`erf`, trigonometric functions, `sqrt`. The rules broke on the f64 loops
over pairs (#833) and needed an order fix (#836). On JAC on 2 threads,
with the rules of #836, the three loops over the 1-4 pairs are "pairs" (they
call `powi` and no trigonometric function), 20 of the 28 loops of the
constraints in the mixed mode and 17 in the double mode are "other", and
the region `mdir_run..omp_par`, which has no number, is not seen; the map
gives them "bonded", "constraints", and its record.

*What.* A record is a dictionary on a location: the metadata `mdir.profile`
of a `FusedLoc`. `md-inline` locates each op it copies from a potential's
body in that potential (`potential: <name>`); `convert-md-exec-to-loops`
gives each op of md_exec its record (the op, its set, the potentials and
terms its kernel holds) before the buffers lose the names of their sets,
and every op its lowering creates at that location, the `scf.parallel`
among them, inherits it. The loops of the templates take the record of
their function. The driver copies the lowered module, puts a mark with an
index at the start of each `omp.parallel`, translates the copy to LLVM IR,
and pairs each record with the function its mark is in. The format, the
keys, and the parts that `prof.py` takes from them are in
[profile-map.md](../profile-map.md).

*Alternatives.*
- Pairing the regions with the outlined functions by the order of the calls
  of `__kmpc_fork_call` in each function: no copy, but it rests on the order
  in which the translation emits blocks and outlines regions. The mark tells
  the function directly.
- Discardable attributes on `scf.parallel`: `convert-scf-to-openmp` drops
  them. Locations are the one thing every pass carries.
- Readable names of the outlined functions (`..pairs.ww`): the names are
  part of the bitcode, which is the key of the object cache of the host, so
  every key would change once. The map is the contract; names are left as
  LLVM gives them.
- Keeping the classifier as a fallback: a fallback that guesses is what the
  map replaces, and a profile without a map shows the regions by their
  names. It is removed.

*What does not change.* The lowered module (`mdir emit --stage=lowered`)
and its LLVM IR are the same byte for byte, so are the keys of the compile
caches: locations are dropped by a translation without debug information,
and the mark is in a copy. Checked over the corpus of
`scripts/builder-equivalence/check.py` (see the PR).

*Left open.* The loops of the forces name the potential `energy` and no
terms: the builder writes the sum of the terms as one potential whose ops
carry no term (the Lennard-Jones and the Coulomb of the pairs in one
kernel). The kernels of a device take the same records after #795.
