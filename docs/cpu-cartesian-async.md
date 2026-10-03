# D[cpu-cartesian-async]: fixed-layout Cartesian execution

Status: implementation in progress; extends PR #34, not production dynamics.

The CPU LJ snapshot tool gains `--grid=auto|Px,Py,Pz` (default auto) and
`--halo=sync|async` (default sync). Auto enumerates integer factor triples
and minimizes a box-aware expanded-domain volume estimate. Explicit grids
must have positive dimensions whose product equals the MPI rank count.
Snapshot input and numerical stdout stay unchanged. Diagnostics go to stderr;
no files are overwritten and no TOML control keys are added.

The implementation will expose `--emit=dist`: a verified, straight-line
execution plan using `mdrt.event`. The reference executor interprets that plan
and dispatches existing compiled `md_exec` pair kernels. This first bridge is
not a general MPI lowering pass or a field-origin/contribution verifier.

Direct periodic owner-to-replica routing includes face, edge, corner, and
more distant domains when the cutoff spans a domain. IDs and counts define a
fixed map before evaluation. Async payload exchange owns its pack buffers;
completion includes receive completion, unpack, and send-buffer release.
Interior and boundary computations use disjoint owned-center subsets.

Validation will cover explicit and auto grids, empty domains, all periodic
interfaces, mixed/double precision, sync/async agreement, and invalid plans.
No migration, production trajectory, distributed PME, GPU transport, UCX,
NVSHMEM, or measured overlap speedup is implied by this scope.
