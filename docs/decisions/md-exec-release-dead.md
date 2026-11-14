# D[md-exec-release-dead]

Recorded at its merge.

**The buffers of the storage of `md_exec` that the loop over steps and the code after it never use are released after their last use in the setup, by a pass of their own, `md-exec-release-dead-storage`, after `convert-md-alg-to-gpu`. The proof is the liveness of D412 in the mode `kills=proven`: a buffer is released after the op of the function's body that holds the last point where it is touched or live, the waits and joins that complete work on it counted as touches, if that op comes before the loop; a buffer that is pinned, carried, read after the loop, already released, unused, shared with the host, or only copied into keeps its storage. On D411's four programs the peak device memory in use falls by every byte released where the peak is in the loop: 38,456,380 bytes at STMV (2.64%) and 2,978,440 at JAC (9.38%). On the two dodecahedra the peak moves into the setup, and falls by 220,264 (19.1%) and 135,916 bytes (4.17%). The peak with the runtime's pool falls much less, 8,536,760 bytes at STMV (0.56%) and 188,464 at JAC (0.57%): the pool gives a freed block only to a request of its size, and most released blocks match none. The kernels are unchanged, the host only frees, and the five checkpoints are identical to the bit.** Issue #885, after D412 (#880) and D411 (#876).

## 1. The rule

The pass runs on each host function with a loop over steps, on the
analysis `StorageLiveness` (D412) with `kills=proven`. For a buffer of the
storage (a `gpu.alloc` of the function's body before the loop), it
releases the buffer after the op of the body that holds the buffer's last
occupied point, if that op comes before the loop. A point occupies a buffer
where the buffer is touched there, live into it, or live out of it. The
points follow the order of the code, so no later point needs the buffer
or its storage. The touches include the completions that the analysis
adds:
- the waits of an asynchronous op's token;
- the join of a side stream for the ops between its begin and its end;
- the next wait after an external function that enqueues work on the
  buffer (the transforms of cuFFT, `mdrtDeviceTuplesOrderEnd`; #883's
  review).

The release goes after the op of the body, not inside it. A last use
inside a loop or a branch of the setup is released after that loop or
branch.

**Occupancy, not liveness alone, decides.** A buffer that the code
overwrites whole after the loop and then reads is dead from its last use
in the setup to that write. Its storage is still needed there. Releasing
where the buffer is not live would free it before that write. Under
`kills=stores` this happens for a launch that writes it; under
`kills=proven` it happens for a fill or a copy. Since occupancy includes
every access, the kill mode cannot change a release. The pass reads the
analysis in `kills=proven`, as #885 asks.

A buffer keeps its storage to the end, as on `main`, when it is:
- **pinned** by the analysis: a raw pointer of it in the host's code, a
  return of it, a token never waited for, a side stream never joined,
  enqueued work with no later wait, or a call to an external function
  that may keep the pointer (any function not in D412's table);
- **carried** by the loop, or **read after it**;
- **occupied at a point of the step**;
- **already released** by the code;
- **unused**, which the lowering removes;
- **shared with the host** (`host_shared`, not a block of the pool) or
  allocated on a stream;
- **only copied into**, never read. The canonicalization of `gpu.memcpy`
  erases a copy into a buffer that nothing else uses, and then the
  allocation. Two buffers on each program are of this kind: tables copied
  to the device in f32 that no kernel reads (of zero rows at STMV). A
  release would be a use of its own and keep both.

Today, the analysis already makes a pinned, carried, or read-after-the-
loop buffer occupied in the step or after the loop. The explicit checks
of the pass state the conditions and keep them if the analysis changes
(Section 4, the mutations).

The release is a `gpu.dealloc` of the buffer cast to the default memory
space, as the lowering's own releases are written. It lowers to
`mgpuMemFree`, which puts the block in the runtime's pool without
waiting. Later work on the shared stream is ordered after the work that
used the block.

### Why this place

#885 named `md-exec-assign-storage`, where the fields are given their
storage. The analysis does not run there. It reads the host's code as
`convert-md-exec-to-gpu` leaves it:
- every buffer is a `gpu.alloc`;
- every kernel is a `gpu.launch` whose body names the buffers it touches;
- the waits and the calls to the runtime are explicit ops.

At `md-exec-assign-storage` the buffers are still fields of `md_exec`
ops, and the waits, the side streams, and the external calls do not exist
yet. The pass runs after `convert-md-alg-to-gpu`, where the host's code
is complete: no later pass adds a use of a buffer of the storage in the
host's code. On the four programs the analysis with `kills=proven`
prints the same there as at the slot of `md-alg-transform-collections`
(D412's IR), to the byte.

## 2. Peak device memory

**How it was measured.** `mdir run` of the four control files of D411
(200 steps, `[output] energy_interval = 0`, mixed precision) on an RTX
4090, with `MDIR_DEVICE_MEMORY=2`, built from `main` 1c11a780 and from
this change. The bytes are the sums of the runtime's tables of the blocks
at each peak. `main`'s are D411's to the byte. To place each peak and
attribute each block, a local build, not committed, also wrote a trace of
`mgpuMemAlloc` and `mgpuMemFree` with the number of launches so far, as
D411's did. A replay of each trace gives the runtime's peaks to the byte.

| | Dodecahedron, groups | Dodecahedron, matrix | JAC, groups | STMV, groups |
|---|---:|---:|---:|---:|
| buffers released | 14 | 13 | 17 | 9 |
| bytes released | 257,348 | 228,332 | 2,978,440 | 38,456,380 |
| of which, D412's buffers that the loop never touches | 120,900 | 91,884 | 2,544,264 | 38,415,420 |
| of which, a buffer of PME's setup (7,840 and 40,960 f64) | 62,720 | 62,720 | 327,680 | 0 |
| of which, the partial sums of the setup's reductions (1,024 f64 and 1,024 × 3 f64) | 73,728 | 73,728 | 106,496 | 40,960 |
| peak in use, `main` | 1,153,960 | 3,256,372 | 31,746,976 | 1,455,830,804 |
| peak in use, this change | 933,696 | 3,120,456 | 28,768,536 | 1,417,374,424 |
| **saved in use** | **220,264 (19.09%)** | **135,916 (4.17%)** | **2,978,440 (9.38%)** | **38,456,380 (2.64%)** |
| peak in use in the loop, this change | 896,612 | 3,028,040 | 28,768,536 | 1,417,374,424 |
| peak with the pool, `main` | 1,190,604 | 3,268,864 | 32,931,908 | 1,518,503,948 |
| peak with the pool, this change | 1,147,080 | 3,254,356 | 32,743,444 | 1,509,967,188 |
| **saved with the pool** | **43,524 (3.66%)** | **14,508 (0.44%)** | **188,464 (0.57%)** | **8,536,760 (0.56%)** |

The buffers released are D412's six at STMV (38,415,420 bytes, its bound)
and their kinds elsewhere. They also include the partial sums of the
setup's reductions and, on the dodecahedra and JAC, an f64 buffer of
PME's setup. D412 counted both outside the storage of `md_exec`, in the
owners of D411. Each column of bytes released is their sum, to the byte.

**In use.**
- **JAC and STMV.** The peak stays at the first build of the lists in the
  loop (D411) and falls by every byte released.
- **The dodecahedra.** The peak in the loop also falls by every byte
  released. The program's peak moves into the setup, before the last
  release:
  - with the groups, to the 109th launch, 37,084 bytes above the loop's
    new peak (the last release is at the 132nd);
  - with the matrix, to the 66th launch, 92,416 bytes above it (the last
    release is at the 90th).

**With the pool.** The runtime keeps a freed block for a later request of
the same size only (D411). A release therefore lowers the peak with the
pool only where it serves a request that would otherwise have taken new
memory from the driver. In the replays:
- **The blocks of the particles' order and its permutation** (n × i32,
  and the copy of an n × f32 input) are the size of the groups build's
  `key`, `sorted`, and `gplace` (D411). Before the peak with the pool, the
  three released blocks of that size take the place of two new blocks
  from the driver: 2 × 4,268,380 bytes at STMV and 2 × 94,232 at JAC. At
  the peak three blocks of that size are idle, against two on `main`.
- **The other blocks** (the n × f64 buffers, PME's buffer, the partial
  sums) match no request of the loop. They stay idle in the pool for the
  rest of the run. At STMV that is 25,651,240 bytes. The pool's slack at
  its peak rises from 62,681,476 to 92,601,096 bytes.
- **The dodecahedra.** Their peak with the pool stays where it is on
  `main`, at the 139th launch with the groups and the 97th with the
  matrix, after the last release. The saving is the released blocks that
  a request took: 3 × 4,836 bytes, and with the groups one 29,016-byte
  block as well.

So the bound of D412 is reached in use. With the pool, the saving is
the released blocks that a later request of their size takes. Keeping the
rest of the saving would need one of these:
- a release of a block that no later request asks for back to the driver,
  as a growth of the runtime's structures is released (D312);
- a pool that serves a request from a larger block (D411's slack).

Both are changes of the runtime's pool, outside this decision (open
item).

## 3. The same kernels, the same trajectories

- **The kernels.** `mdir emit --stage=lowered` with D406's settings
  (`MDIR_GPU_BINARY=ptx MDIR_GPU_ARCH=sm_89 MDIR_COMPILE_CACHE=off
  MDIR_SKIN_CACHE=off MDIR_SKIN_SEARCH=never`) on ten programs, for `main`
  and this change. The programs are the four of Section 2 in mixed and in
  double precision, FactorIX NVE with the groups, and JAC NVE with the
  matrix. The `gpu.binary` op of every kernel, its PTX, is the same to the
  byte: 131 to 408 kernels a program. The host's code differs only by the
  added calls of `mgpuMemFree`, one line each, 8 to 17 a program. This was
  checked by a diff of the host's code with the names of values
  normalized. No line is removed.
- **The trajectories.** The five checkpoints were run in the deterministic
  mode with the compile cache off, `main` and this change alternated in
  one job on an RTX 4090: JAC NVE, Factor IX NPT, and Cellulose NVE of the
  Amber suite (200 steps each), the ala3 minimization, and argon (200
  steps, on the GPU). Every dataset of the final checkpoint is identical
  to the bit, 5 of 5. The datasets are the positions, velocities, box,
  forces where written, and the fingerprints.

## 4. Validation of the releases

- **compute-sanitizer, memcheck**, on the four programs of Section 2 with
  this change (STMV for 25 steps, the others for 200), on an RTX 4090.
  There are two settings:
  - **the pool off** (`MDIR_DEVICE_POOL_BYTES=0`): every release goes back
    to the driver, so a later access to a released buffer is an access to
    freed memory;
  - **the default pool.**

  There are 0 errors in all eight runs. The job held the device for 3.6
  minutes.

  As a positive control, a mutant that releases every buffer that is not
  pinned before the loop was run on the dodecahedron with the groups, with
  the pool off. memcheck reports 1,210 errors (invalid global reads).
- **Negative tests.** `test/Dialect/MDExec/release-dead-storage.mlir` has
  one function for each condition. The function breaks the condition and
  must keep its buffer:
  - read after the loop;
  - overwritten whole by a launch after the loop and then read;
  - filled after the loop and then read;
  - read in the loop, and read there through a view;
  - a raw pointer;
  - returned;
  - given to an external function that may keep it, directly or through a
    function of the module;
  - enqueued work whose next wait is after the loop, and one with no
    wait;
  - a token waited for in the loop, and a token never waited for;
  - a side stream joined after the loop;
  - already released;
  - shared with the host, and unused;
  - only copied into.

  Five functions must be released, each at its place:
  - after the last launch of the setup;
  - after the wait that follows an enqueuing call, not at the call;
  - after the wait of an asynchronous fill;
  - after the join of a side stream;
  - after a loop and after a branch of the setup that hold the last use.
- **Mutations** (local, not committed):
  - **Releasing under `kills=stores`** changes no release: the test
    passes, and the four programs give the same IR to the byte. The
    occupancy that decides includes every access, whatever the kill mode
    (Section 1). The requirement of #885 that this mutation fail a test
    cannot be met by this rule. This is stated as an open item.
  - **A rule that reads the liveness alone**: the last occupied point
    before the loop, the accesses after it ignored, and no check of a read
    after the loop. Under `kills=stores` it fails at
    `@overwritten_after_loop` and `@filled_after_loop`. Under
    `kills=proven` it fails at `@filled_after_loop` only, since a launch's
    write is not a proven kill and keeps the buffer live through the loop.
    This is where the mode matters: a rule by liveness is unsafe in both
    modes, and more so under `stores`.
  - **The completion of enqueued external work ignored** (in the
    analysis): fails at `@enqueued`, `@enqueued_wait_after_loop`, and
    `@enqueued_no_wait`.
  - **The waits of tokens ignored** (in the analysis): fails at
    `@async_wait` and `@async_wait_in_loop`.
  - **The release placed in the region of the last point**, not after its
    op of the body: fails at `@setup_loop` and `@setup_branch`.
  - **A buffer already released**, released again: fails at `@released`.
  - **The checks of pins, of carried buffers, and of the step's
    occupancy, removed**: the test passes. These are equivalent mutants
    today: the analysis makes such a buffer occupied in the step, and the
    last occupied point then falls in the loop.

## 5. Open

- **The pool.** At STMV, 25.7 MB of the 38.5 MB released stays idle in
  the pool for the run. It reaches the driver's footprint only with a
  release to the driver, or a pool that splits blocks (Section 2).
- **The mutation of #885.** "Releasing under `stores` must fail a test"
  holds for a rule that reads the liveness alone, not for the rule by
  occupancy, which does not depend on the kill mode (Section 4).
