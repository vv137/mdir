/* Included by mdrt_cuda.c after the allocation and activation helpers.
 * These functions belong only to the new template route. */
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mdir/Runtime/TemplateMembership.h"

struct TemplatePairs {
  struct MdrtTemplatePairLayout layout;
  void *pairs;
  void *members;
};

static void allocateTemplatePairs(struct TemplatePairs *pairs) {
  allocationTag = "template pairs: shared outer/inner WW/SW/SS";
  pairs->pairs = mgpuMemAlloc((uint64_t)pairs->layout.words * sizeof(int32_t),
                             NULL, false);
  allocationTag = NULL;
}

static int64_t createTemplatePairs(int64_t waters, int64_t atoms, int64_t entries) {
  struct TemplatePairs *pairs = calloc(1, sizeof(*pairs));
  if (!pairs || !mdrtTemplatePairLayout(waters, atoms, entries, &pairs->layout)) {
    fprintf(stderr, "mdrt: invalid template-pair capacity or host allocation failure\n");
    abort();
  }
  allocateTemplatePairs(pairs);
  recordCallHandle(pairs);
  return (int64_t)(intptr_t)pairs;
}

/* The compiler passes host members, already renumbered like the state.
 * Upload once at initialization, never at a step's refresh. */
int64_t _mlir_ciface_mdrtTemplatePairsCreateMembers(
    struct Buffer2 *members, int64_t particles, int64_t entries) {
  int64_t waters = members->sizes[0], sites = members->sizes[1];
  if (particles < 0 || particles > INT32_MAX || waters < 0 ||
      (sites != 3 && sites != 4) || waters > particles / sites) {
    fprintf(stderr, "mdrt: invalid template membership shape\n");
    abort();
  }
  int64_t atoms = particles - sites * waters;
  int64_t count = 3 * waters + atoms;
  int32_t *host = malloc((size_t)(count ? count : 1) * sizeof(int32_t));
  if (!host || !mdrtTemplateMembership(
          particles, waters, sites,
          waters ? (const int32_t *)members->aligned + members->offset : NULL,
          members->strides[0], members->strides[1], host, count)) {
    fprintf(stderr, "mdrt: invalid or overlapping template membership\n");
    abort();
  }
  int64_t handle = createTemplatePairs(waters, atoms, entries);
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  allocationTag = "template pairs: fixed real-unit membership";
  pairs->members = mgpuMemAlloc((uint64_t)(count ? count : 1) * sizeof(int32_t),
                               NULL, false);
  allocationTag = NULL;
  /* Synchronous only at initialization: the temporary host memory can be
   * released immediately, with no outstanding borrowed source. */
  if (count)
    check(cuMemcpyHtoD((CUdeviceptr)pairs->members, host,
                       (size_t)count * sizeof(int32_t)), "cuMemcpyHtoD");
  free(host);
  return handle;
}

void _mlir_ciface_mdrtTemplatePairsMembers(struct Buffer1 *result, int64_t handle) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  result->allocated = pairs->members;
  result->aligned = pairs->members;
  result->offset = 0;
  result->size = 3 * pairs->layout.waters + pairs->layout.atoms;
  result->stride = 1;
}

void _mlir_ciface_mdrtTemplatePairsBuffer(struct Buffer1 *result, int64_t handle) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  result->allocated = pairs->pairs;
  result->aligned = pairs->pairs;
  result->offset = 0;
  result->size = pairs->layout.words;
  result->stride = 1;
}

int64_t mdrtTemplatePairsCapacity(int64_t handle) {
  return ((struct TemplatePairs *)(intptr_t)handle)->layout.entries;
}

/* Growth discards every class range. A caller that grows must rebuild
 * all three classes, even if only one mobility group triggered the build.
 * No wait, allocation or host readback occurs on a refresh that fits. */
int64_t mdrtTemplatePairsGrow(int64_t handle, int64_t required) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  if (required >= 0 && required <= pairs->layout.entries) return 0;
  int64_t capacity = required;
  if (required >= 0 && required <= INT32_MAX)
    capacity = required + (INT32_MAX - required < required / 4
                               ? INT32_MAX - required : required / 4);
  struct MdrtTemplatePairLayout next;
  if (!mdrtTemplatePairLayout(pairs->layout.waters, pairs->layout.atoms,
                              capacity, &next)) {
    fprintf(stderr, "mdrt: template-pair capacity exceeds the i32 index bound\n");
    abort();
  }
  enter();
  check(cuCtxSynchronize(), "cuCtxSynchronize");
  releaseReplaced(pairs->pairs);
  pairs->layout = next;
  allocateTemplatePairs(pairs);
  return 1;
}
