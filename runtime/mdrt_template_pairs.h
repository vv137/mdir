/* Included by mdrt_cuda.c after the allocation and activation helpers.
 * These functions belong only to the new template route. */
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mdir/Runtime/TemplateMembership.h"
#include "mdir/Runtime/TemplateExclusions.h"

struct TemplatePairs {
  struct MdrtTemplatePairLayout layout;
  void *pairs;
  void *metadata;
  int64_t metadataOffsets[TEMPLATE_METADATA_SLOTS];
  int64_t metadataSizes[TEMPLATE_METADATA_SLOTS];
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
    struct Buffer2 *members, int64_t particles, int64_t entries,
    struct Buffer2 *excluded) {
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
  struct MdrtTemplateExclusions graph = {0};
  if (excluded->sizes[0] < 0 || excluded->sizes[1] != 2 ||
      !mdrtTemplateExclusions(particles, waters, sites, host, excluded->sizes[0],
          excluded->sizes[0] ? (const int32_t *)excluded->aligned + excluded->offset : NULL,
          excluded->strides[0], excluded->strides[1], &graph)) {
    fprintf(stderr, "mdrt: invalid template solute exclusion graph\n");
    abort();
  }
  int64_t handle = createTemplatePairs(waters, atoms, entries);
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  int64_t sizes[TEMPLATE_METADATA_SLOTS] = {count, particles, atoms + 1, graph.count};
  int64_t words = 0;
  for (int slot = 0; slot != TEMPLATE_METADATA_SLOTS; ++slot) {
    pairs->metadataOffsets[slot] = words;
    pairs->metadataSizes[slot] = sizes[slot];
    words += sizes[slot];
  }
  int32_t *metadata = malloc((size_t)words * sizeof(int32_t));
  if (!metadata) {
    fprintf(stderr, "mdrt: out of host memory for template metadata\n");
    abort();
  }
  if (count) memcpy(metadata, host, (size_t)count * sizeof(int32_t));
  int32_t *slots = metadata + pairs->metadataOffsets[TEMPLATE_PARTICLE_SLOTS];
  for (int64_t p = 0; p < particles; ++p) slots[p] = -1;
  for (int64_t row = 0; row < count; ++row) slots[host[row]] = (int32_t)row;
  memcpy(metadata + pairs->metadataOffsets[TEMPLATE_SS_ROWS], graph.rows,
          (size_t)(atoms + 1) * sizeof(int32_t));
  if (graph.count)
    memcpy(metadata + pairs->metadataOffsets[TEMPLATE_SS_NEIGHBORS], graph.neighbors,
            (size_t)graph.count * sizeof(int32_t));
  allocationTag = "template pairs: fixed topology metadata";
  pairs->metadata = mgpuMemAlloc((uint64_t)words * sizeof(int32_t), NULL, false);
  allocationTag = NULL;
  /* Synchronous only at initialization: the temporary host memory can be
   * released immediately, with no outstanding borrowed source. */
  check(cuMemcpyHtoD((CUdeviceptr)pairs->metadata, metadata,
                     (size_t)words * sizeof(int32_t)), "cuMemcpyHtoD");
  free(metadata);
  mdrtFreeTemplateExclusions(&graph);
  free(host);
  return handle;
}

void _mlir_ciface_mdrtTemplatePairsMetadata(struct Buffer1 *result, int64_t handle,
                                          int64_t slot) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  if (slot < 0 || slot >= TEMPLATE_METADATA_SLOTS) {
    fprintf(stderr, "mdrt: invalid template metadata slot\n");
    abort();
  }
  result->allocated = pairs->metadata;
  result->aligned = (void *)((uintptr_t)pairs->metadata +
      (uint64_t)pairs->metadataOffsets[slot] * sizeof(int32_t));
  result->offset = 0;
  result->size = pairs->metadataSizes[slot];
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
