/* Included by mdrt_cuda.c after the allocation and activation helpers.
 * These functions belong only to the new template route. */
#include "mdir/Runtime/TemplatePairsABI.h"
#include "mdir/Runtime/TemplateMembership.h"
#include "mdir/Runtime/TemplateExclusions.h"
#include <nvtx3/nvToolsExt.h>
#include <string.h>

struct TemplatePairs {
  struct MdrtTemplatePairLayout layout;
  void *pairs;
  void *metadata;
  void *order, *frames;
  void *groups, *planning;
  int64_t waterGroups, atomGroups, columnsX, columnsY;
  void *frameState;
  int64_t frameOffsets[TEMPLATE_FRAME_SLOTS], frameRows[TEMPLATE_FRAME_SLOTS];
  int64_t metadataOffsets[TEMPLATE_METADATA_SLOTS];
  int64_t metadataSizes[TEMPLATE_METADATA_SLOTS];
  int diagnosticBuilt;
  /* The refinement passes of the last group plan: the next plan runs as
   * many before it first reads its controls (D[rigid-water-template]). */
  int64_t planPasses;
};

/* Diagnostic counters belong to this route, independently of the general
 * neighbor structures. No force arithmetic or per-step device readback. */
static int64_t templateRefreshes, templatePrunes, templateGrowths;
static int64_t templateRowGrowths;
static int64_t templateStartEntries, templateEndEntries;
static int64_t templateStartWork, templateEndWork;
static int64_t templateStartCapacity, templateEndCapacity;
static int64_t templateStartInnerEntries, templateEndInnerEntries;

/* Mark force-evaluation boundaries without waiting for the device. Timeline
 * analysis correlates launches with these host intervals; it does not add
 * a synchronization to delimit a step. */
void mdrtTemplatePairsNoteStep(void) {
  nvtxMarkA("template force interval");
}

/* Temporary stage attribution for the implementation measurements. */
void mdrtTemplatePairsTraceStage(int64_t stage) {
  if (!stage) nvtxRangePop();
  else if (stage == 1) nvtxRangePushA("template spatial order");
  else if (stage == 2) nvtxRangePushA("template group plan");
}

void mdrtTemplatePairsSnapshotOccupancy(int64_t handle, int64_t first) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  if (!pairs->diagnosticBuilt) return;
  int32_t header[MDRT_TEMPLATE_PAIR_HEADER_WORDS];
  finish();
  check(cuMemcpyDtoH(header, (CUdeviceptr)pairs->pairs, sizeof(header)),
        "template occupancy header");
  int64_t entries = 0, work = 0;
  for (int tag = 0; tag != 3; ++tag) {
    const int32_t *range = header + TEMPLATE_PAIR_OUTER_RANGES + 6 * tag;
    entries += (int64_t)range[3] - range[2];
    work += (int64_t)range[5] - range[4];
  }
  int64_t innerEntries = 0;
  int32_t *inner = malloc((size_t)(work ? work : 1) * 4 * sizeof(int32_t));
  if (!inner) abort();
  if (work)
    check(cuMemcpyDtoH(inner,
        (CUdeviceptr)pairs->pairs + pairs->layout.innerUnits * sizeof(int32_t),
        (size_t)work * 4 * sizeof(int32_t)), "template inner occupancy");
  for (int64_t unit = 0; unit != work; ++unit)
    innerEntries += (int64_t)inner[4 * unit + 3] - inner[4 * unit + 2];
  free(inner);
  if (first) {
    templateStartEntries = entries;
    templateStartWork = work;
    templateStartCapacity = pairs->layout.entries;
    templateStartInnerEntries = innerEntries;
  } else {
    templateEndEntries = entries;
    templateEndWork = work;
    templateEndCapacity = pairs->layout.entries;
    templateEndInnerEntries = innerEntries;
  }
}

void mdrtTemplatePairsNoteRefresh(int64_t handle) {
  ((struct TemplatePairs *)(intptr_t)handle)->diagnosticBuilt = 1;
  if (++templateRefreshes == 1)
    mdrtTemplatePairsSnapshotOccupancy(handle, 1);
}

/* A build found the covering reach of the lists beyond half the least face
 * width of the current cell (the cell shrank under the barostat, or the
 * input widened the water's extent or the solute units' radius): a pair
 * could have two images within it. The selection checked the input cell
 * with the nominal values; the run has failed. Lengths inside MDIR are nm;
 * the message gives Å, as the selection's does. */
void mdrtTemplatePairsStopReach(double reach, double widest, double half) {
  char message[512];
  snprintf(message, sizeof message,
           "the template's covering reach, %.4g Å ('pairlist_distance' plus "
           "twice %.4g Å, the larger of the water's extent and the solute "
           "units' radius at this build), exceeds half the least face width "
           "of the current cell, %.4g Å; the run has failed (the cell shrank "
           "or the input widened the extent since the selection: use a "
           "larger cell or 'water.template = \"OFF\"')",
           10.0 * reach, 10.0 * widest, 10.0 * half);
  stopWith(message);
}

/* The solute units' radius that each build held (nm): the first build's,
 * which an unconstrained input may widen beyond the nominal radius, and the
 * largest and the count of the later ones, which constraints hold to it.
 * They belong to the run, whose programs (the terms at the start, then the
 * steps) share them: the first build is the start's. The trial runs of a
 * search of the skins (D281) are not the run's, and clear them when they
 * end, with the counts of mdrtResetCounts. */
static double templateRadiusNominal, templateRadiusFirst, templateRadiusLater;
static int64_t templateRadiusBuilds;
void mdrtTemplatePairsResetRadius(void) {
  templateRadiusNominal = templateRadiusFirst = templateRadiusLater = 0.0;
  templateRadiusBuilds = 0;
}
void mdrtTemplatePairsNoteRadius(double radius, double nominal) {
  templateRadiusNominal = nominal;
  if (templateRadiusBuilds++ == 0) templateRadiusFirst = radius;
  else if (templateRadiusBuilds == 2 || radius > templateRadiusLater)
    templateRadiusLater = radius;
}

/* Slot 0: the nominal radius, 1: the first build's, 2: the largest later
 * one (Å); 3: the count of later builds. Zero without solute units. */
double mdrtGetTemplateRadius(int64_t slot) {
  if (slot == 0) return 10.0 * templateRadiusNominal;
  if (slot == 1) return 10.0 * templateRadiusFirst;
  if (slot == 2) return 10.0 * templateRadiusLater;
  if (slot == 3) return templateRadiusBuilds ? (double)(templateRadiusBuilds - 1) : 0.0;
  return 0.0;
}

void mdrtTemplatePairsNotePrune(int64_t water, int64_t atom) {
  if (water || atom) ++templatePrunes;
}

/* Coverage of the evaluated lists, a test-only diagnostic: compiled with
 * MDIR_TEMPLATE_CHECK_COVERAGE set, every evaluation is followed by an
 * independent all-pairs, all-near-images check on the host that each site
 * pair within the cutoff of the packed frames it evaluated has an inner
 * entry (a superset of the general path's pairs). */
static int64_t templateCoverageChecks, templateCoverageMisses;
static int64_t templateCoveragePairs, templateCoverageDuplicates;
static int64_t templateCoverageRows, templateCoverageSplits;

int64_t mdrtGetTemplatePairStatistic(int64_t slot) {
  if (slot == 12) return templateCoverageChecks;
  if (slot == 13) return templateCoverageMisses;
  if (slot == 14) return templateCoveragePairs;
  if (slot == 15) return templateCoverageDuplicates;
  if (slot == 16) return templateCoverageRows;
  if (slot == 17) return templateCoverageSplits;
  const int64_t values[] = {templateRefreshes, templatePrunes, templateGrowths,
      templateStartEntries, templateEndEntries, templateStartWork,
      templateEndWork, templateStartCapacity, templateEndCapacity};
  if (slot == 9) return templateStartInnerEntries;
  if (slot == 10) return templateEndInnerEntries;
  if (slot == 11) return templateRowGrowths;
  return slot >= 0 && slot < 9 ? values[slot] : 0;
}

static void allocateTemplatePairs(struct TemplatePairs *pairs) {
  allocationTag = "template pairs: shared outer/inner WW/SW/SS";
  pairs->pairs = mgpuMemAlloc((uint64_t)pairs->layout.words * sizeof(int32_t),
                             NULL, false);
  allocationTag = NULL;
  /* Defined empty ranges also make a later row-capacity migration safe
   * before the first completed list. The physical fields are published by
   * the build; no uninitialized payload is read to discover its ranges. */
  int32_t header[MDRT_TEMPLATE_PAIR_HEADER_WORDS] = {0};
  header[TEMPLATE_PAIR_WATERS] = (int32_t)pairs->layout.waters;
  header[TEMPLATE_PAIR_ATOMS] = (int32_t)pairs->layout.atoms;
  header[TEMPLATE_PAIR_ROWS] = (int32_t)(pairs->waterGroups + 2 * pairs->atomGroups);
  header[TEMPLATE_PAIR_CAPACITY] = (int32_t)pairs->layout.entries;
  enter();
  /* A pageable copy may return before its DMA lands, and the shared
   * stream is non-blocking: wait here so no kernel reads the old words. */
  check(cuMemcpyHtoD((CUdeviceptr)pairs->pairs, header, sizeof(header)),
        "template empty header");
  check(cuCtxSynchronize(), "template empty header");
}

static int64_t createTemplatePairs(int64_t waters, int64_t atoms, int64_t entries) {
  struct TemplatePairs *pairs = calloc(1, sizeof(*pairs));
  if (!pairs || !mdrtTemplatePairLayout(waters, atoms, entries, &pairs->layout)) {
    fprintf(stderr, "mdrt: invalid template-pair capacity or host allocation failure\n");
    abort();
  }
  pairs->waterGroups = (waters + 15) / 16;
  pairs->atomGroups = (atoms + 15) / 16;
  allocateTemplatePairs(pairs);
  int64_t units = waters + atoms, plan = units ? units : 1;
  int64_t planBlocks = (units + 1023) / 1024;
  if (!planBlocks) planBlocks = 1;
  allocationTag = "template pairs: compact unit order";
  pairs->order = mgpuMemAlloc((uint64_t)(units ? units : 1) * sizeof(int32_t),
                              NULL, false);
  allocationTag = "template pairs: group descriptors and lengths";
  pairs->groups = mgpuMemAlloc((uint64_t)(units ? 4 * units : 1) * sizeof(int32_t),
                               NULL, false);
  allocationTag = "template pairs: shared sort and group-plan scratch";
  pairs->planning = mgpuMemAlloc((uint64_t)(6 * plan + 10 + planBlocks) * sizeof(int32_t),
                                 NULL, false);
  allocationTag = "template pairs: packed real-site frames";
  int64_t rows = 3 * waters + atoms;
  pairs->frames = mgpuMemAlloc((uint64_t)(rows ? rows : 1) * 3 * sizeof(float),
                               NULL, false);
  int64_t frameWords = 0;
  for (int slot = 0; slot != TEMPLATE_FRAME_SLOTS; ++slot) {
    pairs->frameOffsets[slot] = frameWords;
    pairs->frameRows[slot] = slot < TEMPLATE_FRAME_OUTER_CELLS ? units
        : slot == TEMPLATE_FRAME_REACHES ? 2
        : (slot == TEMPLATE_FRAME_GROUP_BOXES || slot == TEMPLATE_FRAME_FRACTIONAL_BOXES)
            ? 2 * units : 6;
    frameWords += 3 * pairs->frameRows[slot];
  }
  allocationTag = "template pairs: reference epochs and retained packing cells";
  pairs->frameState = mgpuMemAlloc((uint64_t)frameWords * sizeof(double), NULL, false);
  allocationTag = NULL;
  recordCallHandle(pairs);
  return (int64_t)(intptr_t)pairs;
}

/* The compiler passes host members, already renumbered like the state.
 * Upload once at initialization, never at a step's refresh. */
int64_t _mlir_ciface_mdrtTemplatePairsCreateMembers(
    struct Buffer2 *members, int64_t particles, int64_t entries,
    struct Buffer2 *excluded, struct Buffer2 *anchored) {
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
  /* Solute units: (member, anchor) pairs name each member that its
   * constrained bond keeps near a heavy atom. Every other atom anchors
   * itself. Units are fixed IDs [W, W+A). The compiler proves the radius
   * from the constraints; here only the shape of the partition is checked. */
  int64_t anchorCount = anchored->sizes[0];
  int32_t *anchors = malloc((size_t)(atoms ? atoms : 1) * sizeof(int32_t));
  int32_t *dependents = malloc((size_t)(atoms ? atoms : 1) *
      MDRT_TEMPLATE_ANCHOR_DEPENDENTS * sizeof(int32_t));
  if (!anchors || !dependents || anchorCount < 0 ||
      (anchorCount && anchored->sizes[1] != 2)) {
    fprintf(stderr, "mdrt: invalid template anchor table\n");
    abort();
  }
  for (int64_t a = 0; a < atoms; ++a) {
    anchors[a] = (int32_t)(waters + a);
    for (int d = 0; d != MDRT_TEMPLATE_ANCHOR_DEPENDENTS; ++d)
      dependents[a * MDRT_TEMPLATE_ANCHOR_DEPENDENTS + d] = -1;
  }
  {
    int32_t *slotOf = malloc((size_t)(particles ? particles : 1) * sizeof(int32_t));
    if (!slotOf) abort();
    for (int64_t p = 0; p < particles; ++p) slotOf[p] = -1;
    for (int64_t row = 0; row < count; ++row) slotOf[host[row]] = (int32_t)row;
    const int32_t *pairs = anchorCount ? (const int32_t *)anchored->aligned + anchored->offset : NULL;
    for (int64_t k = 0; k < anchorCount; ++k) {
      int32_t member = pairs[k * anchored->strides[0]];
      int32_t anchor = pairs[k * anchored->strides[0] + anchored->strides[1]];
      int ok = member >= 0 && member < particles && anchor >= 0 && anchor < particles &&
               member != anchor && slotOf[member] >= 3 * waters && slotOf[anchor] >= 3 * waters;
      int64_t m = ok ? slotOf[member] - 3 * waters : 0, x = ok ? slotOf[anchor] - 3 * waters : 0;
      ok = ok && anchors[m] == (int32_t)(waters + m);
      int d = 0;
      while (ok && d != MDRT_TEMPLATE_ANCHOR_DEPENDENTS &&
             dependents[x * MDRT_TEMPLATE_ANCHOR_DEPENDENTS + d] >= 0) ++d;
      if (!ok || d == MDRT_TEMPLATE_ANCHOR_DEPENDENTS) {
        fprintf(stderr, "mdrt: invalid or overlapping template solute units\n");
        abort();
      }
      anchors[m] = (int32_t)(waters + x);
      dependents[x * MDRT_TEMPLATE_ANCHOR_DEPENDENTS + d] = (int32_t)(waters + m);
    }
    /* No chains: an anchor is anchored to itself. */
    for (int64_t a = 0; a < atoms; ++a)
      if (anchors[anchors[a] - waters] != anchors[a]) {
        fprintf(stderr, "mdrt: a template solute anchor is itself anchored\n");
        abort();
      }
    free(slotOf);
  }
  int64_t handle = createTemplatePairs(waters, atoms, entries);
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  int64_t sizes[TEMPLATE_METADATA_SLOTS] = {count, particles, atoms + 1, graph.count,
      atoms, MDRT_TEMPLATE_ANCHOR_DEPENDENTS * atoms};
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
  if (atoms) {
    memcpy(metadata + pairs->metadataOffsets[TEMPLATE_ATOM_ANCHORS], anchors,
           (size_t)atoms * sizeof(int32_t));
    memcpy(metadata + pairs->metadataOffsets[TEMPLATE_ATOM_DEPENDENTS], dependents,
           (size_t)atoms * MDRT_TEMPLATE_ANCHOR_DEPENDENTS * sizeof(int32_t));
  }
  free(anchors);
  free(dependents);
  allocationTag = "template pairs: fixed topology metadata";
  pairs->metadata = mgpuMemAlloc((uint64_t)words * sizeof(int32_t), NULL, false);
  allocationTag = NULL;
  /* Synchronous only at initialization: the temporary host memory can be
   * released immediately, with no outstanding borrowed source. From
   * pageable memory the copy may return before its DMA lands, and the
   * shared stream is non-blocking, so the context is synchronized before
   * any kernel or copy on that stream reads the metadata. */
  check(cuMemcpyHtoD((CUdeviceptr)pairs->metadata, metadata,
                     (size_t)words * sizeof(int32_t)), "cuMemcpyHtoD");
  check(cuCtxSynchronize(), "template metadata upload");
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

/* These fixed-size mutable views belong to the activation, like metadata;
 * growing the pair arena never relocates them. A build writes the order,
 * and packing at the position writer writes the frames. */
void _mlir_ciface_mdrtTemplatePairsOrder(struct Buffer1 *result, int64_t handle) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  result->allocated = result->aligned = pairs->order;
  result->offset = 0;
  result->size = pairs->layout.waters + pairs->layout.atoms;
  result->stride = 1;
}

void _mlir_ciface_mdrtTemplatePairsFrames(struct Buffer2 *result, int64_t handle) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  result->allocated = result->aligned = pairs->frames;
  result->offset = 0;
  result->sizes[0] = 3 * pairs->layout.waters + pairs->layout.atoms;
  result->sizes[1] = 3;
  result->strides[0] = 3;
  result->strides[1] = 1;
}

void _mlir_ciface_mdrtTemplatePairsGroupView(struct Buffer1 *result,
                                           int64_t handle, int64_t slot) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  int64_t units = pairs->layout.waters + pairs->layout.atoms;
  if (slot < 0 || slot >= TEMPLATE_GROUP_VIEWS) {
    fprintf(stderr, "mdrt: invalid template group view\n");
    abort();
  }
  result->allocated = pairs->groups;
  result->aligned = (void *)((uintptr_t)pairs->groups +
      (uint64_t)(slot * units) * sizeof(int32_t));
  result->offset = 0;
  result->size = units;
  result->stride = 1;
}

void _mlir_ciface_mdrtTemplatePairsPlanningView(struct Buffer1 *result,
                                              int64_t handle, int64_t slot) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  int64_t units = pairs->layout.waters + pairs->layout.atoms;
  int64_t m = units ? units : 1, blocks = (units + 1023) / 1024;
  if (!blocks) blocks = 1;
  int64_t offset = 0, size = 0;
  switch (slot) {
  case TEMPLATE_PLAN_FLAGS: size = m; break;
  case TEMPLATE_PLAN_SEGMENTS: offset = m; size = m + 1; break;
  case TEMPLATE_PLAN_STARTS: offset = 2 * m + 1; size = m; break;
  case TEMPLATE_PLAN_LENGTHS: offset = 3 * m + 1; size = m; break;
  case TEMPLATE_PLAN_COUNTS: offset = 4 * m + 1; size = m; break;
  case TEMPLATE_PLAN_OFFSETS: offset = 5 * m + 1; size = m + 1; break;
  case TEMPLATE_PLAN_CONTROL: offset = 6 * m + 2; size = 8; break;
  case TEMPLATE_PLAN_SCAN: offset = 6 * m + 10; size = blocks; break;
  case TEMPLATE_PLAN_SORT: size = 4 * m; break;
  default:
    fprintf(stderr, "mdrt: invalid template planning view\n");
    abort();
  }
  result->allocated = pairs->planning;
  result->aligned = (void *)((uintptr_t)pairs->planning + (uint64_t)offset * sizeof(int32_t));
  result->offset = 0;
  result->size = size;
  result->stride = 1;
}

/* A retained class must keep its column interpretation. Replan the common
 * grid only at a joint epoch (empty classes impose no restriction). */
int64_t mdrtTemplatePairsChooseGrid(int64_t handle, int64_t x, int64_t y,
                                   int64_t water, int64_t atom) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  int64_t budget = pairs->layout.spatialBins / 2;
  if (x <= 0 || y <= 0 || x > budget || y > budget / x) {
    fprintf(stderr, "mdrt: invalid template group grid\n");
    abort();
  }
  if (!pairs->columnsX ||
      ((!pairs->layout.waters || water) && (!pairs->layout.atoms || atom))) {
    pairs->columnsX = x;
    pairs->columnsY = y;
  }
  return pairs->columnsX;
}

/* Reference epochs and packing-cell coordinates do not move on pair
 * capacity growth. Only a requested mobility-group rebuild updates them. */
void _mlir_ciface_mdrtTemplatePairsFrameState(struct Buffer2 *result,
                                             int64_t handle, int64_t slot) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  if (slot < 0 || slot >= TEMPLATE_FRAME_SLOTS) {
    fprintf(stderr, "mdrt: invalid template frame-state slot\n");
    abort();
  }
  result->allocated = pairs->frameState;
  result->aligned = (void *)((uintptr_t)pairs->frameState +
      (uint64_t)pairs->frameOffsets[slot] * sizeof(double));
  result->offset = 0;
  result->sizes[0] = pairs->frameRows[slot];
  result->sizes[1] = 3;
  result->strides[0] = 3;
  result->strides[1] = 1;
}

/* Search-only Cartesian boxes reuse the first half of the existing f64
 * auxiliary slot. Reference epochs and the generic f64 frame ABI stay intact. */
void _mlir_ciface_mdrtTemplatePairsGroupBoxesF32(struct Buffer2 *result,
                                                int64_t handle) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  result->allocated = pairs->frameState;
  result->aligned = (void *)((uintptr_t)pairs->frameState +
      (uint64_t)pairs->frameOffsets[TEMPLATE_FRAME_GROUP_BOXES] * sizeof(double));
  result->offset = 0;
  result->sizes[0] = pairs->frameRows[TEMPLATE_FRAME_GROUP_BOXES];
  result->sizes[1] = 3;
  result->strides[0] = 3;
  result->strides[1] = 1;
}

/* Host allocation constants only: no device readback or synchronization. */
int64_t mdrtTemplatePairsLayoutValue(int64_t handle, int64_t slot) {
  struct MdrtTemplatePairLayout *layout =
      &((struct TemplatePairs *)(intptr_t)handle)->layout;
  switch (slot) {
  case TEMPLATE_LAYOUT_WATERS: return layout->waters;
  case TEMPLATE_LAYOUT_ATOMS: return layout->atoms;
  case TEMPLATE_LAYOUT_ROWS:
    return ((struct TemplatePairs *)(intptr_t)handle)->waterGroups +
        2 * ((struct TemplatePairs *)(intptr_t)handle)->atomGroups;
  case TEMPLATE_LAYOUT_ENTRIES: return layout->entries;
  case TEMPLATE_LAYOUT_UNITS: return layout->units;
  case TEMPLATE_LAYOUT_OUTER_ROWS: return layout->outerRows;
  case TEMPLATE_LAYOUT_INNER_ROWS: return layout->innerRows;
  case TEMPLATE_LAYOUT_OUTER_UNITS: return layout->outerUnits;
  case TEMPLATE_LAYOUT_INNER_UNITS: return layout->innerUnits;
  case TEMPLATE_LAYOUT_OUTER_ENTRIES: return layout->outerEntries;
  case TEMPLATE_LAYOUT_INNER_ENTRIES: return layout->innerEntries;
  case TEMPLATE_LAYOUT_STAGE_UNITS: return layout->stageUnits;
  case TEMPLATE_LAYOUT_STAGE_ENTRIES: return layout->stageEntries;
  case TEMPLATE_LAYOUT_BUILD_COUNTS: return layout->buildCounts;
  case TEMPLATE_LAYOUT_BUILD_ENTRY_OFFSETS: return layout->buildEntryOffsets;
  case TEMPLATE_LAYOUT_BUILD_WORK_COUNTS: return layout->buildWorkCounts;
  case TEMPLATE_LAYOUT_BUILD_WORK_OFFSETS: return layout->buildWorkOffsets;
  case TEMPLATE_LAYOUT_BUILD_SCAN_SCRATCH: return layout->buildScanScratch;
  case TEMPLATE_LAYOUT_SPATIAL_KEYS: return layout->spatialKeys;
  case TEMPLATE_LAYOUT_SPATIAL_Z_KEYS: return layout->spatialZKeys;
  case TEMPLATE_LAYOUT_SPATIAL_ITEMS: return layout->spatialItems;
  case TEMPLATE_LAYOUT_SPATIAL_PLACES: return layout->spatialPlaces;
  case TEMPLATE_LAYOUT_SPATIAL_COUNTS: return layout->spatialCounts;
  case TEMPLATE_LAYOUT_SPATIAL_STARTS: return layout->spatialStarts;
  case TEMPLATE_LAYOUT_SPATIAL_CURSORS: return layout->spatialCursors;
  case TEMPLATE_LAYOUT_SPATIAL_BINS: return layout->spatialBins;
  case TEMPLATE_LAYOUT_SCAN_BLOCKS: return layout->scanBlocks;
  case TEMPLATE_LAYOUT_WATER_GROUPS: return ((struct TemplatePairs *)(intptr_t)handle)->waterGroups;
  case TEMPLATE_LAYOUT_ATOM_GROUPS: return ((struct TemplatePairs *)(intptr_t)handle)->atomGroups;
  case TEMPLATE_LAYOUT_COLUMNS_X: return ((struct TemplatePairs *)(intptr_t)handle)->columnsX;
  case TEMPLATE_LAYOUT_COLUMNS_Y: return ((struct TemplatePairs *)(intptr_t)handle)->columnsY;
  case TEMPLATE_LAYOUT_ROW_CAPACITY: return layout->rows;
  default:
    fprintf(stderr, "mdrt: invalid template layout slot\n");
    abort();
  }
}

/* Zero-offset views let shared scan emitters use their ordinary identity
 * memref descriptors. The aligned pointer includes the arena's physical
 * offset; all such offsets are computed in i64. Reacquire after growth. */
void _mlir_ciface_mdrtTemplatePairsArenaView(struct Buffer1 *result,
                                           int64_t handle, int64_t slot) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  struct MdrtTemplatePairLayout *layout = &pairs->layout;
  int64_t offset = mdrtTemplatePairsLayoutValue(handle, slot), size = 0;
  switch (slot) {
  case TEMPLATE_LAYOUT_OUTER_ROWS:
  case TEMPLATE_LAYOUT_INNER_ROWS: size = 4 * layout->rows; break;
  case TEMPLATE_LAYOUT_OUTER_UNITS:
  case TEMPLATE_LAYOUT_INNER_UNITS:
  case TEMPLATE_LAYOUT_STAGE_UNITS: size = 4 * layout->units; break;
  case TEMPLATE_LAYOUT_OUTER_ENTRIES:
  case TEMPLATE_LAYOUT_INNER_ENTRIES: size = 4 * layout->entries; break;
  case TEMPLATE_LAYOUT_STAGE_ENTRIES: size = 4 * 64 * layout->units; break;
  case TEMPLATE_LAYOUT_BUILD_COUNTS:
  case TEMPLATE_LAYOUT_BUILD_WORK_COUNTS: size = layout->rows; break;
  case TEMPLATE_LAYOUT_BUILD_ENTRY_OFFSETS:
  case TEMPLATE_LAYOUT_BUILD_WORK_OFFSETS: size = layout->rows + 1; break;
  case TEMPLATE_LAYOUT_BUILD_SCAN_SCRATCH: size = layout->scanBlocks; break;
  case TEMPLATE_LAYOUT_SPATIAL_KEYS:
  case TEMPLATE_LAYOUT_SPATIAL_Z_KEYS:
  case TEMPLATE_LAYOUT_SPATIAL_ITEMS:
  case TEMPLATE_LAYOUT_SPATIAL_PLACES: size = layout->waters + layout->atoms; break;
  case TEMPLATE_LAYOUT_SPATIAL_COUNTS:
  case TEMPLATE_LAYOUT_SPATIAL_CURSORS: size = layout->spatialBins; break;
  case TEMPLATE_LAYOUT_SPATIAL_STARTS: size = layout->spatialBins + 1; break;
  default:
    fprintf(stderr, "mdrt: template layout slot is not an arena view\n");
    abort();
  }
  result->allocated = pairs->pairs;
  result->aligned = (void *)((uintptr_t)pairs->pairs +
                             (uint64_t)offset * sizeof(int32_t));
  result->offset = 0;
  result->size = size;
  result->stride = 1;
}

void _mlir_ciface_mdrtTemplatePairsCounters(struct Buffer1 *result, int64_t handle) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  result->allocated = pairs->pairs;
  result->aligned = (void *)((uintptr_t)pairs->pairs +
      TEMPLATE_PAIR_STAGE_COUNT * sizeof(int32_t));
  result->offset = 0;
  result->size = 4;
  result->stride = 1;
}

int64_t mdrtTemplatePairsCapacity(int64_t handle) {
  return ((struct TemplatePairs *)(intptr_t)handle)->layout.entries;
}

static void copyTemplateWords(void *to, int64_t destination,
                               void *from, int64_t source, int64_t words) {
  if (words)
    check(cuMemcpyDtoD((CUdeviceptr)to + (uint64_t)destination * sizeof(int32_t),
        (CUdeviceptr)from + (uint64_t)source * sizeof(int32_t),
        (uint64_t)words * sizeof(int32_t)), "template row-growth copy");
}

/* Row capacity grows monotonically. Preserve completed class ranges and
 * this plan's spatial index, so changing one class's group count cannot
 * discard the inactive class. Scratch and unfinished staging are dead. */
static void growTemplateRows(struct TemplatePairs *pairs, int64_t required) {
  if (required <= pairs->layout.rows) return;
  struct MdrtTemplatePairLayout old = pairs->layout, next;
  int64_t maximum = old.waters + 2 * old.atoms;
  int64_t capacity = required + required / 4;
  if (capacity > maximum) capacity = maximum;
  if (!mdrtTemplatePairLayoutCapacity(old.waters, old.atoms, old.entries, capacity, &next)) {
    fprintf(stderr, "mdrt: invalid template row capacity\n");
    abort();
  }
  enter();
  check(cuCtxSynchronize(), "template row-growth readiness");
  uint32_t header[MDRT_TEMPLATE_PAIR_HEADER_WORDS];
  check(cuMemcpyDtoH(header, (CUdeviceptr)pairs->pairs, sizeof(header)),
        "template row-growth header");
  int64_t rows = 0, entries = 0, work = 0;
  for (int partition = 0; partition != 2; ++partition)
    for (int tag = 0; tag != 3; ++tag) {
      const uint32_t *range = header + (partition ? TEMPLATE_PAIR_INNER_RANGES
                                                : TEMPLATE_PAIR_OUTER_RANGES) + 6 * tag;
      if (range[1] > rows) rows = range[1];
      if (range[3] > entries) entries = range[3];
      if (range[5] > work) work = range[5];
    }
  if (rows > old.rows || entries > old.entries || work > old.units) {
    fprintf(stderr, "mdrt: invalid template row-growth ranges\n");
    abort();
  }
  void *previous = pairs->pairs;
  pairs->layout = next;
  allocateTemplatePairs(pairs);
  copyTemplateWords(pairs->pairs, 0, previous, 0, MDRT_TEMPLATE_PAIR_HEADER_WORDS);
  copyTemplateWords(pairs->pairs, next.outerRows, previous, old.outerRows, 4 * rows);
  copyTemplateWords(pairs->pairs, next.innerRows, previous, old.innerRows, 4 * rows);
  copyTemplateWords(pairs->pairs, next.outerUnits, previous, old.outerUnits, 4 * work);
  copyTemplateWords(pairs->pairs, next.innerUnits, previous, old.innerUnits, 4 * work);
  copyTemplateWords(pairs->pairs, next.outerEntries, previous, old.outerEntries, 4 * entries);
  copyTemplateWords(pairs->pairs, next.innerEntries, previous, old.innerEntries, 4 * entries);
  int64_t units = old.waters + old.atoms;
  copyTemplateWords(pairs->pairs, next.spatialKeys, previous, old.spatialKeys, units);
  copyTemplateWords(pairs->pairs, next.spatialZKeys, previous, old.spatialZKeys, units);
  copyTemplateWords(pairs->pairs, next.spatialItems, previous, old.spatialItems, units);
  copyTemplateWords(pairs->pairs, next.spatialPlaces, previous, old.spatialPlaces, units);
  int64_t bins = 2 * pairs->columnsX * pairs->columnsY;
  copyTemplateWords(pairs->pairs, next.spatialCounts, previous, old.spatialCounts, bins);
  copyTemplateWords(pairs->pairs, next.spatialStarts, previous, old.spatialStarts, bins + 1);
  copyTemplateWords(pairs->pairs, next.spatialCursors, previous, old.spatialCursors, bins);
  check(cuCtxSynchronize(), "template row-growth completion");
  releaseReplaced(previous);
  ++templateRowGrowths;
}

/* The passes that the group plan of the last build needed (one before the
 * first build): the next plan runs that many before its first readback.
 * Host state of this list only, no device access. */
int64_t mdrtTemplatePairsPlanPasses(int64_t handle) {
  int64_t passes = ((struct TemplatePairs *)(intptr_t)handle)->planPasses;
  return passes > 0 ? passes : 1;
}
void mdrtTemplatePairsNotePlanPasses(int64_t handle, int64_t passes) {
  ((struct TemplatePairs *)(intptr_t)handle)->planPasses = passes;
}

int64_t mdrtTemplatePairsCommitGroups(int64_t handle, int64_t waterGroups,
                                      int64_t atomGroups, int64_t water, int64_t atom) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  if (water) pairs->waterGroups = waterGroups;
  if (atom) pairs->atomGroups = atomGroups;
  if (pairs->waterGroups < (pairs->layout.waters + 15) / 16 ||
      pairs->waterGroups > pairs->layout.waters ||
      pairs->atomGroups < (pairs->layout.atoms + 15) / 16 ||
      pairs->atomGroups > pairs->layout.atoms) {
    fprintf(stderr, "mdrt: invalid template group counts\n");
    abort();
  }
  int64_t rows = pairs->waterGroups + 2 * pairs->atomGroups;
  growTemplateRows(pairs, rows);
  return rows;
}

/* Growth discards every class range. A caller that grows must rebuild
 * all three classes, even if only one mobility group triggered the build.
 * No wait, allocation or host readback occurs on a refresh that fits. */
int64_t mdrtTemplatePairsGrow(int64_t handle, int64_t required) {
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  if (required >= 0 && required <= pairs->layout.entries) return 0;
  ++templateGrowths;
  int64_t capacity = required;
  if (required >= 0 && required <= INT32_MAX)
    capacity = required + (INT32_MAX - required < required / 4
                               ? INT32_MAX - required : required / 4);
  struct MdrtTemplatePairLayout next;
  if (!mdrtTemplatePairLayoutCapacity(pairs->layout.waters, pairs->layout.atoms,
                                      capacity, pairs->layout.rows, &next)) {
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

/* Covered (first unit, partner unit, image) keys, sorted for search. */
static uint64_t templateCoverageKey(int64_t first, int64_t partner, int code) {
  return ((uint64_t)first << 47) | ((uint64_t)partner << 30) | (uint64_t)code;
}
static int templateImageCode(int n0, int n1, int n2) {
  return ((n0 + MDRT_TEMPLATE_IMAGE_OFFSET) & MDRT_TEMPLATE_IMAGE_MASK) |
         (((n1 + MDRT_TEMPLATE_IMAGE_OFFSET) & MDRT_TEMPLATE_IMAGE_MASK) << MDRT_TEMPLATE_IMAGE_BITS) |
         (((n2 + MDRT_TEMPLATE_IMAGE_OFFSET) & MDRT_TEMPLATE_IMAGE_MASK) << (2 * MDRT_TEMPLATE_IMAGE_BITS));
}
static int templateHasKey(const uint64_t *keys, int64_t count, uint64_t key) {
  int64_t lo = 0, hi = count;
  while (lo < hi) {
    int64_t mid = lo + (hi - lo) / 2;
    if (keys[mid] < key) lo = mid + 1; else hi = mid;
  }
  return lo < count && keys[lo] == key;
}

/* The cell is (ax, by, cz, bx, cx, cy), the lattice vectors as columns of
 * a lower-triangular H; `cutoff` and `alpha` are the template's. */
void mdrtTemplatePairsCheckCoverage(int64_t handle, double cutoff, double alpha,
    int64_t hasSite, double ax, double by, double cz, double bx, double cx,
    double cy) {
  /* A value above one scales the checked cutoff: beyond the lists' reach
   * the check must find pairs without an entry (its own negative test). */
  /* "duplicate" counts the first inner entry twice: the check of
   * duplicate entries must then fail (its own negative test). */
  static double scale = -1;
  static int injectDuplicate;
  if (scale < 0) {
    const char *flag = getenv("MDIR_TEMPLATE_CHECK_COVERAGE");
    scale = flag && strtod(flag, NULL) > 1 ? strtod(flag, NULL) : 1;
    injectDuplicate = flag && strcmp(flag, "duplicate") == 0;
  }
  cutoff *= scale;
  struct TemplatePairs *pairs = (struct TemplatePairs *)(intptr_t)handle;
  struct MdrtTemplatePairLayout *L = &pairs->layout;
  int64_t W = L->waters, A = L->atoms, units = W + A, rows = 3 * W + A;
  if (units >= ((int64_t)1 << 17)) {
    fprintf(stderr, "mdrt: the template coverage check takes fewer than 2^17 units\n");
    abort();
  }
  enter();
  check(cuCtxSynchronize(), "template coverage readiness");
  int32_t header[MDRT_TEMPLATE_PAIR_HEADER_WORDS];
  check(cuMemcpyDtoH(header, (CUdeviceptr)pairs->pairs, sizeof(header)), "coverage header");
  int64_t work = 0;
  for (int tag = 0; tag != 3; ++tag) {
    int64_t end = header[TEMPLATE_PAIR_INNER_RANGES + 6 * tag + 5];
    if (end > work) work = end;
  }
  int32_t *records = malloc((size_t)(work ? work : 1) * 4 * sizeof(int32_t));
  int32_t *order = malloc((size_t)(units ? units : 1) * sizeof(int32_t));
  int32_t *lengths = malloc((size_t)(units ? units : 1) * sizeof(int32_t));
  float *frames = malloc((size_t)(rows ? rows : 1) * 3 * sizeof(float));
  int32_t *exRows = malloc((size_t)(A + 1) * sizeof(int32_t));
  int64_t exCount = pairs->metadataSizes[TEMPLATE_SS_NEIGHBORS];
  int32_t *exNeighbors = malloc((size_t)(exCount ? exCount : 1) * sizeof(int32_t));
  double *references = malloc((size_t)(units ? units : 1) * 3 * sizeof(double));
  if (!records || !order || !lengths || !frames || !exRows || !exNeighbors || !references)
    abort();
  if (work)
    check(cuMemcpyDtoH(records, (CUdeviceptr)pairs->pairs + (uint64_t)L->innerUnits * 4,
                       (size_t)work * 4 * sizeof(int32_t)), "coverage work");
  int64_t entries = 0;
  for (int64_t r = 0; r != work; ++r)
    if (records[4 * r + 3] > entries) entries = records[4 * r + 3];
  int32_t *inner = malloc((size_t)(entries ? entries : 1) * 4 * sizeof(int32_t));
  if (!inner) abort();
  if (entries)
    check(cuMemcpyDtoH(inner, (CUdeviceptr)pairs->pairs + (uint64_t)L->innerEntries * 4,
                       (size_t)entries * 4 * sizeof(int32_t)), "coverage entries");
  if (units) {
    check(cuMemcpyDtoH(order, (CUdeviceptr)pairs->order, (size_t)units * sizeof(int32_t)), "coverage order");
    check(cuMemcpyDtoH(lengths, (CUdeviceptr)pairs->groups +
        (uint64_t)(TEMPLATE_GROUP_AT_START * units) * sizeof(int32_t),
        (size_t)units * sizeof(int32_t)), "coverage lengths");
  }
  if (rows)
    check(cuMemcpyDtoH(frames, (CUdeviceptr)pairs->frames, (size_t)rows * 3 * sizeof(float)), "coverage frames");
  /* The outer build's reference points, by unit: the search splits a row
   * group into the subsets of its members' lattice frames, the floors of
   * these points (U2). The inner list keeps the outer list's rows. */
  if (units)
    check(cuMemcpyDtoH(references, (CUdeviceptr)pairs->frameState +
        (uint64_t)pairs->frameOffsets[TEMPLATE_FRAME_OUTER_REFERENCE] * sizeof(double),
        (size_t)units * 3 * sizeof(double)), "coverage references");
  check(cuMemcpyDtoH(exRows, (CUdeviceptr)pairs->metadata +
      (uint64_t)pairs->metadataOffsets[TEMPLATE_SS_ROWS] * 4, (size_t)(A + 1) * 4), "coverage exclusions");
  if (exCount)
    check(cuMemcpyDtoH(exNeighbors, (CUdeviceptr)pairs->metadata +
        (uint64_t)pairs->metadataOffsets[TEMPLATE_SS_NEIGHBORS] * 4, (size_t)exCount * 4), "coverage exclusions");
  /* The row groups of the evaluated (inner) records, each class apart,
   * and whether the members of each lie in more than one lattice frame:
   * the search then queried the row once per frame subset and wrote an
   * entry per subset, partner and image, an excess over the exact pairs.
   * A record is a stretch of one row's entries; a row may span several. */
  int64_t keyCount = 0, keyRoom = 16 * entries + 1, rowCount = 0, recordCount = 0;
  for (int tag = 0; tag != 3; ++tag) {
    const int32_t *range = header + TEMPLATE_PAIR_INNER_RANGES + 6 * tag;
    if (range[5] > range[4]) recordCount += range[5] - range[4];
  }
  uint64_t *keys = malloc((size_t)keyRoom * sizeof(uint64_t));
  uint64_t *rowKeys = malloc((size_t)(recordCount + 1) * sizeof(uint64_t));
  if (!keys || !rowKeys) abort();
  for (int tag = 0; tag != 3; ++tag) {
    const int32_t *range = header + TEMPLATE_PAIR_INNER_RANGES + 6 * tag;
    for (int64_t r = range[4]; r < range[5]; ++r) {
      const int32_t *record = records + 4 * r;
      int64_t first = record[1], length = lengths[first];
      int members = 0, mixed = 0;
      double lattice[3] = {0, 0, 0};
      for (int64_t lane = 0; lane < length && lane < 16; ++lane) {
        int64_t unit = order[first + lane];
        if (unit < 0) continue;
        const double *point = references + 3 * unit;
        for (int axis = 0; axis != 3; ++axis) {
          /* The floor of a finite point of a few cells, without libm. */
          double whole = (double)(int64_t)point[axis];
          if (whole > point[axis]) whole -= 1;
          if (!members) lattice[axis] = whole;
          else mixed |= whole != lattice[axis];
        }
        ++members;
      }
      rowKeys[rowCount++] = ((uint64_t)tag << 40) | ((uint64_t)first << 1) | (uint64_t)mixed;
      for (int64_t e = record[2]; e < record[3]; ++e) {
        const int32_t *entry = inner + 4 * e;
        for (int64_t lane = 0; lane < length && lane < 16; ++lane)
          if ((entry[2] & (1 << lane)) && order[first + lane] >= 0)
            keys[keyCount++] = templateCoverageKey(order[first + lane], entry[0], entry[1]);
      }
    }
  }
  if (injectDuplicate && keyCount) {
    keys[keyCount] = keys[0];
    ++keyCount;
  }
  qsort(keys, (size_t)keyCount, sizeof(uint64_t), mdrtCompareTemplatePairKeys);
  qsort(rowKeys, (size_t)rowCount, sizeof(uint64_t), mdrtCompareTemplatePairKeys);
  int64_t rowGroups = 0, splits = 0;
  for (int64_t k = 0; k < rowCount; ++k)
    if (!k || (rowKeys[k] >> 1) != (rowKeys[k - 1] >> 1)) {
      ++rowGroups;
      splits += (int64_t)(rowKeys[k] & 1);
    }
  templateCoverageRows += rowGroups;
  templateCoverageSplits += splits;
  free(rowKeys);
  /* Each site pair must be evaluated once: no (unit, partner, image) twice,
   * and no pair both as (u, v, n) and as (v, u, -n). */
  int64_t duplicates = 0;
  for (int64_t k = 0; k != keyCount; ++k) {
    if (k && keys[k] == keys[k - 1]) {
      if (duplicates < 8)
        fprintf(stderr, "mdrt: template coverage: entry %llx appears twice\n",
                (unsigned long long)keys[k]);
      ++duplicates;
      continue;
    }
    int64_t first = (int64_t)(keys[k] >> 47), partner = (int64_t)((keys[k] >> 30) & 0x1ffff);
    int code = (int)(keys[k] & 0x3fffffff), n[3];
    for (int axis = 0; axis != 3; ++axis)
      n[axis] = ((code >> (axis * MDRT_TEMPLATE_IMAGE_BITS)) & MDRT_TEMPLATE_IMAGE_MASK) -
                MDRT_TEMPLATE_IMAGE_OFFSET;
    int reverse = templateImageCode(-n[0], -n[1], -n[2]);
    uint64_t mirror = templateCoverageKey(partner, first, reverse);
    if (mirror != keys[k] && (first < partner || (first == partner && code < reverse)) &&
        templateHasKey(keys, keyCount, mirror)) {
      if (duplicates < 8)
        fprintf(stderr, "mdrt: template coverage: units %lld and %lld, image %d %d %d, "
                "have entries in both orders\n", (long long)first, (long long)partner,
                n[0], n[1], n[2]);
      ++duplicates;
    }
  }
  templateCoverageDuplicates += duplicates;
  /* Sites of each unit, from the frames that were evaluated. */
  int64_t siteRoom = 4 * units + 1;
  double (*site)[3] = malloc((size_t)siteRoom * sizeof(*site));
  int64_t *firstSite = malloc((size_t)(units + 1) * sizeof(int64_t));
  if (!site || !firstSite) abort();
  int64_t sites = 0;
  for (int64_t u = 0; u != units; ++u) {
    firstSite[u] = sites;
    if (u < W) {
      const float *o = frames + 9 * u;
      for (int k = 0; k != 3; ++k)
        for (int c = 0; c != 3; ++c) site[sites + k][c] = o[3 * k + c];
      sites += 3;
      if (hasSite) {
        for (int c = 0; c != 3; ++c)
          site[sites][c] = (double)o[c] + alpha * (((double)o[3 + c] - o[c]) + ((double)o[6 + c] - o[c]));
        ++sites;
      }
    } else {
      const float *x = frames + 3 * (3 * W + (u - W));
      for (int c = 0; c != 3; ++c) site[sites][c] = x[c];
      ++sites;
    }
  }
  firstSite[units] = sites;
  /* f64 predicate widened by 2^-20 of the cutoff: it contains the f32
   * evaluator's predicate and the pairs of the f64 state positions. */
  double reach = cutoff * (1.0 + 0x1p-20), reach2 = reach * reach;
  int64_t misses = 0, required = 0;
  for (int64_t u = 0; u != units; ++u)
    for (int64_t v = u + 1; v != units; ++v) {
      if (u >= W && v >= W) {
        int64_t a = u - W, b = v - W, found = 0;
        for (int64_t k = exRows[a]; k < exRows[a + 1]; ++k) found |= exNeighbors[k] == b;
        if (found) continue;
      }
      /* The near images of the two anchors' displacement. */
      double d[3];
      for (int c = 0; c != 3; ++c) d[c] = site[firstSite[u]][c] - site[firstSite[v]][c];
      double fz = d[2] / cz, fy = (d[1] - cy * fz) / by, fx = (d[0] - cx * fz - bx * fy) / ax;
#define TEMPLATE_NEAREST(x) ((int)((x) < 0 ? (x) - .5 : (x) + .5))
      int m0 = TEMPLATE_NEAREST(fx), m1 = TEMPLATE_NEAREST(fy), m2 = TEMPLATE_NEAREST(fz);
#undef TEMPLATE_NEAREST
      for (int n2 = m2 - 1; n2 <= m2 + 1; ++n2)
        for (int n1 = m1 - 1; n1 <= m1 + 1; ++n1)
          for (int n0 = m0 - 1; n0 <= m0 + 1; ++n0) {
            double shift[3] = {n0 * ax + n1 * bx + n2 * cx, n1 * by + n2 * cy, n2 * cz};
            /* Every site is within 0.5 nm of its unit's first site (the
             * extents are about 0.1 nm): farther first sites cannot have a
             * site pair within the reach. */
            double a2 = 0;
            for (int c = 0; c != 3; ++c) {
              double x = d[c] - shift[c];
              a2 += x * x;
            }
            if (a2 > (reach + 1.0) * (reach + 1.0)) continue;
            int near = 0;
            for (int64_t i = firstSite[u]; i < firstSite[u + 1] && !near; ++i)
              for (int64_t j = firstSite[v]; j < firstSite[v + 1] && !near; ++j) {
                double r2 = 0;
                for (int c = 0; c != 3; ++c) {
                  double x = site[i][c] - (site[j][c] + shift[c]);
                  r2 += x * x;
                }
                near = r2 < reach2;
              }
            if (!near) continue;
            ++required;
            /* x_u - (x_v + n L): u first with image n, or v first with -n.
             * A solute-water pair has the solute first. */
            uint64_t forward = templateCoverageKey(u, v, templateImageCode(n0, n1, n2));
            uint64_t backward = templateCoverageKey(v, u, templateImageCode(-n0, -n1, -n2));
            int ok = (u >= W && v < W) ? templateHasKey(keys, keyCount, forward)
                   : (u < W && v >= W) ? templateHasKey(keys, keyCount, backward)
                   : templateHasKey(keys, keyCount, forward) || templateHasKey(keys, keyCount, backward);
            if (!ok) {
              if (misses < 8)
                fprintf(stderr, "mdrt: template coverage: units %lld and %lld, image %d %d %d, have no entry\n",
                        (long long)u, (long long)v, n0, n1, n2);
              ++misses;
            }
          }
    }
  ++templateCoverageChecks;
  templateCoverageMisses += misses;
  templateCoveragePairs += required;
  free(records); free(order); free(lengths); free(frames); free(exRows);
  free(exNeighbors); free(inner); free(keys); free(site); free(firstSite);
  free(references);
}
