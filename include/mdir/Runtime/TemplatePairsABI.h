/* Compiler/runtime layout of the template route's single pair allocation.
 * Outer and inner WW/SW/SS ranges are views, never separate allocations.
 * No CUDA headers are needed to check the layout. */
#ifndef MDIR_RUNTIME_TEMPLATEPAIRSABI_H
#define MDIR_RUNTIME_TEMPLATEPAIRSABI_H
#include <stdint.h>
#include <limits.h>

#define MDRT_TEMPLATE_PAIRS_ABI_VERSION 5
#define MDRT_TEMPLATE_PAIR_HEADER_WORDS 48
#define MDRT_TEMPLATE_PAIR_ROW_WORDS 4
#define MDRT_TEMPLATE_PAIR_UNIT_WORDS 4
#define MDRT_TEMPLATE_PAIR_ENTRY_WORDS 4
#define MDRT_TEMPLATE_PAIR_UNIT_ENTRIES 64
#define MDRT_TEMPLATE_IMAGE_BITS 10
#define MDRT_TEMPLATE_IMAGE_OFFSET 512
#define MDRT_TEMPLATE_IMAGE_MASK 1023

enum MdrtTemplateMetadataSlot {
  TEMPLATE_MEMBERS = 0, TEMPLATE_PARTICLE_SLOTS = 1,
  TEMPLATE_SS_ROWS = 2, TEMPLATE_SS_NEIGHBORS = 3,
  TEMPLATE_METADATA_SLOTS = 4
};
/* Fixed-size f64 views, independent of pair-arena capacity. Every view
 * has three columns. A class cell occupies two rows (ax,by,cz,bx,cx,cy),
 * with the second row unused for an orthorhombic cell. */
enum MdrtTemplateFrameSlot {
  TEMPLATE_FRAME_OUTER_REFERENCE = 0, TEMPLATE_FRAME_INNER_REFERENCE = 1,
  TEMPLATE_FRAME_PACKING_CELLS = 2, TEMPLATE_FRAME_CURRENT_POINTS = 3,
  TEMPLATE_FRAME_OUTER_CELLS = 4, TEMPLATE_FRAME_INNER_CELLS = 5,
  /* Two rows of actual build reaches: outer WW/SW/SS, then inner. This
   * retains an initial-input extent widening in the validity theorem. */
  TEMPLATE_FRAME_REACHES = 6, TEMPLATE_FRAME_SLOTS = 7
};
enum MdrtTemplateLayoutSlot {
  TEMPLATE_LAYOUT_WATERS = 0, TEMPLATE_LAYOUT_ATOMS = 1,
  TEMPLATE_LAYOUT_ROWS = 2, TEMPLATE_LAYOUT_ENTRIES = 3,
  TEMPLATE_LAYOUT_UNITS = 4, TEMPLATE_LAYOUT_OUTER_ROWS = 5,
  TEMPLATE_LAYOUT_INNER_ROWS = 6, TEMPLATE_LAYOUT_OUTER_UNITS = 7,
  TEMPLATE_LAYOUT_INNER_UNITS = 8, TEMPLATE_LAYOUT_OUTER_ENTRIES = 9,
  TEMPLATE_LAYOUT_INNER_ENTRIES = 10, TEMPLATE_LAYOUT_STAGE_UNITS = 11,
  TEMPLATE_LAYOUT_STAGE_ENTRIES = 12, TEMPLATE_LAYOUT_BUILD_COUNTS = 13,
  TEMPLATE_LAYOUT_BUILD_ENTRY_OFFSETS = 14, TEMPLATE_LAYOUT_BUILD_WORK_COUNTS = 15,
  TEMPLATE_LAYOUT_BUILD_WORK_OFFSETS = 16, TEMPLATE_LAYOUT_BUILD_SCAN_SCRATCH = 17,
  TEMPLATE_LAYOUT_SPATIAL_KEYS = 18, TEMPLATE_LAYOUT_SPATIAL_Z_KEYS = 19,
  TEMPLATE_LAYOUT_SPATIAL_ITEMS = 20, TEMPLATE_LAYOUT_SPATIAL_PLACES = 21,
  TEMPLATE_LAYOUT_SPATIAL_COUNTS = 22, TEMPLATE_LAYOUT_SPATIAL_STARTS = 23,
  TEMPLATE_LAYOUT_SPATIAL_CURSORS = 24, TEMPLATE_LAYOUT_SPATIAL_BINS = 25,
  TEMPLATE_LAYOUT_SCAN_BLOCKS = 26
};

/* Header ranges are [row begin/end, entry begin/end, work begin/end].
 * Within each partition, ranges are ordered WW, SW, SS. */
enum MdrtTemplatePairHeader {
  TEMPLATE_PAIR_WATERS = 0, TEMPLATE_PAIR_ATOMS = 1,
  TEMPLATE_PAIR_ROWS = 2, TEMPLATE_PAIR_CAPACITY = 3,
  TEMPLATE_PAIR_OUTER_RANGES = 4, TEMPLATE_PAIR_INNER_RANGES = 22,
  TEMPLATE_PAIR_STAGE_COUNT = 40
};
/* Four aligned i64 controls occupy header words 40..47. The low word of
 * the chunk count is also TEMPLATE_PAIR_STAGE_COUNT, after bounds checks. */
enum MdrtTemplateCounterSlot {
  TEMPLATE_COUNTER_CHUNKS = 0, TEMPLATE_COUNTER_FLAGS = 1,
  TEMPLATE_COUNTER_ENTRIES = 2, TEMPLATE_COUNTER_EXTENT = 3
};
enum MdrtTemplateBuildFlag {
  TEMPLATE_BUILD_INDEX_OVERFLOW = 1, TEMPLATE_BUILD_CAPACITY_OVERFLOW = 2
};
/* A row is [class, first-unit begin, entry begin, entry count].
 * A work unit is [class, first-unit begin, entry begin, entry end].
 * An entry is [partner unit, lattice-image code, first-unit mask, reserved].
 * The first-unit mask has 16 bits; the final first cluster can be partial.
 * Image components occupy ten bits each, x first, with offset 512. They
 * shift the already packed partner frame, independently of its winding. */
/* A staging unit is [row, count, row-chunk ordinal, reserved]. Its entries
 * are at 64 times its staging-unit index, formed as a physical i64 address.
 * A search writes at most 64 entries per chunk and one total per row. */
struct MdrtTemplatePairLayout {
  int64_t waters, atoms, rows, entries, units;
  int64_t outerRows, innerRows, outerUnits, innerUnits;
  int64_t outerEntries, innerEntries, words;
  int64_t stageUnits, stageEntries, buildCounts, buildEntryOffsets;
  int64_t buildWorkCounts, buildWorkOffsets, buildScanScratch, scanBlocks;
  int64_t spatialKeys, spatialZKeys, spatialItems, spatialPlaces;
  int64_t spatialCounts, spatialStarts, spatialCursors, spatialBins;
};

/* All payload indices fit signless i32. Padded class places use unsigned
 * interpretation; original unit IDs use nonnegative signed i32 with -1
 * reserved for a vacant place. Physical offsets use i64, including on
 * allocations whose flattened byte address exceeds the i32 range. */
static inline int mdrtTemplatePairLayout(
    int64_t waters, int64_t atoms, int64_t entries,
    struct MdrtTemplatePairLayout *layout) {
  if (!layout || waters < 0 || atoms < 0 || entries < 0 ||
      waters > INT32_MAX || atoms > INT32_MAX || entries > INT32_MAX ||
      waters + atoms > INT32_MAX)
    return 0;
  layout->waters = waters;
  layout->atoms = atoms;
  layout->rows = (waters + 15) / 16 + 2 * ((atoms + 15) / 16);
  layout->entries = entries;
  /* Splitting every row into blocks of 64 takes at most this many units,
   * even when every nonempty row has a partial final block. */
  layout->units = (entries + 63) / 64 + layout->rows;
  layout->outerRows = MDRT_TEMPLATE_PAIR_HEADER_WORDS;
  layout->innerRows = layout->outerRows + 4 * layout->rows;
  layout->outerUnits = layout->innerRows + 4 * layout->rows;
  layout->innerUnits = layout->outerUnits + 4 * layout->units;
  layout->outerEntries = layout->innerUnits + 4 * layout->units;
  layout->innerEntries = layout->outerEntries + 4 * entries;
  // One search stages full chunks before the row scans give compact class
  // offsets. Staging and scan scratch are views of the same allocation,
  // not another pair list. A stage chunk's index fits i32; its physical
  // entry address (64 * chunk) must be formed in i64/index.
  layout->stageUnits = layout->innerEntries + 4 * entries;
  layout->stageEntries = layout->stageUnits + 4 * layout->units;
  layout->buildCounts = layout->stageEntries + 4 * 64 * layout->units;
  layout->buildEntryOffsets = layout->buildCounts + layout->rows;
  layout->buildWorkCounts = layout->buildEntryOffsets + layout->rows + 1;
  layout->buildWorkOffsets = layout->buildWorkCounts + layout->rows;
  layout->buildScanScratch = layout->buildWorkOffsets + layout->rows + 1;
  /* The grid builder uses at most four times ceil(units / 64) bins.
   * Twelve bins also permit a small non-square grid without a separate
   * allocation. Both scans share scratch, at distinct stages of a build. */
  layout->spatialBins = 4 * ((waters + atoms + 63) / 64);
  if (layout->spatialBins < 12) layout->spatialBins = 12;
  int64_t scanSize = layout->rows > layout->spatialBins
                        ? layout->rows : layout->spatialBins;
  layout->scanBlocks = (scanSize + 1023) / 1024;
  if (!layout->scanBlocks) layout->scanBlocks = 1;
  layout->spatialKeys = layout->buildScanScratch + layout->scanBlocks;
  layout->spatialZKeys = layout->spatialKeys + waters + atoms;
  layout->spatialItems = layout->spatialZKeys + waters + atoms;
  layout->spatialPlaces = layout->spatialItems + waters + atoms;
  layout->spatialCounts = layout->spatialPlaces + waters + atoms;
  layout->spatialStarts = layout->spatialCounts + layout->spatialBins;
  layout->spatialCursors = layout->spatialStarts + layout->spatialBins + 1;
  layout->words = layout->spatialCursors + layout->spatialBins;
  return 1;
}
#endif
