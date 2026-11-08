/* Compiler/runtime layout of the template route's single pair allocation.
 * Outer and inner WW/SW/SS ranges are views, never separate allocations.
 * No CUDA headers are needed to check the layout. */
#ifndef MDIR_RUNTIME_TEMPLATEPAIRSABI_H
#define MDIR_RUNTIME_TEMPLATEPAIRSABI_H
#include <stdint.h>
#include <limits.h>

#define MDRT_TEMPLATE_PAIRS_ABI_VERSION 3
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
enum MdrtTemplateLayoutSlot {
  TEMPLATE_LAYOUT_WATERS = 0, TEMPLATE_LAYOUT_ATOMS = 1,
  TEMPLATE_LAYOUT_ROWS = 2, TEMPLATE_LAYOUT_ENTRIES = 3,
  TEMPLATE_LAYOUT_UNITS = 4, TEMPLATE_LAYOUT_OUTER_ROWS = 5,
  TEMPLATE_LAYOUT_INNER_ROWS = 6, TEMPLATE_LAYOUT_OUTER_UNITS = 7,
  TEMPLATE_LAYOUT_INNER_UNITS = 8, TEMPLATE_LAYOUT_OUTER_ENTRIES = 9,
  TEMPLATE_LAYOUT_INNER_ENTRIES = 10, TEMPLATE_LAYOUT_STAGE_UNITS = 11,
  TEMPLATE_LAYOUT_STAGE_ENTRIES = 12, TEMPLATE_LAYOUT_BUILD_COUNTS = 13,
  TEMPLATE_LAYOUT_BUILD_ENTRY_OFFSETS = 14, TEMPLATE_LAYOUT_BUILD_WORK_COUNTS = 15,
  TEMPLATE_LAYOUT_BUILD_WORK_OFFSETS = 16, TEMPLATE_LAYOUT_BUILD_SCAN_SCRATCH = 17
};

/* Header ranges are [row begin/end, entry begin/end, work begin/end].
 * Within each partition, ranges are ordered WW, SW, SS. */
enum MdrtTemplatePairHeader {
  TEMPLATE_PAIR_WATERS = 0, TEMPLATE_PAIR_ATOMS = 1,
  TEMPLATE_PAIR_ROWS = 2, TEMPLATE_PAIR_CAPACITY = 3,
  TEMPLATE_PAIR_OUTER_RANGES = 4, TEMPLATE_PAIR_INNER_RANGES = 22,
  TEMPLATE_PAIR_STAGE_COUNT = 40
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
  layout->scanBlocks = (layout->rows + 1023) / 1024;
  if (!layout->scanBlocks) layout->scanBlocks = 1;
  layout->words = layout->buildScanScratch + layout->scanBlocks;
  return 1;
}
#endif
