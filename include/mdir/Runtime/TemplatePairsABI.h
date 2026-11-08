/* Compiler/runtime layout of the template route's single pair allocation.
 * Outer and inner WW/SW/SS ranges are views, never separate allocations.
 * No CUDA headers are needed to check the layout. */
#ifndef MDIR_RUNTIME_TEMPLATEPAIRSABI_H
#define MDIR_RUNTIME_TEMPLATEPAIRSABI_H
#include <stdint.h>
#include <limits.h>

#define MDRT_TEMPLATE_PAIRS_ABI_VERSION 2
#define MDRT_TEMPLATE_PAIR_HEADER_WORDS 48
#define MDRT_TEMPLATE_PAIR_ROW_WORDS 4
#define MDRT_TEMPLATE_PAIR_UNIT_WORDS 4
#define MDRT_TEMPLATE_PAIR_ENTRY_WORDS 4
#define MDRT_TEMPLATE_PAIR_UNIT_ENTRIES 64

enum MdrtTemplateMetadataSlot {
  TEMPLATE_MEMBERS = 0, TEMPLATE_PARTICLE_SLOTS = 1,
  TEMPLATE_SS_ROWS = 2, TEMPLATE_SS_NEIGHBORS = 3,
  TEMPLATE_METADATA_SLOTS = 4
};

/* Header ranges are [row begin/end, entry begin/end, work begin/end].
 * Within each partition, ranges are ordered WW, SW, SS. */
enum MdrtTemplatePairHeader {
  TEMPLATE_PAIR_WATERS = 0, TEMPLATE_PAIR_ATOMS = 1,
  TEMPLATE_PAIR_ROWS = 2, TEMPLATE_PAIR_CAPACITY = 3,
  TEMPLATE_PAIR_OUTER_RANGES = 4, TEMPLATE_PAIR_INNER_RANGES = 22
};
/* A row is [class, first-unit begin, entry begin, entry count].
 * A work unit is [class, first-unit begin, entry begin, entry end].
 * An entry is [partner unit, lattice-image code, first-unit mask, reserved].
 * The first-unit mask has 16 bits; the final first cluster can be partial. */
struct MdrtTemplatePairLayout {
  int64_t waters, atoms, rows, entries, units;
  int64_t outerRows, innerRows, outerUnits, innerUnits;
  int64_t outerEntries, innerEntries, words;
};

/* All payload indices fit i32. Physical offsets use i64, including on
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
  layout->words = layout->innerEntries + 4 * entries;
  return 1;
}
#endif
