/* Shared compiler/runtime contract for the neighbor-group buffers.
 * Slot numbers are stable ABI values, not declaration positions.
 * Bump MDRT_GROUPS_ABI_VERSION when this contract changes; the host compile
 * cache includes it in its compatibility key. No CUDA headers are needed. */
#ifndef MDIR_RUNTIME_GROUPS_ABI_H
#define MDIR_RUNTIME_GROUPS_ABI_H

#define MDRT_GROUPS_ABI_VERSION 1

enum MdrtGroupsSlot {
  GROUPS_ORDER = 0,
  GROUPS_COUNTS = 1,
  GROUPS_ENTRIES = 2,
  GROUPS_MASKS = 3,
  GROUPS_UNIT_GROUPS = 4,
  GROUPS_UNIT_ORDINALS = 5,
  GROUPS_SHIFTS = 6,
  GROUPS_INNER_ENTRIES = 7,
  GROUPS_INNER_MASKS = 8,
  GROUPS_INNER_COUNTS = 9,
  GROUPS_BLOCK_OF = 10,
  GROUPS_ACTIVE_UNITS = 11,
  GROUPS_ACTIVE_COUNT = 12,
  GROUPS_ACTIVE_RUNS = 13,
  GROUPS_ACTIVE_WIDE_RUNS = 14,
  GROUPS_PLACED = 15,
  GROUPS_BUFFERS = 16
};

enum MdrtGroupsElement { MDRT_GROUPS_I32, MDRT_GROUPS_F32 };
/* Shapes count physical elements, including padding. PLACED is exposed
 * separately as [places, 3]; the generic accessor is its flat storage view. */
enum MdrtGroupsShape {
  MDRT_GROUPS_PLACES, MDRT_GROUPS_GROUPS, MDRT_GROUPS_ENTRIES,
  MDRT_GROUPS_BLOCKS, MDRT_GROUPS_BLOCK_OF, MDRT_GROUPS_SCALAR,
  MDRT_GROUPS_RUNS, MDRT_GROUPS_WIDE_RUNS, MDRT_GROUPS_PLACED
};
struct MdrtGroupsDescriptor {
  enum MdrtGroupsElement element;
  enum MdrtGroupsShape shape;
  const char *role;
};
static inline const struct MdrtGroupsDescriptor *
mdrtGroupsDescriptor(long long slot) {
  static const struct MdrtGroupsDescriptor descriptors[GROUPS_BUFFERS] = {
    {MDRT_GROUPS_I32, MDRT_GROUPS_PLACES, "groups: order"},
    {MDRT_GROUPS_I32, MDRT_GROUPS_GROUPS, "groups: counts"},
    {MDRT_GROUPS_I32, MDRT_GROUPS_ENTRIES, "groups: entries"},
    {MDRT_GROUPS_I32, MDRT_GROUPS_ENTRIES, "groups: masks"},
    {MDRT_GROUPS_I32, MDRT_GROUPS_BLOCKS, "groups: unit groups"},
    {MDRT_GROUPS_I32, MDRT_GROUPS_BLOCKS, "groups: unit ordinals"},
    {MDRT_GROUPS_I32, MDRT_GROUPS_PLACES, "groups: shifts"},
    {MDRT_GROUPS_I32, MDRT_GROUPS_ENTRIES, "groups: inner entries"},
    {MDRT_GROUPS_I32, MDRT_GROUPS_ENTRIES, "groups: inner masks"},
    {MDRT_GROUPS_I32, MDRT_GROUPS_BLOCKS, "groups: inner counts"},
    {MDRT_GROUPS_I32, MDRT_GROUPS_BLOCK_OF, "groups: block of"},
    {MDRT_GROUPS_I32, MDRT_GROUPS_BLOCKS, "groups: active units"},
    {MDRT_GROUPS_I32, MDRT_GROUPS_SCALAR, "groups: active count"},
    {MDRT_GROUPS_I32, MDRT_GROUPS_RUNS, "groups: active runs"},
    {MDRT_GROUPS_I32, MDRT_GROUPS_WIDE_RUNS, "groups: active wide runs"},
    {MDRT_GROUPS_F32, MDRT_GROUPS_PLACED, "groups: placed"},
  };
  return slot >= 0 && slot < GROUPS_BUFFERS ? &descriptors[slot] : 0;
}
#endif
