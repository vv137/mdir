/* Shared compiler/runtime contract for the neighbor-group buffers.
 * Slot numbers are stable ABI values, not declaration positions.
 * Bump MDRT_GROUPS_ABI_VERSION when this contract changes; the host compile
 * cache includes it in its compatibility key. No CUDA headers are needed. */
#ifndef MDIR_RUNTIME_GROUPS_ABI_H
#define MDIR_RUNTIME_GROUPS_ABI_H

#define MDRT_GROUPS_ABI_VERSION 5

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
  /* A list of units (D417, D419): the count and the
     last block of each list of a group in a segment (g S + s), the order of
     the places as the traversal reads it, the records in the order of the
     segments (entries, masks, groups, counts), and the first record of each
     segment. Allocated where they are first used, so a structure of
     groups takes none of them. */
  GROUPS_SEGMENT_COUNTS = 16,
  GROUPS_SEGMENT_LAST = 17,
  GROUPS_SEGMENT_ORDER = 18,
  GROUPS_RECORD_ENTRIES = 19,
  GROUPS_RECORD_MASKS = 20,
  GROUPS_RECORD_GROUPS = 21,
  GROUPS_RECORD_COUNTS = 22,
  GROUPS_SEGMENT_OFFSETS = 23,
  GROUPS_OUTER_DEGREES = 24,
  GROUPS_OUTER_MAX = 25,
  GROUPS_BUFFERS = 26
};

/* The frames of a list of units (D422): buffers of
   32-bit values, `rows` a place times their width, which the writer and
   the refresh write and the loops over the list read
   (mdrtGroupsFrame, mdrtGroupsFrameField*). Frame 0 holds the anchors and
   frame 1 the offsets of the members; the fields that the loops read
   follow. A growth of the places drops them with the other buffers. */
#define MDRT_GROUPS_FRAMES 16

enum MdrtGroupsElement { MDRT_GROUPS_I32, MDRT_GROUPS_F32, MDRT_GROUPS_I64 };
/* Shapes count physical elements, including padding. PLACED is exposed
 * separately as [places, 3]; the generic accessor is its flat storage view. */
enum MdrtGroupsShape {
  MDRT_GROUPS_PLACES, MDRT_GROUPS_GROUPS, MDRT_GROUPS_ENTRIES,
  MDRT_GROUPS_BLOCKS, MDRT_GROUPS_BLOCK_OF, MDRT_GROUPS_SCALAR,
  MDRT_GROUPS_RUNS, MDRT_GROUPS_WIDE_RUNS, MDRT_GROUPS_PLACED,
  MDRT_GROUPS_SEGMENT_LISTS, MDRT_GROUPS_SEGMENTS
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
      {MDRT_GROUPS_I32, MDRT_GROUPS_SEGMENT_LISTS, "units: list counts"},
      {MDRT_GROUPS_I32, MDRT_GROUPS_SEGMENT_LISTS, "units: list last blocks"},
      {MDRT_GROUPS_I32, MDRT_GROUPS_PLACES, "units: order"},
      {MDRT_GROUPS_I32, MDRT_GROUPS_ENTRIES, "units: record entries"},
      {MDRT_GROUPS_I32, MDRT_GROUPS_ENTRIES, "units: record masks"},
      {MDRT_GROUPS_I32, MDRT_GROUPS_BLOCKS, "units: record groups"},
      {MDRT_GROUPS_I32, MDRT_GROUPS_BLOCKS, "units: record counts"},
      {MDRT_GROUPS_I32, MDRT_GROUPS_SEGMENTS, "units: segment offsets"},
      {MDRT_GROUPS_I64, MDRT_GROUPS_PLACES, "groups: outer degrees"},
      {MDRT_GROUPS_I64, MDRT_GROUPS_SCALAR, "groups: outer maximum degree"},
  };
  return slot >= 0 && slot < GROUPS_BUFFERS ? &descriptors[slot] : 0;
}
#endif
