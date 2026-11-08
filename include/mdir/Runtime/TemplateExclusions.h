/* Static SS exclusion graph. Water membership has already been proved;
 * WW and SW have no exclusion lookup. CSR indices refer to the fixed atom
 * complement, independently of its changing spatial order. */
#ifndef MDIR_RUNTIME_TEMPLATEEXCLUSIONS_H
#define MDIR_RUNTIME_TEMPLATEEXCLUSIONS_H
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

struct MdrtTemplateExclusions {
  int32_t *rows, *neighbors;
  int64_t atoms, count;
};

static inline void mdrtFreeTemplateExclusions(struct MdrtTemplateExclusions *graph) {
  free(graph->rows);
  free(graph->neighbors);
  graph->rows = NULL;
  graph->neighbors = NULL;
  graph->atoms = graph->count = 0;
}

static inline int mdrtCompareTemplatePairKeys(const void *lhs, const void *rhs) {
  uint64_t a = *(const uint64_t *)lhs, b = *(const uint64_t *)rhs;
  return a < b ? -1 : a > b;
}

static inline int mdrtTemplateExclusions(
    int64_t particles, int64_t waters, int64_t sites, const int32_t *units,
    int64_t excludedCount, const int32_t *excluded,
    int64_t rowStride, int64_t memberStride,
    struct MdrtTemplateExclusions *result) {
  if (!result) return 0;
  result->rows = result->neighbors = NULL;
  result->atoms = result->count = 0;
  if (particles < 0 || particles > INT32_MAX || waters < 0 ||
      (sites != 3 && sites != 4) || waters > particles / sites ||
      excludedCount < 0 || excludedCount > INT32_MAX / 2 ||
      rowStride < 0 || memberStride < 0 ||
      (excludedCount && !excluded))
    return 0;
  int64_t atoms = particles - sites * waters;
  if (atoms && !units) return 0;
  int32_t *atomOf = (int32_t *)malloc((size_t)(particles ? particles : 1) * sizeof(int32_t));
  uint64_t *keys = (uint64_t *)malloc((size_t)(excludedCount ? excludedCount : 1) * sizeof(uint64_t));
  if (!atomOf || !keys) { free(atomOf); free(keys); return 0; }
  for (int64_t i = 0; i < particles; ++i) atomOf[i] = -1;
  int valid = 1;
  for (int64_t atom = 0; atom < atoms; ++atom) {
    int32_t particle = units[3 * waters + atom];
    if (particle < 0 || particle >= particles || atomOf[particle] != -1) {
      valid = 0; break;
    }
    atomOf[particle] = (int32_t)atom;
  }
  int64_t count = 0;
  for (int64_t pair = 0; valid && pair < excludedCount; ++pair) {
    int32_t p = excluded[pair * rowStride];
    int32_t q = excluded[pair * rowStride + memberStride];
    if (p < 0 || p >= particles || q < 0 || q >= particles) {
      valid = 0; break;
    }
    int32_t a = atomOf[p], b = atomOf[q];
    if (a < 0 || b < 0 || a == b) continue;
    if (a > b) { int32_t swap = a; a = b; b = swap; }
    keys[count++] = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
  }
  free(atomOf);
  if (!valid) { free(keys); return 0; }
  qsort(keys, (size_t)count, sizeof(uint64_t), mdrtCompareTemplatePairKeys);
  int64_t unique = 0;
  for (int64_t i = 0; i < count; ++i)
    if (!i || keys[i] != keys[i - 1]) keys[unique++] = keys[i];
  int32_t *rows = (int32_t *)calloc((size_t)(atoms + 1), sizeof(int32_t));
  int32_t *neighbors = (int32_t *)malloc((size_t)(unique ? 2 * unique : 1) * sizeof(int32_t));
  int32_t *next = (int32_t *)malloc((size_t)(atoms ? atoms : 1) * sizeof(int32_t));
  if (!rows || !neighbors || !next) {
    free(keys); free(rows); free(neighbors); free(next); return 0;
  }
  for (int64_t i = 0; i < unique; ++i) {
    ++rows[(uint32_t)(keys[i] >> 32) + 1];
    ++rows[(uint32_t)keys[i] + 1];
  }
  for (int64_t i = 0; i < atoms; ++i) {
    rows[i + 1] += rows[i];
    next[i] = rows[i];
  }
  /* Canonical pair order inserts all lower neighbors before higher ones,
   * each ascending, so every CSR row is sorted without another sort. */
  for (int64_t i = 0; i < unique; ++i) {
    int32_t a = (int32_t)(keys[i] >> 32), b = (int32_t)(uint32_t)keys[i];
    neighbors[next[a]++] = b;
    neighbors[next[b]++] = a;
  }
  free(keys);
  free(next);
  result->rows = rows;
  result->neighbors = neighbors;
  result->atoms = atoms;
  result->count = 2 * unique;
  return 1;
}
#endif
