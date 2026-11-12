/* Fixed membership preparation, shared with the runtime's CPU fixture.
 * Input tuples may be renumbered and need not have consecutive members.
 * Stored extra points belong to the water class but are not real units. */
#ifndef MDIR_RUNTIME_TEMPLATEMEMBERSHIP_H
#define MDIR_RUNTIME_TEMPLATEMEMBERSHIP_H
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

static inline int mdrtTemplateMembership(
    int64_t particles, int64_t waters, int64_t sites,
    const int32_t *members, int64_t rowStride, int64_t memberStride,
    int32_t *units, int64_t capacity) {
  if (particles < 0 || particles > INT32_MAX || waters < 0 ||
      (sites != 3 && sites != 4) || waters > particles / sites ||
      rowStride < 0 || memberStride < 0)
    return 0;
  int64_t atoms = particles - waters * sites;
  if (capacity < 3 * waters + atoms ||
      ((3 * waters + atoms) && !units) || (waters && !members))
    return 0;
  unsigned char *covered = (unsigned char *)calloc((size_t)particles, 1);
  if (particles && !covered) return 0;
  int valid = 1;
  for (int64_t water = 0; water < waters && valid; ++water)
    for (int64_t site = 0; site < sites; ++site) {
      int32_t atom = members[water * rowStride + site * memberStride];
      if (atom < 0 || atom >= particles || covered[atom]) {
        valid = 0;
        break;
      }
      covered[atom] = 1;
      if (site < 3) units[3 * water + site] = atom;
    }
  int64_t next = 3 * waters;
  if (valid)
    for (int64_t atom = 0; atom < particles; ++atom)
      if (!covered[atom]) units[next++] = (int32_t)atom;
  free(covered);
  return valid && next == 3 * waters + atoms;
}
#endif
