#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUdevice;
typedef uintptr_t CUdeviceptr;
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 1
static int cuCtxGetDevice(int *device) {
  *device = 0;
  return 0;
}
static int cuDeviceGetAttribute(int *value, int attribute, int device) {
  assert(attribute == 1 && device == 0);
  *value = 128;
  return 0;
}
static int cuMemFree(CUdeviceptr ptr) {
  free((void *)ptr);
  return 0;
}
static void check(int status, const char *call) {
  (void)call;
  assert(status == 0);
}
struct MdrtTuples {
  int particles, arity, tuples, ordered;
  int32_t *deviceRenumbered;
  int32_t *file;
  struct BondedDirectory *bondedPlans;
};
static struct Store {
  size_t deviceBytes[3];
} store;
static struct Store *deviceTupleStore(void) { return &store; }
#define tuplesDeviceKept malloc
#define tuplesToDevice memcpy
#define mdrtTuplesAllocate malloc
#define tuplesToHost memcpy
struct MdrtMemref1 {
  void *aligned;
  size_t size;
};
struct MdrtMemref2 {
  void *aligned;
  size_t offset;
};
static void setMemref1(struct MdrtMemref1 *m, void *p, size_t size) {
  m->aligned = p;
  m->size = size;
}
#include "mdrt_bonded_units.h"

static void test(int owners, int families, int unpacked) {
  struct BondedDirectory plan = {.count = families, .unpacked = unpacked};
  struct MdrtTuples sets[4] = {0};
  int perGroup = 80;
  size_t slots = (size_t)owners * perGroup * families * 2;
  int *seen = calloc(slots + 1, sizeof(*seen));
  for (int s = 0; s < families; ++s) {
    struct MdrtTuples *set = &sets[s];
    set->particles = 128 * (owners ? owners : 1);
    set->arity = 2;
    set->tuples = perGroup * owners;
    set->file = calloc(2 * set->tuples + 1, sizeof(int32_t));
    for (int t = 0; t < set->tuples; ++t) {
      int g = t / perGroup;
      set->file[2 * t] = 128 * g + (t % perGroup);
      set->file[2 * t + 1] = 128 * ((g + 1) % owners) + (t % perGroup + 1);
    }
    plan.sets[s] = set;
  }
  buildBondedDirectory(&plan);
  int capped = families == 4 && !unpacked && owners < 128;
  assert(owners == 0 ? plan.blocks == 1
         : capped    ? plan.blocks > owners
                     : plan.blocks == owners);
  const int32_t *dir = plan.words;
  int groupCount = owners ? owners : 1;
  int *blocksPerGroup = calloc(groupCount, sizeof(int));
  for (int b = 0; b < plan.blocks; ++b) {
    int group = dir[dir[b] + 1];
    assert(group >= 0 && group < groupCount);
    ++blocksPerGroup[group];
  }
  for (int b = 0; b < plan.blocks; ++b) {
    const int32_t *row = dir + dir[b];
    int count = *row++;
    int group = *row++, blocks = *row++;
    assert(blocks == blocksPerGroup[group]);
    const int32_t *length = row, *atoms = row + families;
    row = atoms + count;
    int terms = 0;
    for (int s = 0; s < families; ++s)
      for (int j = 0; j < length[s]; ++j) {
        ++terms;
        int t = *row++;
        assert(t >= 0 && t < sets[s].tuples);
        int packed = unpacked ? 0 : *row++;
        for (int k = 0; k < 2; ++k) {
          int slot = unpacked ? *row++ : (packed >> (10 * k)) & 1023;
          assert(slot >= 0 && slot < count);
          int encoded = atoms[slot];
          assert((encoded < 0 ? ~encoded : encoded) == sets[s].file[2 * t + k]);
          assert((encoded >= 0) == (sets[s].file[2 * t + k] / 128 == group));
          if (encoded >= 0)
            ++seen[2 * (s * sets[s].tuples + t) + k];
        }
      }
    assert(!capped || terms <= 64);
    assert(row - dir ==
           (b + 1 == plan.blocks ? plan.size : (size_t)dir[b + 1]));
  }
  for (size_t k = 0; k < slots; ++k)
    assert(seen[k] == 1);
  printf("%d owners, %d families, unpacked=%d: %ld blocks, %zu owned slots "
         "checked\n",
         owners, families, unpacked, (long)plan.blocks, slots);
  free(blocksPerGroup);
  free(seen);
  free(plan.words);
  for (int s = 0; s < families; ++s)
    free(sets[s].file);
}
int main(void) {
  test(0, 4, 0);
  test(1, 4, 0);
  test(127, 4, 0);
  test(128, 4, 0);
  test(129, 4, 0);
  test(2, 3, 0);
  test(2, 4, 1);
}
