#ifndef MDRT_DIST_STATS_H
#define MDRT_DIST_STATS_H
#include <stdint.h>

// The compiled step and its host runtime share these event codes.
enum DistPhase { DistPairs = 1, DistOther, DistCopy, DistSum, DistReduce,
                 DistMigration, DistWait, DistPhaseCount, DistStep };
struct DistObserver {
  void *context;
  void (*domain)(void *, int64_t, int64_t, int64_t, const double *, const double *);
  void (*migration)(void *, int64_t, int64_t);
  void (*transfer)(void *, int64_t, int64_t, int64_t);
  void (*phase)(void *, int64_t);
};
#endif
