#ifndef MDRT_DIST_STATS_H
#define MDRT_DIST_STATS_H
#include <stdint.h>

// The compiled step and its host runtime share these event codes.
enum DistPhase { DistPairs = 1, DistOther, DistCopy, DistSum, DistReduce,
                 DistMigration, DistWait, DistList, DistPack, DistIntra, DistInter,
                 DistIntegrate, DistOutput, DistState, DistHaloBuild, DistSpread, DistOverlap, DistFFT, DistTranspose, DistSolve, DistGather, DistPhaseCount, DistStep, DistTupleWork };
/* The quantities of the fixed-point words of the deterministic mode
   (D447); a range error names its quantity. */
enum DistQuantity {
  DistPairEnergy,
  DistPairForce,
  DistParticleSum,
  DistKinetic,
  DistMomentum,
  DistForceSquare,
  DistQuantityCount
};
enum DistMeshTraffic { DistOverlapBytes, DistTransposeBytes, DistMeshTrafficCount };
#ifdef __cplusplus
extern "C" {
#endif
void mdrtDistMeshBytes(enum DistMeshTraffic kind, int64_t bytes);
#ifdef __cplusplus
}
#endif
struct DistObserver {
  void *context;
  void (*domain)(void *, int64_t, int64_t, int64_t, const double *, const double *);
  void (*movementCounts)(void *, int64_t, int64_t, int64_t);
  void (*transfer)(void *, int64_t, int64_t, int64_t);
  void (*phase)(void *, int64_t);
  void (*meshBytes)(void *, enum DistMeshTraffic, int64_t);
};
#endif
