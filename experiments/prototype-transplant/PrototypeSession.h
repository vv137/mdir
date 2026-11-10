// Experimental I/O/lifecycle wrapper. Archived device bodies remain unchanged.
#ifndef MDIR_EXPERIMENT_PROTOTYPE_SESSION_H
#define MDIR_EXPERIMENT_PROTOTYPE_SESSION_H
#include "EntryFormat.h"
#include <cstdint>
#include <string>
#include <vector>
namespace prototype_io {
struct Session;
struct SessionInfo {
  int waters, atoms, sites, particles;
  uint64_t builds[3], prunes[3], evaluations;
};
struct SessionDeviceList {
  const void *outerEntries, *innerEntries; // int2 arrays
  const int32_t *outerCount, *innerCount, *blockStart;
  const int32_t *unitGroup, *unitOrdinal, *activeBlocks, *activeCount;
  int groups, blocks;
};
struct SessionDeviceView {
  const int32_t *waterOrder, *atomOrder;
  const void *waterShifts, *atomShifts; // int4, indexed by logical unit ID
  int waterGroups, atomGroups, waterPadded, atomPadded;
  SessionDeviceList classes[3];
};
Session *createSession(const std::string &sidecar);
void destroySession(Session *);
SessionInfo sessionInfo(const Session *);
// Borrowed device pointers; valid until the next build or session destruction.
SessionDeviceView sessionDeviceView(const Session *);
// Fractions are host f64, water units first, then atom units. Cell/reaches nm.
void buildSession(Session *, bool waterActive, bool atomActive,
    const double *fractions, const double cellNm[3], const double reachesNm[3],
    const std::vector<int32_t> &ssRows, const std::vector<int32_t> &ssNeighbors);
void pruneSession(Session *, bool waterActive, bool atomActive,
    const double *currentFractions, const double innerReachesNm[3]);
// The returned records retain the original prototype group/slot/image basis.
void exportSession(Session *, bool inner, PrototypeBasis &,
                   std::vector<PrototypeEntry> &);
// Input is DEVICE f64 Angstrom in the native retained packing frame,
// solute first then whole OHH[EP] per water. The caller supplies the native
// lattice shift; the unchanged pack kernel adds the prototype's own shift.
void prepareSession(Session *, const double *deviceCoordinatesAngstrom);
// Fast path: keep inactive species' packed inner references, copy the active
// species from prepareSession's original pack output, then prune/select.
// Do not mix this API with the host-fraction pruneSession on one session.
void prunePreparedSession(Session *, bool waterActive, bool atomActive,
                          const double innerReachesNm[3]);
const float *forcePreparedSession(Session *);
// Convenience prepare + force (a prune must already have initialized lists).
// Output is DEVICE f32 kcal/mol/Angstrom in the same original-particle layout.
// The caller owns native-unit conversion and additive force publication.
const float *forceSession(Session *, const double *deviceCoordinatesAngstrom);
}
#endif
