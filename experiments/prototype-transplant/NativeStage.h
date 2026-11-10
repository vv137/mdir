// Experimental I/O conversion only: no distance or search predicate.
#ifndef MDIR_EXPERIMENT_PROTOTYPE_NATIVE_STAGE_H
#define MDIR_EXPERIMENT_PROTOTYPE_NATIVE_STAGE_H
#include "PrototypeSession.h"
namespace prototype_io {
// Launches on the CUDA default stream. The caller orders prior producers
// and subsequent native-stream consumers. Controls[3] (extent) is preserved.
void stageNative(Session *, bool inner, int32_t *deviceStageUnits,
    int32_t *deviceStageEntries, int32_t *deviceRowCounts,
    uint64_t *deviceControls, uint64_t unitCapacity);
}
#endif
