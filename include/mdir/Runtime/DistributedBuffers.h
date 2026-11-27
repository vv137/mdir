#ifndef MDIR_RUNTIME_DISTRIBUTEDBUFFERS_H
#define MDIR_RUNTIME_DISTRIBUTEDBUFFERS_H

#include <stdint.h>

enum MdirDistributedBuffer {
  MdirMatrixEntries,
  MdirMatrixRows,
  MdirImageSlots,
  MdirTupleIncidence,
  MdirDeviceScratch,
  MdirHaloPlans,
  MdirHaloImageSlots,
  MdirHaloExchange,
  MdirHaloPairIds,
  MdirTuplePlans,
  MdirTupleImports,
  MdirTransportScratch,
  MdirPME,
  MdirDistributedBuffers
};

static inline const char *mdirDistributedBufferName(int slot) {
  static const char *const names[] = {
    "matrix_entries", "matrix_rows", "image_slots", "tuple_incidence",
    "device_scratch", "halo_plans", "halo_image_slots", "halo_exchange", "halo_pair_ids", "tuple_plans",
    "tuple_imports", "transport_scratch", "pme"
  };
  return names[slot];
}

#endif
