#ifndef MDIR_RUNTIME_COPYAUDIT_H
#define MDIR_RUNTIME_COPYAUDIT_H
#include <stdint.h>
/* Private measurement ABI for a device-frame evaluation. Generic copies
   count separately from explicit structural/status metadata boundaries.
   Ranges identify borrowed device buffers, which the allocator does not
   own or record. No driver query of a host pointer is needed. */
struct MdirDeviceCopyAudit {
  uint64_t uploads, downloads, metadataUploads, metadataDownloads;
  uintptr_t first[16], last[16];
  unsigned ranges;
};
#endif
