#ifndef MDRT_MPI_H
#define MDRT_MPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Rank intervals are consecutive in the global domain order. */
int64_t mdrtMPIDomainBound(int64_t domains, int64_t end);
int mdrtMPIOwner(int64_t domains, int64_t domain);
int mdrtMPIRank(void);
int mdrtMPISize(void);
int mdrtMPINodeSize(void);
void mdrtMPIRowsToRoot(double *values, int64_t count, int owner);
void mdrtMPIBroadcastIndices(int64_t *values, int64_t count, int root);
void mdrtMPIBroadcastWords(void *values, int64_t count, int root);
void mdrtMPIBroadcast(double *values, int64_t count, int root);
int mdrtMPISameNode(int peer);
struct HaloRequests;
struct HaloRequests *mdrtMPIStartHalo(void *send, const int *ns, const int *os,
    void *recv, const int *nr, const int *offsets, int words, int transpose);
void mdrtMPIWaitHalo(struct HaloRequests *pending);
void mdrtMPIGatherRankValue(const void *value, void *terms, int words);
void mdrtMPIShareBlocks(void *values, const int *counts, const int *offsets, int words);
void mdrtMPIExchange(void *send, int *sendCounts, int *sendOffsets,
                     void *recv, int *recvCounts, int *recvOffsets, int words);
void mdrtMPIExchangeFloats(float *send, int *ns, int *os,
                           float *recv, int *nr, int *offsets);
int mdrtMPIShareDomainPartials(void *values, int64_t domains, int64_t width, int words);
void mdrtMPICollectIntervals(void *data, int64_t count, int64_t width, int integer, int rootOnly);
void mdrtMPIShareIndexRows(int64_t *rows, int64_t domains, int64_t width);
void mdrtMPIShareDomainCounts(int64_t *rows, int64_t domains);
void mdrtMPISumMigration(int64_t *counts, int64_t domains);
void mdrtMPIExchangeCounts(const int *send, int *recv);
void mdrtMPIExchangeRecords(void *send, int *ns, int *os, void *recv,
                            int *nr, int *recvOffsets, int bytes);
void mdrtMPIPencilCommunicators(int px, int py, int gx, int gy, int rankBase, int logical);
void mdrtMPIPencilExchange(int group, void *send, const int *ns, const int *os,
                            void *recv, const int *nr, const int *offsets, int mixed);
int mdrtMPIAllTrue(int predicate);
int mdrtMPIActive(void);
void mdrtMPIGatherMetrics(const void *send, void *recv, int bytes);
int mdrtMPIMax(int value);
void mdrtMPIAbort(const char *reason);

#ifdef __cplusplus
}
#endif
#endif
