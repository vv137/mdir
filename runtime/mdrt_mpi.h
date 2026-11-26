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
void mdrtMPIRowsToRoot(double *values, int64_t count, int owner);
void mdrtMPIBroadcastIndices(int64_t *values, int64_t count, int root);
void mdrtMPIBroadcastWords(void *values, int64_t count, int root);
void mdrtMPIBroadcast(double *values, int64_t count, int root);
int mdrtMPISameNode(int peer);
void mdrtMPIForwardHalo(double *send, int *sendCounts, int *sendOffsets,
                        double *recv, int *recvCounts, int *recvOffsets);
void mdrtMPIShareBlocks(void *values, const int *counts, const int *offsets, int words);
void mdrtMPIExchange(void *send, int *sendCounts, int *sendOffsets,
                     void *recv, int *recvCounts, int *recvOffsets, int words);
void mdrtMPIExchangeCharges(int64_t *send, int *ns, int *os,
                            int64_t *recv, int *nr, int *offsets);
void mdrtMPIExchangeCounts(const int *send, int *recv);
void mdrtMPIExchangeRecords(void *send, int *ns, int *os, void *recv,
                            int *nr, int *recvOffsets, int bytes);
int mdrtMPIAllTrue(int predicate);
int mdrtMPIActive(void);
void mdrtMPIGatherMetrics(const void *send, void *recv, int bytes);
int mdrtMPIMax(int value);
void mdrtMPIAbort(const char *reason);

#ifdef __cplusplus
}
#endif
#endif
