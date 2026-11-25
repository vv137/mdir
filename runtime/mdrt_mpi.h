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
void mdrtMPIBroadcast(double *values, int64_t count, int root);
void mdrtMPIExchange(double *send, int *sendCounts, int *sendOffsets,
                     double *recv, int *recvCounts, int *recvOffsets);
int mdrtMPIAllTrue(int predicate);
int mdrtMPIActive(void);
int mdrtMPIMax(int value);
void mdrtMPIAbort(const char *reason);

#ifdef __cplusplus
}
#endif
#endif
