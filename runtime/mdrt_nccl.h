#ifndef MDRT_NCCL_H
#define MDRT_NCCL_H

#include <mpi.h>
#include <stdint.h>

/* All ranks enter with the same MPI communicator. Counts and offsets for
   the halo payload are in doubles. Device pointers may be null for an
   empty send or receive. */
void mdrtNCCLStart(MPI_Comm world);
void mdrtNCCLFinish(void);
void mdrtNCCLExchangeDevice(const double *send, const int *ns, const int *os,
                            double *recv, const int *nr, const int *or);
void mdrtNCCLExchangeHost(const double *send, const int *ns, const int *os,
                          double *recv, const int *nr, const int *or);
int mdrtNCCLNode(int rank);

#endif
