#include "mdrt_mpi.h"
#include "mdrt_dist_stats.h"
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static MPI_Comm comm = MPI_COMM_NULL;
static int initializedHere, rank, size;

static void finish(void) {
  int done;
  MPI_Finalized(&done);
  if (!done) {
    MPI_Comm_free(&comm);
    if (initializedHere)
      MPI_Finalize();
  }
}

int mdrtMPIActive(void) { return comm != MPI_COMM_NULL; }

extern int mdrtDistObserving(void);
extern void mdrtDistPhase(int64_t phase);

// Time arrival at the payload collective with a rendezvous on the same
// communicator. The measurement includes the latency of the barrier.
static void arrival(void) {
  if (size > 1 && mdrtDistObserving()) {
    mdrtDistPhase(DistWait);
    MPI_Barrier(comm);
    mdrtDistPhase(-DistWait);
  }
}
void mdrtMPIGatherTimes(const double *send, double *recv, int count) {
  MPI_Allgather(send, count, MPI_DOUBLE, recv, count, MPI_DOUBLE, comm);
}

int mdrtMPIMax(int value) {
  int result;
  arrival();
  MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MAX, comm);
  return result;
}

void mdrtMPIAbort(const char *reason) {
  fprintf(stderr, "mdrt MPI: %s\n", reason);
  if (comm != MPI_COMM_NULL)
    MPI_Abort(comm, 1);
  abort();
}

static void initialize(void) {
  if (comm != MPI_COMM_NULL)
    return;
  int started, done;
  MPI_Finalized(&done);
  if (done)
    mdrtMPIAbort("cannot start a transport after MPI_Finalize");
  MPI_Initialized(&started);
  if (!started) {
    int provided;
    if (MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
      mdrtMPIAbort("MPI_Init_thread failed");
    if (provided < MPI_THREAD_FUNNELED)
      MPI_Abort(MPI_COMM_WORLD, 1);
    initializedHere = 1;
  }
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  atexit(finish);
}

int mdrtMPIRank(void) { initialize(); return rank; }
int mdrtMPISize(void) { initialize(); return size; }

int64_t mdrtMPIDomainBound(int64_t domains, int64_t end) {
  initialize();
  if (domains < size)
    mdrtMPIAbort("each rank needs at least one domain");
  int r = rank + (end != 0);
  int64_t q = domains / size, remainder = domains % size;
  return r * q + (r < remainder ? r : remainder);
}

int mdrtMPIOwner(int64_t domains, int64_t d) {
  int64_t q = domains / size, remainder = domains % size;
  int64_t split = remainder * (q + 1);
  return d < split ? d / (q + 1) : remainder + (d - split) / q;
}

// A field used exclusively by the checkpoint has a single consumer.
void mdrtMPIRowsToRoot(double *values, int64_t count, int owner) {
  if (!owner || (rank != owner && rank != 0)) return;
  while (count > 0) {
    int n = count > INT_MAX ? INT_MAX : (int)count;
    if (rank == owner)
      MPI_Send(values, n, MPI_DOUBLE, 0, 0, comm);
    else
      MPI_Recv(values, n, MPI_DOUBLE, owner, 0, comm, MPI_STATUS_IGNORE);
    count -= n;
    values += n;
  }
}

void mdrtMPIBroadcast(double *values, int64_t count, int root) {
  while (count > 0) {
    int n = count > INT_MAX ? INT_MAX : (int)count;
    arrival();
    MPI_Bcast(values, n, MPI_DOUBLE, root, comm);
    values += n;
    count -= n;
  }
}

int mdrtMPIAllTrue(int predicate) {
  int result;
  arrival();
  MPI_Allreduce(&predicate, &result, 1, MPI_INT, MPI_LAND, comm);
  return result;
}

void mdrtMPIExchange(void *send, int *sendCounts, int *sendOffsets,
                     void *recv, int *recvCounts, int *recvOffsets, int words) {
  arrival();
  MPI_Datatype type = words ? MPI_INT64_T : MPI_DOUBLE;
  MPI_Alltoallv(send, sendCounts, sendOffsets, type,
                recv, recvCounts, recvOffsets, type, comm);
}
