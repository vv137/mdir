#include "mdrt_mpi.h"
#include "mdrt_dist_stats.h"
#include <limits.h>
#include <mpi.h>
#ifdef MDIR_ENABLE_NCCL
#include "mdrt_nccl.h"
static int useNCCL;
static int *domainOwner, localDomain;
#endif
#include <stdio.h>
#include <stdlib.h>

static MPI_Comm comm = MPI_COMM_NULL;
static MPI_Datatype wideWord = MPI_DATATYPE_NULL;
static int initializedHere, rank, size;

static void finish(void) {
  int done;
  MPI_Finalized(&done);
  if (!done) {
#ifdef MDIR_ENABLE_NCCL
    if (useNCCL) mdrtNCCLFinish();
    free(domainOwner);
#endif
    MPI_Type_free(&wideWord);
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
  MPI_Type_contiguous(2, MPI_UINT64_T, &wideWord);
  MPI_Type_commit(&wideWord);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  atexit(finish);
}

void mdrtMPIStartNCCL(void) {
  initialize();
#ifdef MDIR_ENABLE_NCCL
  mdrtNCCLStart(comm);
  useNCCL = 1;
#else
  mdrtMPIAbort("NCCL transport requires MDIR_ENABLE_NCCL");
#endif
}

int mdrtMPIRank(void) { initialize(); return rank; }
int mdrtMPISize(void) { initialize(); return size; }

/* Equal-size nodes take periodic rectangular blocks. The cost of a cut
   perpendicular to axis k is its full cross section times the number of
   cuts. The common factors (halo thickness, density and row size) cancel.
   If blocks do not tile, pack x-fast domain IDs in node order. */
void mdrtMPIPlaceDomains(const int64_t grid[3], const double edges[3]) {
#ifdef MDIR_ENABLE_NCCL
  if (!useNCCL) return;
  if (grid[0] * grid[1] * grid[2] != size)
    mdrtMPIAbort("NCCL requires one domain per rank");
  int *order = calloc(size, sizeof(*order));
  if (!domainOwner) domainOwner = calloc(size, sizeof(*domainOwner));
  if (!order || !domainOwner) mdrtMPIAbort("out of memory placing domains");
  int n = 0, uniform = 1, peers = 0;
  for (int leader = 0; leader < size; ++leader) {
    if (mdrtNCCLNode(leader) != leader) continue;
    int first = n;
    for (int r = leader; r < size; ++r)
      if (mdrtNCCLNode(r) == leader) order[n++] = r;
    if (!peers) peers = n - first;
    if (n - first != peers) uniform = 0;
  }
  int bx = 1, by = 1, bz = 1;
  double best = -1;
  for (int x = 1; uniform && x <= grid[0] && x <= peers; ++x) {
    if (grid[0] % x || peers % x) continue;
    for (int y = 1; y <= grid[1] && y <= peers/x; ++y) {
      if (grid[1] % y || peers/x % y) continue;
      int z = peers/x/y;
      if (grid[2] % z) continue;
      int block[] = {x, y, z};
      double cost = 0;
      for (int k = 0; k < 3; ++k)
        if (block[k] < grid[k])
          cost += (grid[k]/block[k]) * edges[(k+1)%3] * edges[(k+2)%3];
      if (best < 0 || cost < best) {
        bx = x; by = y; bz = z; best = cost;
      }
    }
  }
  int i = 0;
  for (int z = 0; z < grid[2]; z += bz)
    for (int y = 0; y < grid[1]; y += by)
      for (int x = 0; x < grid[0]; x += bx)
        for (int dz = 0; dz < bz; ++dz)
          for (int dy = 0; dy < by; ++dy)
            for (int dx = 0; dx < bx; ++dx) {
              int domain = x+dx + grid[0]*(y+dy + grid[1]*(z+dz));
              domainOwner[domain] = order[i++];
              if (domainOwner[domain] == rank) localDomain = domain;
            }
  free(order);
#endif
}

int mdrtMPINode(int peer) {
#ifdef MDIR_ENABLE_NCCL
  if (useNCCL) return mdrtNCCLNode(peer);
#endif
  return peer;
}

int64_t mdrtMPIDomainBound(int64_t domains, int64_t end) {
  initialize();
#ifdef MDIR_ENABLE_NCCL
  if (useNCCL && domainOwner) return localDomain + (end != 0);
#endif
  if (domains < size)
    mdrtMPIAbort("each rank needs at least one domain");
  int r = rank + (end != 0);
  int64_t q = domains / size, remainder = domains % size;
  return r * q + (r < remainder ? r : remainder);
}

int mdrtMPIOwner(int64_t domains, int64_t d) {
#ifdef MDIR_ENABLE_NCCL
  if (useNCCL && domainOwner) return domainOwner[d];
#endif
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

void mdrtMPIBroadcastIndices(int64_t *values, int64_t count, int root) {
  while (count > 0) {
    int n = count > INT_MAX ? INT_MAX : (int)count;
    arrival();
    MPI_Bcast(values,n,MPI_INT64_T,root,comm);
    values += n;
    count -= n;
  }
}

void mdrtMPIBroadcastWords(void *buffer, int64_t count, int root) {
  unsigned char *values = buffer;
  while (count > 0) {
    int n = count > INT_MAX ? INT_MAX : (int)count;
    arrival();
    MPI_Bcast(values, n, wideWord, root, comm);
    values += n;
    count -= n;
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
  MPI_Datatype type = words ? wideWord : MPI_DOUBLE;
  MPI_Alltoallv(send, sendCounts, sendOffsets, type,
                recv, recvCounts, recvOffsets, type, comm);
}
