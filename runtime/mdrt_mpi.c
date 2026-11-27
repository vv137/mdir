#include "mdrt_mpi.h"
#include "mdrt_dist_stats.h"
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static MPI_Comm comm = MPI_COMM_NULL;
static MPI_Comm nodeComm = MPI_COMM_NULL;
static MPI_Comm pencilComm[] = {MPI_COMM_NULL,MPI_COMM_NULL,MPI_COMM_NULL};
static int pencilPx, pencilPy, pencilGx, pencilGy, pencilRankBase;
static int nodeSize, *nodeRanks, *rankNodes;
static MPI_Datatype wideWord = MPI_DATATYPE_NULL;
static int initializedHere, rank, size;

static void freePencilCommunicators(void) {
  for (int i = 0; i < 3; ++i)
    if (pencilComm[i] != MPI_COMM_NULL) MPI_Comm_free(pencilComm+i);
}

static void finish(void) {
  int done;
  MPI_Finalized(&done);
  if (!done) {
    MPI_Type_free(&wideWord);
    free(nodeRanks);
    free(rankNodes);
    freePencilCommunicators();
    MPI_Comm_free(&nodeComm);
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
static void arrivalOn(MPI_Comm group, int participants) {
  if (participants > 1 && mdrtDistObserving()) {
    mdrtDistPhase(DistWait);
    MPI_Barrier(group);
    mdrtDistPhase(-DistWait);
  }
}
static void arrival(void) { arrivalOn(comm,size); }

void mdrtMPIGatherMetrics(const void *send, void *recv, int bytes) {
  MPI_Allgather(send, bytes, MPI_BYTE, recv, bytes, MPI_BYTE, comm);
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
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
  MPI_Comm_size(nodeComm, &nodeSize);
  nodeRanks = malloc(nodeSize*sizeof *nodeRanks);
  rankNodes = malloc(size*sizeof *rankNodes);
  if (!nodeRanks || !rankNodes) mdrtMPIAbort("out of memory indexing shared-memory ranks");
  MPI_Allgather(&rank, 1, MPI_INT, nodeRanks, 1, MPI_INT, nodeComm);
  MPI_Allgather(nodeRanks, 1, MPI_INT, rankNodes, 1, MPI_INT, comm);
  atexit(finish);
}

int mdrtMPIRank(void) { initialize(); return rank; }
int mdrtMPISize(void) { initialize(); return size; }
int mdrtMPINodeSize(void) { initialize(); return nodeSize; }

int64_t mdrtMPIDomainBound(int64_t domains, int64_t end) {
  initialize();
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

void mdrtMPIBroadcastIndices(int64_t *values, int64_t count, int root) {
  while (count > 0) {
    int n = count > INT_MAX ? INT_MAX : (int)count;
    arrival();
    MPI_Bcast(values, n, MPI_INT64_T, root, comm);
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

/* Deliver one observable component per rank in communicator order. */
void mdrtMPIGatherRankValue(const void *value, void *terms, int words) {
  MPI_Datatype type = words ? wideWord : MPI_DOUBLE;
  arrival();
  MPI_Allgather(value, 1, type, terms, 1, type, comm);
}

// Each rank has already filled its contiguous interval in domain order.
void mdrtMPIShareBlocks(void *values, const int *counts, const int *offsets, int words) {
  if (size == 1) return;
  arrival();
  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, values, counts, offsets,
                 words ? wideWord : MPI_DOUBLE, comm);
}

/* Logical pencil indices encode the two process-grid coordinates.
   A new layout has the same cache key on every transport rank. */
void mdrtMPIPencilCommunicators(int px, int py, int gx, int gy, int rankBase, int logical) {
  if (pencilPx == px && pencilPy == py && pencilGx == gx && pencilGy == gy &&
      pencilRankBase == rankBase) return;
  freePencilCommunicators();
  int active = logical >= 0;
  MPI_Comm_split(comm,active ? logical/py : MPI_UNDEFINED,active ? logical%py : 0,pencilComm);
  MPI_Comm_split(comm,active ? logical%py : MPI_UNDEFINED,active ? logical/py : 0,pencilComm+1);
  MPI_Comm_split(comm,active ? 0 : MPI_UNDEFINED,active ? logical : 0,pencilComm+2);
  pencilPx = px;
  pencilPy = py;
  pencilGx = gx;
  pencilGy = gy;
  pencilRankBase = rankBase;
}

void mdrtMPIPencilExchange(int group, void *send, const int *ns, const int *os,
                            void *recv, const int *nr, const int *offsets, int mixed) {
  MPI_Comm participants = pencilComm[group];
  MPI_Datatype type = mixed ? MPI_FLOAT : MPI_DOUBLE;
  int count = group == 0 ? pencilPy : (group == 1 ? pencilPx : pencilPx*pencilPy);
  arrivalOn(participants,count);
  MPI_Alltoallv(send,ns,os,type,recv,nr,offsets,type,participants);
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

void mdrtMPIExchangeFloats(float *send, int *ns, int *os,
                           float *recv, int *nr, int *offsets) {
  arrival();
  MPI_Alltoallv(send, ns, os, MPI_FLOAT, recv, nr, offsets, MPI_FLOAT, comm);
}

MPI_Request mdrtMPIStartChargeExchange(int64_t *send, int *ns, int *os,
                                       int64_t *recv, int *nr, int *offsets) {
  MPI_Request request;
  MPI_Ialltoallv(send, ns, os, MPI_INT64_T, recv, nr, offsets, MPI_INT64_T,
                 comm, &request);
  return request;
}

/* Ranks own consecutive domains. For a packed table one all-gather
   places each live subtotal in its original row without doing arithmetic. */
static int shareDomainBlocks(void *values, int64_t domains, int64_t width,
                             MPI_Datatype type) {
  if (size == 1 || !width) return 1;
  if (domains > INT_MAX / width) return 0;
  int64_t q = domains / size, rem = domains % size;
  int *counts = malloc(2*(size_t)size*sizeof *counts);
  if (!counts) mdrtMPIAbort("out of memory sharing domain rows");
  int *offsets = counts + size;
  for (int p = 0; p < size; ++p) {
    offsets[p] = (p*q + (p < rem ? p : rem)) * width;
    counts[p] = (q + (p < rem)) * width;
  }
  arrival();
  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, values, counts, offsets, type, comm);
  free(counts);
  return 1;
}

int mdrtMPIShareDomainPartials(void *values, int64_t domains, int64_t width, int words) {
  return shareDomainBlocks(values, domains, width, words ? wideWord : MPI_DOUBLE);
}

void mdrtMPICollectIntervals(void *data, int64_t count, int64_t width, int integer, int rootOnly) {
  if (size == 1 || count == 0) return;
  MPI_Datatype type = integer ? MPI_INT32_T : MPI_DOUBLE;
  size_t bytes = integer ? sizeof(int32_t) : sizeof(double);
  int64_t q = count/size, rem = count%size;
  if (count <= INT_MAX/width) {
    if (!rootOnly) { shareDomainBlocks(data,count,width,type); return; }
    int *sizes = malloc(2*(size_t)size*sizeof *sizes);
    if (!sizes) mdrtMPIAbort("out of memory collecting native fields");
    int *offsets = sizes+size;
    for (int p = 0; p < size; ++p) {
      sizes[p] = (q+(p < rem))*width;
      offsets[p] = (p*q+(p < rem ? p : rem))*width;
    }
    void *local = (char *)data+offsets[rank]*bytes;
    MPI_Gatherv(rank ? local : MPI_IN_PLACE,sizes[rank],type,data,sizes,offsets,type,0,comm);
    free(sizes);
    return;
  }
  for (int p = 0; p < size; ++p) {
    int64_t off = (p*q+(p < rem ? p : rem))*width, n = (q+(p < rem))*width;
    for (int64_t i = 0; i < n; ) {
      int block = n-i > INT_MAX ? INT_MAX : (int)(n-i);
      void *buf = (char *)data+(off+i)*bytes;
      if (!rootOnly) MPI_Bcast(buf,block,type,p,comm);
      else if (p && rank == p) MPI_Send(buf,block,type,0,0,comm);
      else if (p && rank == 0) MPI_Recv(buf,block,type,p,0,comm,MPI_STATUS_IGNORE);
      i += block;
    }
  }
}

void mdrtMPIShareIndexRows(int64_t *rows, int64_t domains, int64_t width) {
  if (!shareDomainBlocks(rows,domains,width,MPI_INT64_T))
    for (int64_t d = 0; d < domains; ++d)
      mdrtMPIBroadcastIndices(rows+d*width,width,mdrtMPIOwner(domains,d));
}

void mdrtMPIShareDomainCounts(int64_t *rows, int64_t domains) {
  if (shareDomainBlocks(rows, domains, 1, MPI_INT64_T)) return;
  int64_t q = domains / size, rem = domains % size;
  for (int p = 0; p < size; ++p)
    mdrtMPIBroadcastIndices(rows + p*q + (p < rem ? p : rem), q + (p < rem), p);
}

int mdrtMPISameNode(int peer) { return rankNodes[peer] == rankNodes[rank]; }

/* A halo invocation owns its request set until all peers finish. Only
   nonempty streams participate; forward and transpose use separate tags. */
struct HaloRequests {
  int count;
  MPI_Request requests[];
};

struct HaloRequests *mdrtMPIStartHalo(void *send, const int *ns, const int *os,
    void *recv, const int *nr, const int *offsets, int words, int transpose) {
  struct HaloRequests *pending = malloc(sizeof *pending + 2*(size_t)size*sizeof(MPI_Request));
  if (!pending) mdrtMPIAbort("out of memory posting halo exchange");
  pending->count = 0;
  MPI_Datatype type = words ? wideWord : MPI_DOUBLE;
  size_t width = words ? 2*sizeof(uint64_t) : sizeof(double);
  for (int p = 0; p < size; ++p)
    if (nr[p])
      MPI_Irecv((char *)recv + (size_t)offsets[p]*width, nr[p], type, p, transpose,
                comm, &pending->requests[pending->count++]);
  for (int p = 0; p < size; ++p)
    if (ns[p])
      MPI_Isend((char *)send + (size_t)os[p]*width, ns[p], type, p, transpose,
                comm, &pending->requests[pending->count++]);
  return pending;
}

void mdrtMPIWaitHalo(struct HaloRequests *pending) {
  MPI_Waitall(pending->count, pending->requests, MPI_STATUSES_IGNORE);
  free(pending);
}

void mdrtMPISumMigration(int64_t *counts, int64_t domains) {
  for (int64_t left = 2*domains; left > 0; ) {
    int n = left > INT_MAX ? INT_MAX : (int)left;
    MPI_Allreduce(MPI_IN_PLACE,counts,n,MPI_INT64_T,MPI_SUM,comm);
    counts += n;
    left -= n;
  }
}

void mdrtMPIExchangeCounts(const int *send, int *recv) {
  arrival();
  MPI_Alltoall(send, 1, MPI_INT, recv, 1, MPI_INT, comm);
}

void mdrtMPIExchangeRecords(void *send, int *ns, int *os, void *recv,
                            int *nr, int *or, int bytes) {
  MPI_Datatype record;
  MPI_Type_contiguous(bytes, MPI_BYTE, &record);
  MPI_Type_commit(&record);
  arrival();
  MPI_Alltoallv(send, ns, os, record, recv, nr, or, record, comm);
  MPI_Type_free(&record);
}
