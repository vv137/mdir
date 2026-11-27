#include "mdrt_nccl.h"
#include "mdrt_mpi.h"
#include "mdrt_dist_stats.h"
#include <limits.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ncclComm_t peers;
static cudaStream_t stream;
static cudaEvent_t tupleReady;
static cudaMemPool_t rowPool;
static int size;
static int *nodes;
static double *outgoing, *incoming, *ownedField, *ghostField;
static size_t outCapacity, inCapacity, ownedCapacity, ghostCapacity;

static void checkCuda(cudaError_t status) {
  if (status != cudaSuccess) mdrtMPIAbort(cudaGetErrorString(status));
}

static void checkNCCL(ncclResult_t status) {
  if (status != ncclSuccess) mdrtMPIAbort(ncclGetErrorString(status));
}

void mdrtNCCLStart(MPI_Comm world) {
  if (peers) return;
  MPI_Comm local;
  int rank, localRank, localSize, visible;
  MPI_Comm_rank(world, &rank);
  MPI_Comm_size(world, &size);
  MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
  MPI_Comm_rank(local, &localRank);
  MPI_Comm_size(local, &localSize);
  checkCuda(cudaGetDeviceCount(&visible));
  int ordinal = visible == 1 ? 0 : localRank;
  checkCuda(cudaSetDevice(ordinal));
  int pools = 0;
  checkCuda(cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, ordinal));
  if (pools) {
    struct cudaMemPoolProps props = {0};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = ordinal;
    checkCuda(cudaMemPoolCreate(&rowPool, &props));
    uint64_t retained = 256ULL << 20;
    checkCuda(cudaMemPoolSetAttribute(rowPool, cudaMemPoolAttrReleaseThreshold, &retained));
  }
  struct cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, ordinal));
  cudaUUID_t *ids = calloc(localSize, sizeof(*ids));
  nodes = calloc(size, sizeof(*nodes));
  if (!ids || !nodes) mdrtMPIAbort("out of memory mapping NCCL peers");
  MPI_Allgather(&prop.uuid, sizeof(prop.uuid), MPI_BYTE, ids, sizeof(*ids), MPI_BYTE, local);
  for (int i = 0; i < localRank; ++i)
    if (memcmp(&ids[i], &prop.uuid, sizeof(prop.uuid)) == 0)
      mdrtMPIAbort("two local ranks selected the same CUDA device");
  free(ids);
  char value[32];
  snprintf(value, sizeof value, "%d", ordinal);
  if (setenv("MDRT_DEVICE", value, 1))
    mdrtMPIAbort("cannot select the rank's device for MDIR");
  int leader = rank;
  MPI_Bcast(&leader, 1, MPI_INT, 0, local);
  MPI_Allgather(&leader, 1, MPI_INT, nodes, 1, MPI_INT, world);
  MPI_Comm_free(&local);
  ncclUniqueId id;
  if (rank == 0) checkNCCL(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, world);
  checkNCCL(ncclCommInitRank(&peers, size, id, rank));
  checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}

int mdrtNCCLNode(int rank) { return nodes[rank]; }

int mdrtNCCLActive(void) { return peers != NULL; }

struct NCCLPlan {
  int *counts, *ns, *nr, *os, *offsetRecv;
  int *source, *dest;
  double *shifts;
  int sendRows, recvRows;
  int hasPeers[2], pooled;
  int *centerMasks;
  int *centerSlots, *pairPlaces, *exported;
  int interiorRows, boundaryRows, placesReady;
  int64_t ownedRows;
  double *interiorPositions;
  cudaEvent_t producerReady, haloReady;
};

static void *upload(const void *host, size_t bytes) {
  void *device = NULL;
  if (bytes) {
    checkCuda(cudaMalloc(&device, bytes));
    checkCuda(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
  }
  return device;
}

struct NCCLPlan *mdrtNCCLPlan(int64_t ownedRows, int64_t domains, const int64_t *ghosts,
    int64_t capacity, const int64_t *offsets, const int64_t *sourceDomain, const int64_t *sourceRow,
    const int64_t *image, const double *edges) {
  int rank = mdrtMPIRank();
  struct NCCLPlan *p = calloc(1, sizeof *p);
  if (!p || !(p->counts = calloc(6 * (size_t)size, sizeof(int))))
    mdrtMPIAbort("out of memory building halo plan");
  p->ns = p->counts; p->nr = p->ns + size;
  p->os = p->nr + size; p->offsetRecv = p->os + size;
  int *ws = p->offsetRecv + size, *wr = ws + size;
  int *sources = NULL, *targets = NULL;
  double *shifts = NULL;
  for (int pass = 0; pass < 2; ++pass) {
    for (int64_t d = 0; d < domains; ++d) {
      int to = mdrtMPIOwner(domains, d);
      for (int64_t g = 0; g < ghosts[d]; ++g) {
        int64_t i = (offsets ? offsets[d] : d*capacity) + g;
        int from = mdrtMPIOwner(domains, sourceDomain[i]);
        if (!pass) {
          if (from == rank) {
            if (p->ns[to] == INT_MAX) mdrtMPIAbort("too many send rows");
            ++p->ns[to];
          }
          if (to == rank) {
            if (p->nr[from] == INT_MAX) mdrtMPIAbort("too many receive rows");
            ++p->nr[from];
          }
        } else {
          if (from == rank) {
            if (sourceRow[i] > INT_MAX) mdrtMPIAbort("too many owned rows");
            sources[ws[to]++] = sourceRow[i];
          }
          if (to == rank) {
            int r = wr[from]++;
            targets[r] = g;
            for (int k = 0; k < 3; ++k)
              shifts[3LL*r+k] = image[3*i+k]*edges[k];
          }
        }
      }
    }
    if (!pass) {
      for (int peer = 0; peer < size; ++peer) {
        if (p->ns[peer] > INT_MAX - p->sendRows || p->nr[peer] > INT_MAX - p->recvRows)
          mdrtMPIAbort("halo row displacement exceeds int");
        if (p->ns[peer] || p->nr[peer])
          p->hasPeers[nodes[rank] != nodes[peer]] = 1;
        p->os[peer] = ws[peer] = p->sendRows;
        p->offsetRecv[peer] = wr[peer] = p->recvRows;
        p->sendRows += p->ns[peer]; p->recvRows += p->nr[peer];
      }
      sources = calloc(1u+(size_t)p->sendRows, sizeof(int));
      targets = calloc(1u+(size_t)p->recvRows, sizeof(int));
      shifts = calloc(1u+3*(size_t)p->recvRows, sizeof(double));
      if (!sources || !targets || !shifts) mdrtMPIAbort("out of memory mapping halo rows");
    }
  }
  p->source = upload(sources, (size_t)p->sendRows*sizeof(int));
  p->dest = upload(targets, (size_t)p->recvRows*sizeof(int));
  p->shifts = upload(shifts, (size_t)p->recvRows*3*sizeof(double));
  if (size > 1) {
    p->ownedRows = ownedRows;
    p->exported = sources;
    sources = NULL;
  }
  free(sources); free(targets); free(shifts);
  return p;
}

extern void mdrtNCCLUpdateFaceState(int64_t *, const int64_t *, int, int);
extern void mdrtNCCLFaceColumns(const int64_t *, int, int, const unsigned *,
    const void *, int, int *, int *, int *, unsigned *, int *, int *, double *,
    const int64_t *, const double *);

int64_t *mdrtNCCLFaceState(const int64_t *previous, const int64_t *owners,
    int count, const int64_t *records, int changes) {
  size_t bytes = (size_t)count*sizeof(*owners);
  int64_t *state = (int64_t *)mdrtNCCLAllocate(bytes);
  if (previous) {
    if (bytes) checkCuda(cudaMemcpyAsync(state,previous,bytes,cudaMemcpyDeviceToDevice,0));
  } else {
    mdrtNCCLUpload((double *)state,(const double *)owners,bytes);
    mdrtNCCLUpdateFaceState(state,NULL,count,1);
  }
  if (changes) {
    size_t deltaBytes = 2*(size_t)changes*sizeof(*records);
    double *delta = mdrtNCCLAllocate(deltaBytes);
    mdrtNCCLUpload(delta,(const double *)records,deltaBytes);
    mdrtNCCLUpdateFaceState(state,(const int64_t *)delta,changes,0);
    mdrtNCCLFreePositions(delta);
  }
  return state;
}

struct NCCLPlan *mdrtNCCLFacePlan(int local, int count, int64_t ownedRows,
    const int64_t *state, const unsigned *incoming, const unsigned *outgoingMasks,
    const int64_t *send, const int64_t *receive, unsigned *previous,
    const int64_t *grid, const double *edges) {
  struct NCCLPlan *p = calloc(1,sizeof(*p));
  if (!p || !(p->counts = calloc(4*(size_t)size,sizeof(int))))
    mdrtMPIAbort("out of memory indexing device faces");
  p->pooled = 1;
  p->ownedRows = ownedRows;
  p->ns = p->counts; p->nr = p->ns+size;
  p->os = p->nr+size; p->offsetRecv = p->os+size;
  struct Column { int kind, domain, offset; unsigned mask; };
  struct Column *columns = calloc(2+2*(size_t)size,sizeof(*columns));
  if (!columns) mdrtMPIAbort("out of memory indexing device face columns");
  columns[1].kind = 1;
  int nc = 2;
  // Physical rank order determines the start of each contiguous message.
  for (int d=0; d<size; ++d) {
    int peer = mdrtMPIOwner(size,d);
    if (send[d]>INT_MAX || receive[d]>INT_MAX) mdrtMPIAbort("too many face rows");
    p->ns[peer]=send[d]; p->nr[peer]=receive[d];
  }
  for (int peer=0; peer<size; ++peer) {
    p->os[peer]=p->sendRows; p->offsetRecv[peer]=p->recvRows;
    if (p->ns[peer]>INT_MAX-p->sendRows || p->nr[peer]>INT_MAX-p->recvRows)
      mdrtMPIAbort("face rows exceed device index range");
    p->sendRows+=p->ns[peer]; p->recvRows+=p->nr[peer];
    if (p->ns[peer] || p->nr[peer]) p->hasPeers[nodes[peer]!=nodes[mdrtMPIRank()]]=1;
  }
  for (int d=0; d<size; ++d) {
    int peer = mdrtMPIOwner(size,d);
    if (send[d]) columns[nc++] = (struct Column){2,local,p->os[peer],outgoingMasks[d]};
    if (receive[d]) columns[nc++] = (struct Column){3,d,p->offsetRecv[peer],incoming[d]};
  }
  p->source = (int *)mdrtNCCLAllocate((size_t)p->sendRows*sizeof(int));
  p->dest = (int *)mdrtNCCLAllocate((size_t)p->recvRows*sizeof(int));
  p->shifts = mdrtNCCLAllocate(3*(size_t)p->recvRows*sizeof(double));
  size_t blockBytes = nc*((count+255LL)/256)*sizeof(int);
  double *blocks = mdrtNCCLAllocate(blockBytes);
  double *rows = mdrtNCCLAllocate(2*(size_t)count*sizeof(int));
  double *masks = mdrtNCCLAllocate(size*sizeof(*incoming));
  double *deviceColumns = mdrtNCCLAllocate(nc*sizeof(*columns));
  mdrtNCCLUpload(masks,(const double *)incoming,size*sizeof(*incoming));
  mdrtNCCLUpload(deviceColumns,(const double *)columns,nc*sizeof(*columns));
  mdrtNCCLFaceColumns(state,count,local,(unsigned *)masks,deviceColumns,nc,
      (int *)blocks,(int *)rows,(int *)rows+count,previous,p->source,p->dest,
      p->shifts,grid,edges);
  mdrtNCCLFreePositions(blocks);
  mdrtNCCLFreePositions(rows);
  mdrtNCCLFreePositions(masks);
  mdrtNCCLFreePositions(deviceColumns);
  free(columns);
  return p;
}

// Plans used only by ordinary exchanges do not allocate split-row storage.
static void preparePairRows(struct NCCLPlan *p, const uint32_t *faces, const int64_t *ids) {
  if (p->producerReady) return;
  int64_t ownedRows = p->ownedRows;
  size_t rows = ownedRows + p->recvRows;
  int *masks = calloc(3 * (rows ? rows : 1), sizeof *masks);
  if (!masks) mdrtMPIAbort("out of memory classifying halo centers");
  for (int64_t r = 0; r < ownedRows; ++r)
    masks[r] = masks[2*rows+r] = 1;
  if (faces) {
    for (int64_t r = 0; r < ownedRows; ++r) {
      masks[r] = !faces[ids[r]];
      masks[rows+r] = !masks[r];
    }
  } else {
    for (int r = 0; r < p->sendRows; ++r) {
      masks[p->exported[r]] = 0;
      masks[rows+p->exported[r]] = 1;
    }
  }
  p->centerMasks = upload(masks, 3*(rows ? rows : 1)*sizeof(int));
  // Each owned row has one slot in the two compact launch lists. Reserve
  // place zero for an empty phase (its center mask suppresses the row).
  int *slots = calloc(ownedRows + 1, sizeof(*slots));
  if (!slots) mdrtMPIAbort("out of memory numbering pair centers");
  for (int64_t r = 0; r < ownedRows; ++r)
    p->interiorRows += masks[r];
  int interior = 1, boundary = 1 + p->interiorRows;
  for (int64_t r = 0; r < ownedRows; ++r)
    slots[r] = masks[r] ? interior++ : boundary++;
  p->boundaryRows = ownedRows - p->interiorRows;
  p->centerSlots = upload(slots, (ownedRows + 1) * sizeof(*slots));
  free(slots);
  int *places = calloc(1 + ownedRows + rows, sizeof(*places));
  if (!places) mdrtMPIAbort("out of memory allocating pair row lists");
  for (size_t r = 0; r < rows; ++r)
    places[1 + ownedRows + r] = r;
  p->pairPlaces = upload(places, (1 + ownedRows + rows) * sizeof(*places));
  free(places);
  free(masks);
  checkCuda(cudaEventCreateWithFlags(&p->producerReady, cudaEventDisableTiming));
  checkCuda(cudaEventCreateWithFlags(&p->haloReady, cudaEventDisableTiming));
  free(p->exported);
  p->exported = NULL;
}

extern void mdrtNCCLSelfFaces(const double *, long long, const double *, double,
    unsigned *, int *, int *, int *, double *);
extern void mdrtNCCLSelfGather(const double *, const int *, const double *,
    double *, long long, int, CUstream);

struct NCCLPlan *mdrtNCCLSelfPlan(double *x, int64_t rows, const double *edges,
                                  double reach, int64_t *ghosts) {
  struct NCCLPlan *p=calloc(1,sizeof *p);
  if (!p || size!=1 || rows>INT_MAX) mdrtMPIAbort("invalid local face plan");
  p->counts=calloc(4,sizeof *p->counts);
  if (!p->counts) mdrtMPIAbort("out of memory counting local faces");
  p->ns=p->counts; p->nr=p->counts+1; p->os=p->counts+2; p->offsetRecv=p->counts+3;
  p->pooled=1;
  int64_t blocks=(rows+255)/256;
  int *prefix=calloc(blocks ? blocks : 1,sizeof *prefix);
  if (!prefix) mdrtMPIAbort("out of memory indexing local faces");
  unsigned *masks=(unsigned *)mdrtNCCLAllocate(rows*sizeof *masks);
  int *deviceBlocks=(int *)mdrtNCCLAllocate(blocks*sizeof *deviceBlocks);
  mdrtNCCLSelfFaces(x,rows,edges,reach,masks,deviceBlocks,NULL,NULL,NULL);
  if (blocks) checkCuda(cudaMemcpy(prefix,deviceBlocks,blocks*sizeof *prefix,cudaMemcpyDeviceToHost));
  int total=0;
  for (int64_t b=0; b<blocks; ++b) {
    int count=prefix[b];
    if (count>INT_MAX-total) mdrtMPIAbort("too many local face rows");
    prefix[b]=total;
    total+=count;
  }
  if (blocks) checkCuda(cudaMemcpy(deviceBlocks,prefix,blocks*sizeof *prefix,cudaMemcpyHostToDevice));
  free(prefix);
  p->sendRows=p->recvRows=*p->ns=*p->nr=total;
  p->hasPeers[0]=total!=0;
  p->source=(int *)mdrtNCCLAllocate((size_t)total*sizeof *p->source);
  p->dest=(int *)mdrtNCCLAllocate((size_t)total*sizeof *p->dest);
  p->shifts=mdrtNCCLAllocate(3*(size_t)total*sizeof *p->shifts);
  mdrtNCCLSelfFaces(x,rows,edges,reach,masks,deviceBlocks,p->source,p->dest,p->shifts);
  mdrtNCCLFreePositions((double *)masks);
  mdrtNCCLFreePositions((double *)deviceBlocks);
  *ghosts=total;
  return p;
}

void mdrtNCCLFreePlan(struct NCCLPlan *p) {
  if (!p) return;
  if (p->pooled) {
    mdrtNCCLFreePositions((double *)p->source);
    mdrtNCCLFreePositions((double *)p->dest);
    mdrtNCCLFreePositions(p->shifts);
  } else {
    checkCuda(cudaFree(p->source));
    checkCuda(cudaFree(p->dest));
    checkCuda(cudaFree(p->shifts));
  }
  if (p->producerReady) {
    checkCuda(cudaEventDestroy(p->producerReady));
    checkCuda(cudaEventDestroy(p->haloReady));
    checkCuda(cudaFree(p->centerMasks));
    checkCuda(cudaFree(p->centerSlots));
    checkCuda(cudaFree(p->pairPlaces));
    if (p->interiorPositions) mdrtNCCLFreePositions(p->interiorPositions);
  }
  free(p->exported);
  free(p->counts);
  free(p);
}

static void reserve(double **buffer, size_t *capacity, size_t bytes) {
  if (bytes <= *capacity) return;
  checkCuda(cudaFree(*buffer));
  checkCuda(cudaMalloc((void **)buffer, bytes));
  *capacity = bytes;
}

extern void mdrtDistPhase(int64_t phase);
extern void mdrtNCCLRows(int gather, const double *input, const int *rows,
    double *output, long long n, int k, const double *shift, int transpose, CUstream stream);

static void exchange(const struct NCCLPlan *p, double *owned, int64_t nOwned,
    double *ghosts, int k, int image, int reverse, double *packed, int asynchronous, int resident) {
  if (image && k != 3) mdrtMPIAbort("imaged halo positions require three components");
  size_t row = (size_t)k * sizeof(double);
  if (packed && size==1 && !reverse) {
    if (nOwned) checkCuda(cudaMemcpyAsync(packed,owned,nOwned*row,cudaMemcpyDeviceToDevice,stream));
    mdrtNCCLSelfGather(owned,p->source,image ? p->shifts : NULL,packed+nOwned*k,
                       (long long)p->recvRows*k,k,stream);
    if (!asynchronous) checkCuda(cudaStreamSynchronize(stream));
    return;
  }
  if (!packed && !resident) {
    reserve(&ownedField, &ownedCapacity, nOwned*row);
    reserve(&ghostField, &ghostCapacity, p->recvRows*row);
  }
  double *ownedDevice = packed ? packed : resident ? owned : ownedField;
  double *ghostDevice = packed ? packed + nOwned*k : resident ? ghosts : ghostField;
  int send = reverse ? p->recvRows : p->sendRows;
  int recv = reverse ? p->sendRows : p->recvRows;
  reserve(&outgoing, &outCapacity, send*row);
  reserve(&incoming, &inCapacity, recv*row);
  if (!resident && nOwned) checkCuda(cudaMemcpyAsync(ownedDevice, owned, nOwned*row, packed ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, stream));
  if (!resident && reverse && p->recvRows)
    checkCuda(cudaMemcpyAsync(ghostDevice, ghosts, p->recvRows*row, cudaMemcpyHostToDevice, stream));
  mdrtNCCLRows(1, reverse ? ghostDevice : ownedDevice, reverse ? p->dest : p->source,
               outgoing, (long long)send*k, k, NULL, reverse, stream);
  // Packing, peer transfers, and unpacking are ordered on this stream.
  int rank = mdrtMPIRank();
  // An outer group lets NCCL schedule both link classes together. The
  // per-peer slices are disjoint. The forward clock includes their joint
  // launch, while the link clocks count the corresponding enqueue calls.
  int joint = p->hasPeers[0] && p->hasPeers[1];
  if (joint) checkNCCL(ncclGroupStart());
  for (int inter = 0; inter <= 1; ++inter) {
    if (!p->hasPeers[inter]) continue;
    int phase = inter ? DistInterHalo : DistIntraHalo;
    if (!reverse) mdrtDistPhase(phase);
    // A periodic image on the same device uses the same packed row
    // order in both buffers. Only remote peers need the communicator.
    if (!inter && p->ns[rank]) {
      size_t src = (size_t)(reverse ? p->offsetRecv[rank] : p->os[rank])*k;
      size_t dst = (size_t)(reverse ? p->os[rank] : p->offsetRecv[rank])*k;
      checkCuda(cudaMemcpyAsync(incoming+dst,outgoing+src,
          (size_t)p->ns[rank]*row,cudaMemcpyDeviceToDevice,stream));
    }
    int grouped = 0;
    for (int peer = 0; peer < size; ++peer) {
      if (peer == rank || (nodes[peer] != nodes[rank]) != inter) continue;
      size_t ns = (size_t)(reverse ? p->nr[peer] : p->ns[peer])*k;
      size_t nr = (size_t)(reverse ? p->ns[peer] : p->nr[peer])*k;
      if (!ns && !nr) continue;
      if (!grouped) {
        checkNCCL(ncclGroupStart());
        grouped = 1;
      }
      if (ns) {
        size_t offset = (size_t)(reverse ? p->offsetRecv[peer] : p->os[peer])*k;
        checkNCCL(ncclSend(outgoing+offset, ns, ncclDouble, peer, peers, stream));
      }
      if (nr) {
        size_t offset = (size_t)(reverse ? p->os[peer] : p->offsetRecv[peer])*k;
        checkNCCL(ncclRecv(incoming+offset, nr, ncclDouble, peer, peers, stream));
      }
    }
    if (grouped) checkNCCL(ncclGroupEnd());
    if (!reverse) mdrtDistPhase(-phase);
  }
  if (joint) checkNCCL(ncclGroupEnd());
  double *result = reverse ? ownedDevice : ghostDevice;
  mdrtNCCLRows(0, incoming, reverse ? p->source : p->dest, result,
               (long long)recv*k, k, image ? p->shifts : NULL, reverse, stream);
  size_t bytes = (reverse ? nOwned : p->recvRows)*row;
  if (!packed && !resident && bytes) checkCuda(cudaMemcpyAsync(reverse ? owned : ghosts, result, bytes,
                                       cudaMemcpyDeviceToHost, stream));
  if (!asynchronous) checkCuda(cudaStreamSynchronize(stream));
}

void mdrtNCCLResidentHalo(const struct NCCLPlan *plan,double *owned,int64_t rows,
                          double *ghosts,int k,int image,int reverse) {
  mdrtNCCLWaitForRows(stream);
  exchange(plan,owned,rows,ghosts,k,image,reverse,NULL,1,1);
  checkCuda(cudaEventRecord(tupleReady,stream));
  checkCuda(cudaStreamWaitEvent(0,tupleReady,0));
}

void mdrtNCCLHalo(const struct NCCLPlan *plan, double *owned, int64_t rows,
    double *ghosts, int k, int image, int reverse) {
  exchange(plan, owned, rows, ghosts, k, image, reverse, NULL, 0, 0);
}

/* The outlined MATRIX borrows this immutable buffer until its block ends.
   Owned rows precede ghosts in the order of the neighborhood. */
double *mdrtNCCLPackedRows(const struct NCCLPlan *plan, double *owned, int64_t rows, int k, int image) {
  double *positions;
  size_t count = rows + (size_t)plan->recvRows;
  positions = mdrtNCCLAllocate((count ? count : 1)*k*sizeof(double));
  // The halo stream consumes the default stream's integration result.
  mdrtNCCLWaitForRows(stream);
  exchange(plan, owned, rows, NULL, k, image, 0, positions, 0, 0);
  return positions;
}

/* An epoch's export rows include every center within the halo reach of a
   remote or periodic face. An interior center cannot acquire a ghost
   neighbor before the half-skin displacement test expires. */
double *mdrtNCCLBeginPositions(struct NCCLPlan *p, double *owned, int64_t n,
                               const uint32_t *faces, const int64_t *ids) {
  preparePairRows(p,faces,ids);
  size_t bytes = (size_t)(n+p->recvRows)*3*sizeof(double);
  double *positions = mdrtNCCLAllocate(bytes);
  checkCuda(cudaEventRecord(p->producerReady, 0));
  checkCuda(cudaStreamWaitEvent(stream, p->producerReady, 0));
  exchange(p, owned, n, NULL, 3, 1, 0, positions, 1, 0);
  checkCuda(cudaEventRecord(p->haloReady, stream));
  return positions;
}

extern void mdrtNCCLIndexPlaces(const int *, const int *, int *, int64_t,
                                int64_t, void *);

int *mdrtNCCLPairPlaces(struct NCCLPlan *p, const int *order, int64_t rows,
                        int phase, int cached, void *compute, int64_t *count) {
  if (cached && !p->placesReady) {
    mdrtNCCLIndexPlaces(order, p->centerSlots, p->pairPlaces, rows,
                        p->ownedRows, compute);
    p->placesReady = 1;
  }
  if (phase == 2) {
    *count = rows;
    return p->pairPlaces + 1 + p->ownedRows;
  }
  if (!cached || !p->placesReady) mdrtMPIAbort("missing matrix for split pair rows");
  *count = phase ? p->boundaryRows : p->interiorRows;
  if (!*count) {
    *count = rows ? 1 : 0;
    return p->pairPlaces;
  }
  return p->pairPlaces + 1 + (phase ? p->interiorRows : 0);
}

int *mdrtNCCLCenterMask(struct NCCLPlan *p, int64_t rows, int phase) {
  return p->centerMasks + rows*phase;
}

double *mdrtNCCLPairPositions(struct NCCLPlan *p, double *full, double *owned,
                              int64_t n, int phase, void *compute) {
  if (phase != 0) {
    checkCuda(cudaStreamWaitEvent(compute, p->haloReady, 0));
    return full;
  }
  if (!p->interiorPositions) {
    p->interiorPositions = mdrtNCCLAllocate((n+p->recvRows)*3*sizeof(double));
    if (p->recvRows)
      checkCuda(cudaMemsetAsync(p->interiorPositions+n*3, 0,
                                p->recvRows*3*sizeof(double), 0));
  }
  checkCuda(cudaEventRecord(p->producerReady, 0));
  checkCuda(cudaStreamWaitEvent(compute, p->producerReady, 0));
  if (n) checkCuda(cudaMemcpyAsync(p->interiorPositions, owned,
                    n*3*sizeof(double), cudaMemcpyDeviceToDevice, compute));
  return p->interiorPositions;
}

void mdrtNCCLFreePositions(double *positions) {
  if (rowPool) checkCuda(cudaFreeAsync(positions, 0));
  else checkCuda(cudaFree(positions));
}

double *mdrtNCCLAllocate(size_t bytes) {
  double *device;
  size_t size = bytes ? bytes : sizeof(double);
  if (rowPool) checkCuda(cudaMallocFromPoolAsync((void **)&device,size,rowPool,0));
  else checkCuda(cudaMalloc((void **)&device,size));
  return device;
}

/* The row table is immutable for the lifetime of the halo. Each consumer
   stream owns its copy, including the allocation and final free. */
struct PairIds {
  struct PairIds *next;
  cudaStream_t stream;
  int64_t *rows;
};
int64_t *mdrtNCCLPairIds(struct PairIds **head, void *stream, const int64_t *owned,
                         int64_t n, const int64_t *ghost, int64_t g) {
  struct PairIds *p = *head;
  for (; p; p = p->next)
    if (p->stream == stream) return p->rows;
  p = calloc(1, sizeof(*p));
  if (!p) abort();
  p->stream = stream;
  size_t count = (size_t)n + g;
  size_t bytes = (count ? count : 1) * sizeof(*p->rows);
  if (rowPool)
    checkCuda(cudaMallocFromPoolAsync((void **)&p->rows, bytes, rowPool, stream));
  else
    checkCuda(cudaMalloc((void **)&p->rows, bytes));
  if (n)
    checkCuda(cudaMemcpyAsync(p->rows, owned, n * sizeof(*owned),
                              cudaMemcpyHostToDevice, stream));
  if (g)
    checkCuda(cudaMemcpyAsync(p->rows + n, ghost, g * sizeof(*ghost),
                              cudaMemcpyHostToDevice, stream));
  p->next = *head;
  *head = p;
  return p->rows;
}

void mdrtNCCLReleasePairIds(struct PairIds *p) {
  while (p) {
    struct PairIds *next = p->next;
    if (rowPool) checkCuda(cudaFreeAsync(p->rows, p->stream));
    else checkCuda(cudaFree(p->rows));
    free(p);
    p = next;
  }
}

void mdrtNCCLUpload(double *device, const double *host, size_t bytes) {
  if (bytes) checkCuda(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
}

void mdrtNCCLDownload(double *host, const double *device, size_t bytes) {
  if (bytes) checkCuda(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
}

void mdrtNCCLImport(double *dest, const double *device, size_t bytes) {
  // The ordinary pair runtime owns the producer stream and temporary.
  checkCuda(cudaDeviceSynchronize());
  if (bytes) checkCuda(cudaMemcpy(dest, device, bytes, cudaMemcpyDeviceToDevice));
  checkCuda(cudaStreamSynchronize(0));
}

// Use rank order for the all-gather. The tuple map translates global
// identities to this rank-major storage even with geometric rank placement.
void mdrtNCCLWaitForRows(void *compute) {
  if (!tupleReady) checkCuda(cudaEventCreateWithFlags(&tupleReady,cudaEventDisableTiming));
  checkCuda(cudaEventRecord(tupleReady,0));
  checkCuda(cudaStreamWaitEvent(compute,tupleReady,0));
}

void mdrtNCCLImportOnStream(double *dest,const double *source,size_t bytes,void *compute) {
  mdrtNCCLWaitForRows(compute);
  if (bytes) checkCuda(cudaMemcpyAsync(dest,source,bytes,cudaMemcpyDeviceToDevice,compute));
  checkCuda(cudaEventRecord(tupleReady,compute));
  checkCuda(cudaStreamWaitEvent(0,tupleReady,0));
}

void mdrtNCCLGatherCoordinates(double *all, const double *owned, int64_t rows,
                               int64_t stride) {
  double *mine = all + 3 * stride * mdrtMPIRank();
  if (rows) checkCuda(cudaMemcpyAsync(mine,owned,3*rows*sizeof(double),cudaMemcpyDeviceToDevice,0));
  if (stride > rows) checkCuda(cudaMemsetAsync(mine+3*rows,0,3*(stride-rows)*sizeof(double),0));
  if (size>1) checkNCCL(ncclAllGather(mine,all,3*stride,ncclDouble,peers,0));
}

void mdrtNCCLJoin(double *out,const double *owned,size_t n,const double *ghost,size_t g) {
  if (n) checkCuda(cudaMemcpyAsync(out,owned,n*sizeof(double),cudaMemcpyDeviceToDevice,0));
  if (g) checkCuda(cudaMemcpyAsync(out+n,ghost,g*sizeof(double),cudaMemcpyDeviceToDevice,0));
}

extern void mdrtNCCLExpandMigration(const void *, int, int *, int);

int *mdrtNCCLMigrationSpans(const void *host, int spans, int rows) {
  size_t bytes=2*(size_t)spans*sizeof(int);
  double *device=mdrtNCCLAllocate(bytes);
  mdrtNCCLUpload(device,host,bytes);
  int *map=(int *)mdrtNCCLAllocate((size_t)rows*sizeof(int));
  mdrtNCCLExpandMigration(device,spans,map,rows);
  mdrtNCCLFreePositions(device);
  return map;
}

int *mdrtNCCLUploadIndices(const int *rows, int64_t n) {
  int *device = (int *)mdrtNCCLAllocate(n*sizeof(int));
  if (n) checkCuda(cudaMemcpy(device,rows,n*sizeof(int),cudaMemcpyHostToDevice));
  return device;
}

void mdrtNCCLDownloadRows(const double *source, const int *indices, int64_t rows,
                          int width, double *host) {
  int *map = mdrtNCCLUploadIndices(indices,rows);
  size_t bytes = (size_t)rows*width*sizeof(double);
  double *wire = mdrtNCCLAllocate(bytes);
  mdrtNCCLRows(1,source,map,wire,rows*width,width,NULL,0,NULL);
  mdrtNCCLDownload(host,wire,bytes);
  mdrtNCCLFreePositions((double *)map);
  mdrtNCCLFreePositions(wire);
}

extern void mdrtNCCLReorderKernel(const double *, const double *, const int *,
                                  double *, long long, int);

// The receive buffer is in former rank and particle identity order. Each
// field follows the same map, including ranks with only sends or receives.
void mdrtNCCLMigrateRows(const double *owned, const int *exportRows, const int *counts,
                         const int *map, double *result, int64_t rows, int width) {
  int64_t send = 0, recv = 0;
  for (int p = 0; p < size; ++p) {
    send += counts[p];
    recv += counts[size+p];
  }
  double *exports = mdrtNCCLAllocate(send*width*sizeof(double));
  double *arrivals = mdrtNCCLAllocate(recv*width*sizeof(double));
  mdrtNCCLRows(1, owned, exportRows, exports, send*width, width, NULL, 0, NULL);
  if (send || recv) {
    size_t out = 0, in = 0;
    checkNCCL(ncclGroupStart());
    for (int peer = 0; peer < size; ++peer) {
      size_t ns = (size_t)counts[peer]*width, nr = (size_t)counts[size+peer]*width;
      if (ns) checkNCCL(ncclSend(exports+out, ns, ncclDouble, peer, peers, 0));
      if (nr) checkNCCL(ncclRecv(arrivals+in, nr, ncclDouble, peer, peers, 0));
      out += ns;
      in += nr;
    }
    checkNCCL(ncclGroupEnd());
  }
  mdrtNCCLReorderKernel(owned, arrivals, map, result, rows, width);
  mdrtNCCLFreePositions(arrivals);
  mdrtNCCLFreePositions(exports);
}

void mdrtNCCLRowsReady(void) {
  // The ordinary reduction stream is nonblocking. Its borrowed fields
  // include updates previously enqueued by Verlet on the default stream.
  checkCuda(cudaStreamSynchronize(0));
}

extern void mdrtNCCLReleaseScratch(void);

void mdrtNCCLFinish(void) {
  if (!peers) return;
  mdrtNCCLReleaseScratch();
  checkNCCL(ncclCommDestroy(peers));
  checkCuda(cudaFree(ownedField));
  checkCuda(cudaFree(ghostField));
  checkCuda(cudaFree(outgoing));
  checkCuda(cudaFree(incoming));
  checkCuda(cudaStreamDestroy(stream));
  if (tupleReady) { checkCuda(cudaEventDestroy(tupleReady)); tupleReady=NULL; }
  if (rowPool) {
    checkCuda(cudaStreamSynchronize(0));
    checkCuda(cudaMemPoolDestroy(rowPool));
    rowPool = NULL;
  }
  free(nodes);
  peers = NULL;
}
