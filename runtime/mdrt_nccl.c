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
};

static void *upload(const void *host, size_t bytes) {
  void *device = NULL;
  if (bytes) {
    checkCuda(cudaMalloc(&device, bytes));
    checkCuda(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
  }
  return device;
}

struct NCCLPlan *mdrtNCCLPlan(int64_t domains, const int64_t *ghosts,
    int64_t capacity, const int64_t *sourceDomain, const int64_t *sourceRow,
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
        int64_t i = d*capacity+g;
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
  free(sources); free(targets); free(shifts);
  return p;
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
    double *ghosts, int k, int image, int reverse, double *packed) {
  if (image && k != 3) mdrtMPIAbort("imaged halo positions require three components");
  size_t row = (size_t)k * sizeof(double);
  if (packed && size==1 && !reverse) {
    if (nOwned) checkCuda(cudaMemcpyAsync(packed,owned,nOwned*row,cudaMemcpyDeviceToDevice,stream));
    mdrtNCCLSelfGather(owned,p->source,image ? p->shifts : NULL,packed+nOwned*k,
                       (long long)p->recvRows*k,k,stream);
    checkCuda(cudaStreamSynchronize(stream));
    return;
  }
  if (!packed) {
    reserve(&ownedField, &ownedCapacity, nOwned*row);
    reserve(&ghostField, &ghostCapacity, p->recvRows*row);
  }
  double *ownedDevice = packed ? packed : ownedField;
  double *ghostDevice = packed ? packed + nOwned*k : ghostField;
  int send = reverse ? p->recvRows : p->sendRows;
  int recv = reverse ? p->sendRows : p->recvRows;
  reserve(&outgoing, &outCapacity, send*row);
  reserve(&incoming, &inCapacity, recv*row);
  if (nOwned) checkCuda(cudaMemcpyAsync(ownedDevice, owned, nOwned*row, packed ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, stream));
  if (reverse && p->recvRows)
    checkCuda(cudaMemcpyAsync(ghostDevice, ghosts, p->recvRows*row, cudaMemcpyHostToDevice, stream));
  mdrtNCCLRows(1, reverse ? ghostDevice : ownedDevice, reverse ? p->dest : p->source,
               outgoing, (long long)send*k, k, NULL, reverse, stream);
  checkCuda(cudaStreamSynchronize(stream));
  int rank = mdrtMPIRank();
  for (int inter = 0; inter <= 1; ++inter) {
    if (!p->hasPeers[inter]) continue;
    int phase = inter ? DistInterHalo : DistIntraHalo;
    if (!reverse) mdrtDistPhase(phase);
    checkNCCL(ncclGroupStart());
    for (int peer = 0; peer < size; ++peer) {
      if ((nodes[peer] != nodes[rank]) != inter) continue;
      size_t ns = (size_t)(reverse ? p->nr[peer] : p->ns[peer])*k;
      size_t nr = (size_t)(reverse ? p->ns[peer] : p->nr[peer])*k;
      if (ns) {
        size_t offset = (size_t)(reverse ? p->offsetRecv[peer] : p->os[peer])*k;
        checkNCCL(ncclSend(outgoing+offset, ns, ncclDouble, peer, peers, stream));
      }
      if (nr) {
        size_t offset = (size_t)(reverse ? p->os[peer] : p->offsetRecv[peer])*k;
        checkNCCL(ncclRecv(incoming+offset, nr, ncclDouble, peer, peers, stream));
      }
    }
    checkNCCL(ncclGroupEnd());
    checkCuda(cudaStreamSynchronize(stream));
    if (!reverse) mdrtDistPhase(-phase);
  }
  double *result = reverse ? ownedDevice : ghostDevice;
  mdrtNCCLRows(0, incoming, reverse ? p->source : p->dest, result,
               (long long)recv*k, k, image ? p->shifts : NULL, reverse, stream);
  size_t bytes = (reverse ? nOwned : p->recvRows)*row;
  if (!packed && bytes) checkCuda(cudaMemcpyAsync(reverse ? owned : ghosts, result, bytes,
                                       cudaMemcpyDeviceToHost, stream));
  checkCuda(cudaStreamSynchronize(stream));
}

void mdrtNCCLHalo(const struct NCCLPlan *plan, double *owned, int64_t rows,
    double *ghosts, int k, int image, int reverse) {
  exchange(plan, owned, rows, ghosts, k, image, reverse, NULL);
}

/* The outlined MATRIX borrows this immutable buffer until its block ends.
   Owned rows precede ghosts in the order of the neighborhood. */
double *mdrtNCCLPackedRows(const struct NCCLPlan *plan, double *owned, int64_t rows, int k, int image) {
  double *positions;
  size_t count = rows + (size_t)plan->recvRows;
  positions = mdrtNCCLAllocate((count ? count : 1)*k*sizeof(double));
  // The integration kernel writes owned rows on the default stream.
  checkCuda(cudaStreamSynchronize(0));
  exchange(plan, owned, rows, NULL, k, image, 0, positions);
  return positions;
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

void mdrtNCCLReorder(const double *owned, const double *incoming, int64_t received,
                     const int *map, double *result, int64_t rows, int width) {
  double *arrivals = upload(incoming,(size_t)received*width*sizeof(double));
  mdrtNCCLReorderKernel(owned,arrivals,map,result,rows,width);
  checkCuda(cudaFree(arrivals));
}

void mdrtNCCLRowsReady(void) {
  // The ordinary reduction stream is nonblocking. Its borrowed fields
  // include updates previously enqueued by Verlet on the default stream.
  checkCuda(cudaStreamSynchronize(0));
}

void mdrtNCCLFinish(void) {
  if (!peers) return;
  checkNCCL(ncclCommDestroy(peers));
  checkCuda(cudaFree(ownedField));
  checkCuda(cudaFree(ghostField));
  checkCuda(cudaFree(outgoing));
  checkCuda(cudaFree(incoming));
  checkCuda(cudaStreamDestroy(stream));
  if (rowPool) {
    checkCuda(cudaStreamSynchronize(0));
    checkCuda(cudaMemPoolDestroy(rowPool));
    rowPool = NULL;
  }
  free(nodes);
  peers = NULL;
}
