#include "mdrt_nccl.h"
#include "mdrt_mpi.h"
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ncclComm_t peers;
static cudaStream_t stream;
static int size;
static int *nodes;
static double *outgoing, *incoming;
static size_t outCapacity, inCapacity;

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

void mdrtNCCLExchangeDevice(const double *send, const int *ns, const int *os,
                            double *recv, const int *nr, const int *or) {
  checkNCCL(ncclGroupStart());
  for (int p = 0; p < size; ++p) {
    if (ns[p])
      checkNCCL(ncclSend(send + os[p], ns[p], ncclDouble, p, peers, stream));
    if (nr[p])
      checkNCCL(ncclRecv(recv + or[p], nr[p], ncclDouble, p, peers, stream));
  }
  checkNCCL(ncclGroupEnd());
  checkCuda(cudaStreamSynchronize(stream));
}

static void reserve(double **buffer, size_t *capacity, size_t bytes) {
  if (bytes <= *capacity) return;
  checkCuda(cudaFree(*buffer));
  checkCuda(cudaMalloc((void **)buffer, bytes));
  *capacity = bytes;
}

/* The step-state lowering currently packs MPI payloads in host storage.
   Keep both peer buffers on this rank's GPU across exchanges. */
void mdrtNCCLExchangeHost(const double *send, const int *ns, const int *os,
                          double *recv, const int *nr, const int *or) {
  size_t sent = ((size_t)os[size - 1] + ns[size - 1]) * sizeof(double);
  size_t received = ((size_t)or[size - 1] + nr[size - 1]) * sizeof(double);
  reserve(&outgoing, &outCapacity, sent);
  reserve(&incoming, &inCapacity, received);
  if (sent) checkCuda(cudaMemcpyAsync(outgoing, send, sent, cudaMemcpyHostToDevice, stream));
  mdrtNCCLExchangeDevice(outgoing, ns, os, incoming, nr, or);
  if (received) {
    checkCuda(cudaMemcpyAsync(recv, incoming, received, cudaMemcpyDeviceToHost, stream));
    checkCuda(cudaStreamSynchronize(stream));
  }
}

void mdrtNCCLFinish(void) {
  if (!peers) return;
  checkNCCL(ncclCommDestroy(peers));
  checkCuda(cudaFree(outgoing));
  checkCuda(cudaFree(incoming));
  checkCuda(cudaStreamDestroy(stream));
  free(nodes);
  peers = NULL;
}
