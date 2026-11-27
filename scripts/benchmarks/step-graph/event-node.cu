// Does cuEventSynchronize on an event recorded by an external event-record
// node of a graph wait for the launch just issued? A kernel spins ~1 ms and
// then stores the launch number into mapped host memory; the event node
// follows it. After each launch the host waits for the event and reads the
// word: it must equal the launch number, every time.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { CUresult r = (x); if (r) { const char *s; cuGetErrorName(r, &s); printf("%s: %s\n", #x, s); exit(1); } } while (0)
__global__ void spin(volatile int *word, const int *counter, long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {}
  *word = *counter;
}
__global__ void bump(int *counter) { ++*counter; }
int main() {
  cudaFree(0);
  int *host; cudaHostAlloc(&host, 4, cudaHostAllocMapped); *host = 0;
  int *dev; cudaHostGetDevicePointer(&dev, host, 0);
  int *counter; cudaMalloc(&counter, 4); cudaMemset(counter, 0, 4);
  CUstream s; CK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
  CUevent e; CK(cuEventCreate(&e, CU_EVENT_DISABLE_TIMING));
  int *sink; cudaMalloc(&sink, 4);
  // The graph: bump, spin (stores the counter), external event record, then
  // a long tail that the host must not wait for.
  CK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
  bump<<<1, 1, 0, (cudaStream_t)s>>>(counter);
  spin<<<1, 1, 0, (cudaStream_t)s>>>(dev, counter, 2000000LL);
  CK(cuEventRecordWithFlags(e, s, CU_EVENT_RECORD_EXTERNAL));
  spin<<<1, 1, 0, (cudaStream_t)s>>>(dev + 0, counter, 0);  // same value again
  spin<<<1, 1, 0, (cudaStream_t)s>>>(sink, counter, 4000000LL);
  CUgraph g; CK(cuStreamEndCapture(s, &g));
  CUgraphExec x; CK(cuGraphInstantiate(&x, g, 0));
  int bad = 0, early = 0;
  for (int i = 1; i <= 200; ++i) {
    CK(cuGraphLaunch(x, s));
    CK(cuEventSynchronize(e));
    int seen = *(volatile int *)host;
    if (seen != i) ++bad;
    if (cuStreamQuery(s) == CUDA_SUCCESS) ++early;  // the tail should still run
  }
  CK(cuStreamSynchronize(s));
  printf("event node: %d launches, %d stale reads, %d waits that also found the tail done\n", 200, bad, early);
  return bad != 0;
}
