// Time fixed conditional handles to isolate IF dispatch from predicate production.
// Each IF has a matching static graph with the same executed kernels.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
    std::exit(1);
  }
}

__global__ void tick(unsigned *count) { ++*count; }

__global__ void produce(unsigned *count, cudaGraphConditionalHandle condition, unsigned value) {
  ++*count;
  cudaGraphSetConditional(condition, value);
}

template<class Kernel, class... Args>
static cudaGraphNode_t kernel(cudaGraph_t g, cudaGraphNode_t dep, Kernel f, Args... args) {
  cudaKernelNodeParams p{};
  p.func = reinterpret_cast<void *>(f);
  p.gridDim = p.blockDim = dim3(1);
  void *params[] = {&args...};
  p.kernelParams = params;
  cudaGraphNode_t n;
  check(cudaGraphAddKernelNode(&n, g, dep ? &dep : nullptr, dep ? 1 : 0, &p));
  return n;
}

static void trial(bool conditional, bool taken, bool body, bool produced, unsigned *ptr) {
  constexpr int sites = 32, repeats = 40;
  cudaGraph_t g;
  check(cudaGraphCreate(&g, 0));
  cudaGraphNode_t tail = nullptr;
  for (int i = 0; i < sites; ++i) {
    if (conditional) {
      cudaGraphNodeParams p{};
      p.type = cudaGraphNodeTypeConditional;
      p.conditional.type = cudaGraphCondTypeIf;
      p.conditional.size = 1;
      check(cudaGraphConditionalHandleCreate(&p.conditional.handle, g, produced ? !taken : taken,
                                            cudaGraphCondAssignDefault));
      if (produced)
        tail = kernel(g, tail, produce, ptr, p.conditional.handle, unsigned(taken));
      else
        tail = kernel(g, tail, tick, ptr);
      cudaGraphNode_t n;
      check(cudaGraphAddNode(&n, g, &tail, nullptr, 1, &p));
      if (body)
        kernel(p.conditional.phGraph_out[0], nullptr, tick, ptr);
      tail = n;
    } else {
      tail = kernel(g, tail, tick, ptr);
      if (taken && body) tail = kernel(g, tail, tick, ptr);
    }
  }
  cudaGraphExec_t exec;
  cudaEvent_t start, end;
  check(cudaGraphInstantiate(&exec, g, 0));
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&end));
  check(cudaMemset(ptr, 0, sizeof(*ptr)));
  constexpr int warm = 256;
  for (int i = 0; i < warm; ++i) check(cudaGraphLaunch(exec, nullptr));
  check(cudaDeviceSynchronize());
  std::array<float, 9> samples;
  for (float &sample : samples) {
    check(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i)
      check(cudaGraphLaunch(exec, nullptr));
    check(cudaEventRecord(end));
    check(cudaEventSynchronize(end));
    check(cudaEventElapsedTime(&sample, start, end));
    sample *= 1000.f / (sites * repeats);
  }
  unsigned result;
  check(cudaMemcpy(&result, ptr, sizeof(result), cudaMemcpyDeviceToHost));
  if (result != sites * (1 + (taken && body)) * (warm + repeats * samples.size()))
    std::exit(2);
  std::sort(samples.begin(), samples.end());
  std::printf("conditional=%d taken=%d body=%d produced=%d us/site %.5f [%.5f, %.5f]\n",
              conditional, taken, body, produced, samples[4], samples.front(), samples.back());
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(end));
  check(cudaGraphExecDestroy(exec));
  check(cudaGraphDestroy(g));
}

int main(int argc, char **) {
  unsigned *ptr;
  check(cudaMalloc(&ptr, sizeof(*ptr)));
  for (bool produced : {false, true})
    for (bool body : {false, true})
      for (bool taken : {false, true})
        for (bool conditional : {false, true})
          trial(conditional != (argc > 1), taken, body, produced, ptr);
  check(cudaFree(ptr));
}
