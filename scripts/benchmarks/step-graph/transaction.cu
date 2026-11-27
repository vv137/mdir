// Exercise the same graph runtime as the TOWERS step, including
// recovery of the last accepted banks after drift or force rejects a step.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <math_constants.h>

extern "C" cudaStream_t mgpuStreamCreate();
extern "C" intptr_t mdrtStepCreate();
extern "C" void mdrtStepSeed(intptr_t, int64_t, double, double);
extern "C" intptr_t mdrtStepBank(intptr_t, int64_t, int64_t);
extern "C" int64_t mdrtStepPhases(intptr_t);
extern "C" intptr_t mdrtStepFatalAddress(intptr_t);
extern "C" int mdrtStepBegin(intptr_t, int64_t, int64_t, int64_t, intptr_t);
extern "C" void mdrtStepCapture(intptr_t);
extern "C" void mdrtStepDrift(intptr_t, intptr_t, int64_t, intptr_t, intptr_t, intptr_t, intptr_t);
extern "C" void mdrtStepForce(intptr_t, intptr_t);
extern "C" void mdrtStepEnd(intptr_t);
extern "C" int64_t mdrtStepRun(intptr_t, int64_t, int64_t);
extern "C" int64_t mdrtStepRead(intptr_t, int64_t);
extern "C" void mdrtStepDestroy(intptr_t);
extern "C" void mdrtStepMirror(intptr_t, intptr_t, intptr_t, int64_t);

static void ck(cudaError_t e) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "transaction: %s\n", cudaGetErrorString(e));
    std::exit(1);
  }
}

// Mode 5 drifts by the cell that a mirror holds (D[towers-graph-npt]):
// 1 for the first chunk and 2 after the host changes it.
__global__ void drift(const double *input, double *proposal, int mode, int *flag, const int *fatal,
                      const double *cell) {
  if (fatal && *fatal) return;
  proposal[0] = input[0] + (mode == 5 ? *cell : 1);
  proposal[1] = mode == 2 && proposal[0] == 4 ? CUDART_INF : 0;
  proposal[2] = 0;
  *flag = mode == 1 && proposal[0] == 4;
}
__global__ void force(const double *proposal, int mode, int *status, int *calls, const int *fatal) {
  if (fatal && *fatal) return;
  // Mode 5 rejects the first proposal after the cell changed, once.
  *status = proposal[0] == 4 && (mode == 3 || (mode == 4 && *calls == 3) || (mode == 5 && *calls == 2));
  ++*calls;
}
__global__ void integrate(int *calls, const int *certificate, const int *fatal) {
  if (fatal && *fatal) return;
  if (!certificate || !*certificate) ++*calls;
}

static void trial(int mode) {
  cudaStream_t stream = mgpuStreamCreate();
  double *bank[2];
  for (auto &ptr : bank) {
    ck(cudaMalloc(&ptr, 3 * sizeof(double)));
    ck(cudaMemsetAsync(ptr, 0, 3*sizeof(double), stream));
  }
  int *flags;
  double *cell, host = 1;
  ck(cudaMalloc(&cell, sizeof(double)));
  ck(cudaMalloc(&flags, 4*sizeof(int)));
  ck(cudaMemsetAsync(flags, 0, 4*sizeof(int), stream));
  constexpr int64_t restart = (int64_t(1) << 32) + 7;
  intptr_t cache = mdrtStepCreate();
  mdrtStepSeed(cache, restart, 19.25, .002);
  int64_t table[] = {intptr_t(bank[0]),24,1, intptr_t(bank[1]),24,2};
  auto capture = [&]() {
    if (!mdrtStepBegin(cache, 1, 2, 0, intptr_t(table))) return false;
    mdrtStepCapture(cache);
    auto *fatal = reinterpret_cast<int *>(mdrtStepFatalAddress(cache));
    int64_t phases = mdrtStepPhases(cache);
    for (int64_t i = 0; i < phases; ++i) {
      auto *input = reinterpret_cast<double *>(mdrtStepBank(cache,0,0));
      auto *proposal = reinterpret_cast<double *>(mdrtStepBank(cache,1,1));
      drift<<<1,1,0,stream>>>(input,proposal,mode,flags,fatal,cell);
      mdrtStepDrift(cache,intptr_t(proposal),1,intptr_t(flags),0,0,0);
      force<<<1,1,0,stream>>>(proposal,mode,flags+1,flags+2,fatal);
      mdrtStepForce(cache,intptr_t(flags+1));
      integrate<<<1,1,0,stream>>>(flags+3, flags+1,fatal);
      mdrtStepEnd(cache);
    }
    return true;
  };
  if (!capture()) std::exit(2);
  mdrtStepMirror(cache, intptr_t(cell), intptr_t(&host), sizeof(host));
  int64_t accepted = mdrtStepRun(cache,restart,2);
  if (accepted != restart+2) std::exit(3);
  table[0] = mdrtStepBank(cache,0,2);
  table[3] = mdrtStepBank(cache,1,2);
  capture();
  if (mode == 5) host = 2;
  mdrtStepMirror(cache, intptr_t(cell), intptr_t(&host), sizeof(host));
  accepted = mdrtStepRun(cache,accepted,18);
  int expected = mode > 0 && mode < 4 ? 3 : 20;
  double value = mode == 5 ? 2 + 2 * 18 : expected;
  if (mdrtStepRead(cache,restart+expected) != accepted) std::exit(4);
  int f[4];
  double result;
  ck(cudaMemcpyAsync(f,flags,sizeof(f),cudaMemcpyDeviceToHost,stream));
  auto *published = reinterpret_cast<double *>(mdrtStepBank(cache,0,2));
  ck(cudaMemcpyAsync(&result,published,sizeof(result),cudaMemcpyDeviceToHost,stream));
  ck(cudaStreamSynchronize(stream));
  if (result != value || f[2] != expected + (mode == 3 ? 2 : mode >= 4 ? 1 : 0) || f[3] != expected) {
    std::fprintf(stderr, "transaction mismatch: mode %d value %g force %d kick %d\n",
                 mode,result,f[2],f[3]);
    std::exit(5);
  }
  std::printf("failure %d last accepted %lld value %.0f forces %d kicks %d\n",
              mode,(long long)accepted,result,f[2],f[3]);
  mdrtStepDestroy(cache);
  for (auto ptr : bank) ck(cudaFree(ptr));
  ck(cudaFree(flags));
  ck(cudaFree(cell));
}
int main() {
  for (int mode=0;mode<6;++mode) trial(mode);
}
