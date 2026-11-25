// Device row maps of the halo incidence operator and its transpose.
#include "mdrt_mpi.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <string>
#include <vector>

namespace {
constexpr char source[] = R"(
extern "C" __global__ void pack(const double *field, const int *row,
                                double *wire, long long n, int k) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) wire[i] = field[(long long)row[i/k]*k + i%k];
}
extern "C" __global__ void unpack(const double *wire, const int *row,
                                  double *field, long long n, int k,
                                  const double *shift, int transpose) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long dest = (long long)row[i/k]*k + i%k;
  if (transpose) atomicAdd(field + dest, wire[i]);
  else field[dest] = wire[i] + (shift ? shift[(i/k)*3+i%k] : 0.);
})";
CUmodule module;
CUfunction pack, unpack;

void check(CUresult status) {
  if (status == CUDA_SUCCESS) return;
  const char *message = "CUDA halo kernel error";
  cuGetErrorString(status, &message);
  mdrtMPIAbort(message);
}

void compile() {
  CUdevice device;
  int major, minor;
  check(cuCtxGetDevice(&device));
  check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
  const char *options[] = {arch.c_str()};
  nvrtcProgram program;
  auto verify = [](nvrtcResult result) {
    if (result != NVRTC_SUCCESS) mdrtMPIAbort(nvrtcGetErrorString(result));
  };
  verify(nvrtcCreateProgram(&program, source, "halo.cu", 0, nullptr, nullptr));
  if (nvrtcCompileProgram(program, 1, options) != NVRTC_SUCCESS) {
    size_t size;
    verify(nvrtcGetProgramLogSize(program, &size));
    std::vector<char> log(size);
    verify(nvrtcGetProgramLog(program, log.data()));
    mdrtMPIAbort(log.data());
  }
  size_t size;
  verify(nvrtcGetPTXSize(program, &size));
  std::vector<char> ptx(size);
  verify(nvrtcGetPTX(program, ptx.data()));
  verify(nvrtcDestroyProgram(&program));
  check(cuModuleLoadData(&module, ptx.data()));
  check(cuModuleGetFunction(&pack, module, "pack"));
  check(cuModuleGetFunction(&unpack, module, "unpack"));
}
}

extern "C" void mdrtNCCLRows(int gather, const double *input, const int *rows,
                             double *output, long long n, int k,
                             const double *shifts, int transpose, CUstream stream) {
  if (!n) return;
  if (!module) compile();
  void *args[] = {&input, &rows, &output, &n, &k, &shifts, &transpose};
  check(cuLaunchKernel(gather ? pack : unpack, (n + 255)/256, 1, 1, 256, 1, 1,
                       0, stream, args, nullptr));
}
