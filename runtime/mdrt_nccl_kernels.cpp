// Device row maps of the halo incidence operator and its transpose.
#include "mdrt_mpi.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <cstdlib>
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
}
extern "C" __global__ void reorder(const double *owned, const double *incoming,
    const int *map, double *result, long long rows, int width) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows*width) return;
  int row = map[i/width];
  result[i] = row >= 0 ? owned[(long long)row*width+i%width]
                       : incoming[(-1LL-row)*width+i%width];
}
struct Box { double length[3]; int grid[3]; };
extern "C" __global__ void faceRecords(const double *x, long long rows, Box box,
    int oldDomain, double reach, int *count, long long *output) {
  long long r = (long long)blockIdx.x*blockDim.x+threadIdx.x;
  if (r >= rows) return;
  int cell[3];
  double lower[3], upper[3];
  for (int k=0; k<3; ++k) {
    double length=box.length[k], u=x[3*r+k]/length;
    if (!isfinite(u)) {
      int i=atomicAdd(count,1);
      output[2LL*i]=-1;
      output[2LL*i+1]=0;
      return;
    }
    double whole=floor(u), wrapped=__dsub_rn(x[3*r+k],__dmul_rn(length,whole));
    int grid=box.grid[k];
    cell[k]=max(0,min(grid-1,(int)floor((u-whole)*grid)));
    lower[k]=fmax(0.,wrapped-length*cell[k]/grid);
    upper[k]=fmax(0.,length*(cell[k]+1)/grid-wrapped);
  }
  int domain=cell[0]+box.grid[0]*(cell[1]+box.grid[1]*cell[2]);
  unsigned mask=0;
  // With a cell wider than two reaches only immediate neighbors qualify.
  for (int a=(lower[0]<=reach ? -1 : 0); a<=(upper[0]<=reach ? 1 : 0); ++a)
    for (int b=(lower[1]<=reach ? -1 : 0); b<=(upper[1]<=reach ? 1 : 0); ++b)
      for (int c=(lower[2]<=reach ? -1 : 0); c<=(upper[2]<=reach ? 1 : 0); ++c) {
        if (!a && !b && !c) continue;
        double dx=a<0?lower[0]:a>0?upper[0]:0.;
        double dy=b<0?lower[1]:b>0?upper[1]:0.;
        double dz=c<0?lower[2]:c>0?upper[2]:0.;
        if (dx*dx+dy*dy+dz*dz<=reach*reach)
          mask |= 1u<<((a+1)*9+(b+1)*3+c+1);
      }
  if (mask || domain!=oldDomain) {
    int i=atomicAdd(count,1);
    output[2LL*i]=r;
    output[2LL*i+1]=((long long)domain<<32)|mask;
  }
}
extern "C" __global__ void wrapRows(const double *source, double *target,
                                     long long rows, Box box) {
  long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=3*rows) return;
  double x=source[i], length=box.length[i%3];
  target[i]=__dsub_rn(x,__dmul_rn(length,floor(x/length)));
}
extern "C" __global__ void narrow(const double *source, int *target, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) target[i] = (int)source[i];
}
extern "C" __global__ void advance(const double *field, const double *slope,
    const double *mass, double *result, long long n, double dt) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3*n) return;
  double w = !mass ? dt : mass[i/3] == 0. ? 0. : dt/mass[i/3];
  result[i] = __dadd_rn(field[i], __dmul_rn(w, slope[i]));
}
extern "C" __global__ void coverage(const double *x, const double *ref,
    long long n, double limit, int *flag) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double r2 = 0.;
  for (int c = 0; c < 3; ++c) {
    double v = x[3*i+c]-ref[3*i+c];
    if (!isfinite(v)) { atomicMax(flag,2); return; }
    r2 = __dadd_rn(r2,__dmul_rn(v,v));
  }
  if (!(r2 < limit)) atomicMax(flag,1);
})";
CUmodule module;
CUfunction pack, unpack, advance, coverage, narrow, reorder, faceRecords, wrapRows;

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
  check(cuModuleGetFunction(&faceRecords,module,"faceRecords"));
  check(cuModuleGetFunction(&wrapRows,module,"wrapRows"));
  check(cuModuleGetFunction(&reorder, module, "reorder"));
  check(cuModuleGetFunction(&narrow, module, "narrow"));
  check(cuModuleGetFunction(&advance, module, "advance"));
  check(cuModuleGetFunction(&coverage, module, "coverage"));
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

extern "C" void mdrtNCCLAdvance(const double *x, const double *v,
    const double *mass, double *result, long long rows, double dt) {
  if (!rows) return;
  if (!module) compile();
  void *args[] = {&x,&v,&mass,&result,&rows,&dt};
  check(cuLaunchKernel(advance, (3*rows+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));
}

extern "C" int mdrtNCCLCoverage(const double *x, const double *ref,
                                long long rows, double limit) {
  if (!rows) return 0;
  if (!module) compile();
  CUdeviceptr flag;
  check(cuMemAlloc(&flag,sizeof(int)));
  check(cuMemsetD32(flag,0,1));
  void *args[] = {&x,&ref,&rows,&limit,&flag};
  check(cuLaunchKernel(coverage,(rows+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));
  int value;
  check(cuMemcpyDtoH(&value,flag,sizeof(value)));
  check(cuMemFree(flag));
  return value;
}

extern "C" void mdrtNCCLNarrow(const double *source, void *target, long long rows) {
  if (!rows) return;
  if (!module) compile();
  void *args[] = {&source,&target,&rows};
  check(cuLaunchKernel(narrow,(rows+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));
  check(cuStreamSynchronize(nullptr));
}

extern "C" void mdrtNCCLReorderKernel(const double *owned, const double *incoming,
    const int *map, double *result, long long rows, int width) {
  if (!rows) return;
  if (!module) compile();
  void *args[] = {&owned, &incoming, &map, &result, &rows, &width};
  check(cuLaunchKernel(reorder,(rows*width+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));
}

namespace {
struct Box { double length[3]; int grid[3]; };
}

extern "C" int64_t *mdrtNCCLFaceRecords(const double *positions, long long rows,
    int domain, const int64_t *grid, const double *length, double reach, int64_t *size) {
  *size=0;
  if (!rows) return nullptr;
  if (!module) compile();
  Box box;
  for (int k=0; k<3; ++k) {
    box.length[k]=length[k];
    box.grid[k]=grid[k];
  }
  CUdeviceptr deviceCount, records;
  check(cuMemAlloc(&deviceCount,sizeof(int)));
  check(cuMemAlloc(&records,2*rows*sizeof(int64_t)));
  check(cuMemsetD32(deviceCount,0,1));
  void *args[]={&positions,&rows,&box,&domain,&reach,&deviceCount,&records};
  check(cuLaunchKernel(faceRecords,(rows+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));
  int count;
  check(cuMemcpyDtoH(&count,deviceCount,sizeof count));
  size_t bytes=2*(size_t)count*sizeof(int64_t);
  auto *host=static_cast<int64_t *>(std::malloc(bytes ? bytes : sizeof(int64_t)));
  if (!host) mdrtMPIAbort("out of memory collecting face records");
  if (bytes) check(cuMemcpyDtoH(host,records,bytes));
  check(cuMemFree(records));
  check(cuMemFree(deviceCount));
  *size=count;
  return host;
}

extern "C" void mdrtNCCLWrapRows(const double *source, double *target, long long rows,
                                 const double *length) {
  if (!rows) return;
  if (!module) compile();
  Box box{};
  for (int k=0; k<3; ++k) box.length[k]=length[k];
  void *args[]={&source,&target,&rows,&box};
  check(cuLaunchKernel(wrapRows,(3*rows+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));
}
