// Device timing of a private, rebuilt and certified frame at a fixed grid.
#include "TowersTrial.h"
#include "TowersCell.h"
#if MDIR_HAS_CUDA
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/DynamicLibrary.h"
#include <algorithm>
#include <cuda.h>
#include <string>
#include <vector>

namespace {
#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)
struct Api {
  llvm::sys::DynamicLibrary library =
      llvm::sys::DynamicLibrary::getPermanentLibrary("libcuda.so.1");
#define ENTRY(f)                                                               \
  decltype(&::f) f = library.isValid()                                         \
                         ? reinterpret_cast<decltype(&::f)>(                   \
                               library.getAddressOfSymbol(STRINGIFY(f)))       \
                         : nullptr;
  ENTRY(cuCtxSynchronize)
  ENTRY(cuCtxPopCurrent)
  ENTRY(cuCtxPushCurrent)
  ENTRY(cuDevicePrimaryCtxRelease)
  ENTRY(cuDevicePrimaryCtxRetain)
  ENTRY(cuEventCreate)
  ENTRY(cuEventDestroy)
  ENTRY(cuEventElapsedTime)
  ENTRY(cuEventRecord)
  ENTRY(cuEventSynchronize)
  ENTRY(cuFuncSetAttribute)
  ENTRY(cuGetErrorName)
  ENTRY(cuGraphDestroy)
  ENTRY(cuGraphExecDestroy)
  ENTRY(cuGraphInstantiateWithFlags)
  ENTRY(cuGraphLaunch)
  ENTRY(cuLaunchKernel)
  ENTRY(cuMemAlloc)
  ENTRY(cuMemFree)
  ENTRY(cuMemcpyDtoH)
  ENTRY(cuMemcpyHtoD)
  ENTRY(cuMemsetD8)
  ENTRY(cuMemsetD8Async)
  ENTRY(cuModuleGetFunction)
  ENTRY(cuModuleLoadData)
  ENTRY(cuModuleUnload)
  ENTRY(cuStreamBeginCapture)
  ENTRY(cuStreamCreate)
  ENTRY(cuStreamDestroy)
  ENTRY(cuStreamEndCapture)
  ENTRY(cuStreamSynchronize)
#undef ENTRY
  bool complete() const {
    return cuCtxSynchronize && cuCtxPopCurrent && cuCtxPushCurrent &&
           cuDevicePrimaryCtxRelease && cuDevicePrimaryCtxRetain &&
           cuEventCreate && cuEventDestroy && cuEventElapsedTime &&
           cuEventRecord && cuEventSynchronize && cuFuncSetAttribute &&
           cuGetErrorName && cuGraphDestroy && cuGraphExecDestroy &&
           cuGraphInstantiateWithFlags && cuGraphLaunch && cuLaunchKernel &&
           cuMemAlloc && cuMemFree && cuMemcpyDtoH && cuMemcpyHtoD &&
           cuMemsetD8 && cuMemsetD8Async && cuModuleGetFunction &&
           cuModuleLoadData && cuModuleUnload && cuStreamBeginCapture &&
           cuStreamCreate && cuStreamDestroy && cuStreamEndCapture &&
           cuStreamSynchronize;
  }
};
const Api &api() {
  static const Api symbols;
  return symbols;
}

} // namespace
namespace mdir::tool {
struct TowersTrial {
  int device, atoms, cells, reach, capacity, exclusions;
  CUcontext context{};
  CUstream stream{};
  CUevent start{}, stop{};
  CUdeviceptr raw{}, box{}, charge{}, types{}, map{}, counts{}, rebuildStatus{},
      geometry{}, keys{}, local{}, packed{}, packedTypes{}, ids{}, inverse{},
      spans{}, state{}, failed{}, words{}, low{}, high{}, bonds{}, output{};
  std::vector<CUdeviceptr> allocations, parameters;
  std::string error;

  bool check(CUresult result, const char *operation) {
    if (result == CUDA_SUCCESS)
      return true;
    const char *name = nullptr;
    api().cuGetErrorName(result, &name);
    error = std::string(operation) + " failed (" + (name ? name : "CUDA") + ")";
    return false;
  }
  bool allocate(CUdeviceptr &ptr, size_t bytes, const void *input = nullptr) {
    if (!check(api().cuMemAlloc(&ptr, bytes), "cuMemAlloc"))
      return false;
    allocations.push_back(ptr);
    return input ? check(api().cuMemcpyHtoD(ptr, input, bytes),
                         "copy trial input")
                 : check(api().cuMemsetD8(ptr, 0, bytes),
                         "initialize trial buffer");
  }
  bool setup(const driver::System &system, const TowersBuffers &buffers) {
    std::vector<double> positions(3ULL * atoms);
    TowersCell cell(system);
    for (int i = 0; i < atoms; ++i) {
      auto point = cell.coordinates(&system.positions[size_t(i) * 3]);
      for (int a = 0; a < 3; ++a)
        positions[size_t(a) * atoms + i] = point[a];
    }
    std::vector<std::pair<unsigned, unsigned>> pairs;
    if (system.topology)
      pairs = system.topology->exclusions;
    for (auto &[i, j] : pairs)
      if (i > j)
        std::swap(i, j);
    llvm::sort(pairs);
    std::vector<int32_t> edges;
    edges.reserve(2 * pairs.size());
    for (auto [i, j] : pairs) {
      edges.push_back(i);
      edges.push_back(j);
    }
    size_t n = atoms, c = cells, slots = c * capacity;
    size_t entries = c * (5 + 2 * reach);
    if (!check(api().cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
               "cuStreamCreate") ||
        !check(api().cuEventCreate(&start, 0), "cuEventCreate") ||
        !check(api().cuEventCreate(&stop, 0), "cuEventCreate") ||
        !allocate(raw, 24 * n, positions.data()) ||
        !allocate(box, cell.triclinic ? 72 : 24,
                  cell.triclinic ? cell.lattice.data() : system.box) ||
        !allocate(map, 4 * entries) || !allocate(counts, 4 * c) ||
        !allocate(rebuildStatus, 4) ||
        !allocate(geometry, 24 * (cell.triclinic ? 13 : 3)) ||
        !allocate(charge, 4 * n) || !allocate(types, 4 * n) ||
        !allocate(keys, 4 * slots) || !allocate(local, 16 * n) ||
        !allocate(packed, 16 * slots) || !allocate(packedTypes, 4 * slots) ||
        !allocate(ids, 4 * slots) || !allocate(inverse, 4 * n) ||
        !allocate(spans, 8 * entries) || !allocate(state, 8) ||
        !allocate(failed, 8) || !allocate(words, 24 * n) ||
        !allocate(low, 24 * n) || !allocate(high, 24 * n) ||
        !allocate(output, 24 * n))
      return false;
    if (exclusions && !allocate(bonds, 8ULL * exclusions, edges.data()))
      return false;
    parameters.resize(buffers.size());
    for (size_t i = 0; i < parameters.size(); ++i)
      if (!allocate(parameters[i], buffers[i].size(), buffers[i].data()))
        return false;
    return check(api().cuCtxSynchronize(), "initialize trial buffers");
  }
};

// All objects created for one candidate are released on errors and on success.
namespace {
struct Sample {
  TowersTrial &f;
  CUmodule module{};
  CUgraph graph{};
  CUgraphExec exec{};
  enum Kernel { Setup, Insert, Pack, Force, Certify, Replay, Publish, Size };
  CUfunction fn[Size]{};
  int threads, shared, split;
  bool capturing = false;
  ~Sample() {
    if (capturing)
      api().cuStreamEndCapture(f.stream, &graph);
    if (exec)
      api().cuGraphExecDestroy(exec);
    if (graph)
      api().cuGraphDestroy(graph);
    if (module)
      api().cuModuleUnload(module);
  }
  bool launch(Kernel k, int64_t blocks, int block, std::vector<void *> args,
              int dynamic = 0) {
    return f.check(api().cuLaunchKernel(fn[k], blocks, 1, 1, block, 1, 1,
                                        dynamic, f.stream, args.data(),
                                        nullptr),
                   "launch trial kernel");
  }
  bool rebuild() {
    int forward = 0;
    return launch(Insert, (f.atoms + 127) / 128, 128,
                  {&f.raw, &f.charge, &f.geometry, &f.counts, &f.keys, &f.local,
                   &forward}) &&
           launch(Pack, (f.cells + 3) / 4, 128,
                  {&f.counts, &f.keys, &f.local, &f.types, &f.map, &f.packed,
                   &f.packedTypes, &f.ids, &f.inverse, &f.spans,
                   &f.rebuildStatus});
  }
  bool step() {
    float scale = 262144;
    std::vector<void *> args{&f.spans, &f.packed, &f.ids, &f.geometry,
                             &f.words, &f.state,  &scale, &f.failed};
    if (f.exclusions)
      args.insert(args.end(), {&f.bonds, &f.inverse, &f.raw, &f.box});
    for (auto &ptr : f.parameters)
      args.push_back(&ptr);
    auto n = f.atoms;
    auto blocks =
        int64_t(f.cells) * split + (f.exclusions + threads - 1) / threads;
    if (!rebuild() ||
        !f.check(api().cuMemsetD8Async(f.words, 0, 24ULL * n, f.stream),
                 "clear force") ||
        !f.check(api().cuMemsetD8Async(f.state, 0, 8, f.stream),
                 "clear status") ||
        !f.check(api().cuMemsetD8Async(f.failed, 0, 8, f.stream),
                 "clear conversions") ||
        !launch(Force, blocks, threads, args, shared) ||
        !launch(Certify, (n + 127) / 128, 128, {&f.words, &f.failed, &f.state}))
      return false;
    args[4] = &f.low;
    args.push_back(&f.high);
    return launch(Replay, 1, threads, args, shared) &&
           launch(Publish, (n + 127) / 128, 128,
                  {&f.state, &f.words, &f.low, &f.output});
  }
  bool measure(const void *image, float &us) {
    static const char *names[] = {"towers_setup",   "towers_insert",
                                  "towers_pack",    "towers_force",
                                  "towers_certify", "towers_recover_once",
                                  "towers_publish"};
    if (!f.check(api().cuModuleLoadData(&module, image),
                 "load trial specialization"))
      return false;
    for (int k = 0; k < Size; ++k)
      if (!f.check(api().cuModuleGetFunction(&fn[k], module, names[k]),
                   names[k]))
        return false;
    for (auto k : {Force, Replay})
      if (!f.check(api().cuFuncSetAttribute(
                       fn[k], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                       shared),
                   names[k]))
        return false;
    int overflow = 0;
    if (!launch(Setup, (f.cells + 127) / 128, 128,
                {&f.map, &f.counts, &f.rebuildStatus, &f.box, &f.geometry}) ||
        !rebuild() ||
        !f.check(api().cuStreamSynchronize(f.stream),
                 "check initial rebuild") ||
        !f.check(api().cuMemcpyDtoH(&overflow, f.rebuildStatus, 4),
                 "read rebuild status"))
      return false;
    if (overflow) {
      f.error =
          "trial cell capacity must be at least " + std::to_string(overflow);
      return false;
    }
    if (!f.check(api().cuStreamBeginCapture(
                     f.stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
                 "begin trial graph"))
      return false;
    capturing = true;
    for (int i = 0; i < 9; ++i)
      if (!step())
        return false;
    capturing = false;
    if (!f.check(api().cuStreamEndCapture(f.stream, &graph),
                 "end trial graph") ||
        !f.check(api().cuGraphInstantiateWithFlags(&exec, graph, 0),
                 "instantiate trial graph") ||
        !f.check(api().cuGraphLaunch(exec, f.stream), "warm trial graph") ||
        !f.check(api().cuEventRecord(f.start, f.stream),
                 "record trial start") ||
        !f.check(api().cuGraphLaunch(exec, f.stream), "launch trial graph") ||
        !f.check(api().cuEventRecord(f.stop, f.stream), "record trial stop") ||
        !f.check(api().cuEventSynchronize(f.stop), "wait for trial") ||
        !f.check(api().cuEventElapsedTime(&us, f.start, f.stop),
                 "read elapsed time"))
      return false;
    int fatal = 0;
    if (!f.check(api().cuMemcpyDtoH(&fatal, f.state, 4),
                 "read trial certificate"))
      return false;
    if (fatal) {
      f.error =
          "trial failed force certification, status " + std::to_string(fatal);
      return false;
    }
    us *= 1000.0f / 9;
    return true;
  }
};
} // namespace

llvm::Expected<TrialFrame> createTowersTrial(int device,
                                             const driver::System &system,
                                             int cells, int reach, int capacity,
                                             const TowersBuffers &buffers) {
  if (!api().complete())
    return llvm::createStringError(
        "towers_shape: missing CUDA graph trial entry points");
  TrialFrame f(new TowersTrial{
      device, int(system.getNumParticles()), cells, reach, capacity,
      system.topology ? int(system.topology->exclusions.size()) : 0});
  if (!f->check(api().cuDevicePrimaryCtxRetain(&f->context, device),
                "retain trial context") ||
      !f->check(api().cuCtxPushCurrent(f->context), "enter trial context"))
    return llvm::createStringError("towers_shape: %s", f->error.c_str());
  auto restore = llvm::scope_exit([] {
    CUcontext old;
    api().cuCtxPopCurrent(&old);
  });
  if (!f->setup(system, buffers))
    return llvm::createStringError("towers_shape: %s", f->error.c_str());
  return f;
}

llvm::Expected<float> measureTowersTrial(TowersTrial &f,
                                         const std::string &image,
                                         const driver::TowersShape &shape) {
  if (!f.check(api().cuCtxPushCurrent(f.context), "enter trial context"))
    return llvm::createStringError("towers_shape: %s", f.error.c_str());
  auto restore = llvm::scope_exit([] {
    CUcontext old;
    api().cuCtxPopCurrent(&old);
  });
  Sample sample{f};
  sample.threads = shape.threads;
  sample.shared = 8 * shape.records;
  sample.split = shape.split;
  float us;
  if (!sample.measure(image.data(), us))
    return llvm::createStringError("towers_shape: %s", f.error.c_str());
  return us;
}
void DropTowersTrial::operator()(TowersTrial *f) const {
  if (f->context) {
    if (api().cuCtxPushCurrent(f->context) == CUDA_SUCCESS) {
      if (f->stream)
        api().cuStreamSynchronize(f->stream);
      for (auto ptr : f->allocations)
        api().cuMemFree(ptr);
      if (f->start)
        api().cuEventDestroy(f->start);
      if (f->stop)
        api().cuEventDestroy(f->stop);
      if (f->stream)
        api().cuStreamDestroy(f->stream);
      CUcontext old;
      api().cuCtxPopCurrent(&old);
    }
    api().cuDevicePrimaryCtxRelease(f->device);
  }
  delete f;
}

} // namespace mdir::tool
#endif
