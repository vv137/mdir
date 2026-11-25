// Device timing of a private, rebuilt and certified frame at a fixed grid.
#include "TowersTrial.h"
#include "TowersCell.h"
#include "mdir/Conversion/TowersLaunch.h"
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
  ENTRY(cuEventRecordWithFlags)
  ENTRY(cuEventSynchronize)
  ENTRY(cuFuncSetAttribute)
  ENTRY(cuGetErrorName)
  ENTRY(cuGraphDestroy)
  ENTRY(cuGraphExecDestroy)
  ENTRY(cuGraphInstantiateWithFlags)
  ENTRY(cuGraphLaunch)
  ENTRY(cuDeviceGetAttribute)
  ENTRY(cuLaunchCooperativeKernel)
  ENTRY(cuLaunchKernel)
  ENTRY(cuMemAlloc)
  ENTRY(cuMemFree)
  ENTRY(cuMemcpy2DUnaligned)
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
    return cuDeviceGetAttribute && cuLaunchCooperativeKernel &&
           cuCtxSynchronize && cuCtxPopCurrent && cuCtxPushCurrent &&
           cuDevicePrimaryCtxRelease && cuDevicePrimaryCtxRetain &&
           cuEventCreate && cuEventDestroy && cuEventElapsedTime &&
           cuEventRecordWithFlags && cuEventSynchronize && cuFuncSetAttribute &&
           cuGetErrorName && cuGraphDestroy && cuGraphExecDestroy &&
           cuGraphInstantiateWithFlags && cuGraphLaunch && cuLaunchKernel &&
           cuMemAlloc && cuMemFree && cuMemcpy2DUnaligned && cuMemcpyDtoH && cuMemcpyHtoD &&
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
  int sms = 1;
  size_t fields = 0;
  bool cartesian = false;
  CUcontext context{};
  CUstream stream{};
  static constexpr int iterations = 9;
  CUevent boundaries[2 * iterations + 1]{};
  CUdeviceptr positions{}, edgesBox{}, raw{}, box{}, charge{}, types{}, map{},
      counts{}, rebuildStatus{}, geometry{}, keys{}, local{}, packed{},
      packedTypes{}, ids{}, inverse{}, spans{}, state{}, failed{}, words{},
      low{}, high{}, bonds{}, index{}, arrivals{}, virial{}, output{},
      retries{};
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
    TowersCell cell(system);
    cartesian = !cell.triclinic && cells <= atoms;
    std::array<double, 6> cellEdges{system.box[0],  system.box[1],
                                    system.box[2],  system.tilt[0],
                                    system.tilt[1], system.tilt[2]};
    std::vector<std::pair<unsigned, unsigned>> pairs;
    if (system.topology)
      pairs = system.topology->exclusions;
    for (auto &[i, j] : pairs)
      if (i > j)
        std::swap(i, j);
    llvm::sort(pairs);
    std::vector<int32_t> edges;
    edges.reserve(2 * pairs.size());
    std::vector<uint32_t> indexRows(2ULL * atoms);
    for (auto [i, j] : pairs) {
      if (j > i && j - i <= 32)
        indexRows[2ULL * i] |= 1u << (j - i - 1);
      indexRows[2ULL * i + 1] = std::max(indexRows[2ULL * i + 1], j);
      edges.push_back(i);
      edges.push_back(j);
    }
    size_t n = atoms, c = cells, slots = c * capacity + n;
    size_t entries = c * (5 + 2 * reach);
    fields = buffers.fields;
    for (auto &event : boundaries)
      if (!check(api().cuEventCreate(&event, 0), "create trial boundary"))
        return false;
    int cooperative = 0;
    if (!check(
            api().cuDeviceGetAttribute(
                &cooperative, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device),
            "query cooperative launch support") ||
        (cooperative &&
         !check(api().cuDeviceGetAttribute(
                    &sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
                "query cooperative grid size")) ||
        !check(api().cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
               "cuStreamCreate") ||
        !allocate(retries, 2 * c * sizeof(int64_t)) ||
        !allocate(positions, 24 * n, system.positions.data()) ||
        !allocate(edgesBox, cell.triclinic ? 48 : 24, cellEdges.data()) ||
        !allocate(raw, 24 * n) ||
        !allocate(box, cell.triclinic ? 72 : 24,
                  cell.triclinic ? cell.lattice.data() : system.box) ||
        !allocate(map, 4 * entries) || !allocate(counts, 4 * (2 * c + 1)) ||
        !allocate(rebuildStatus, 4) ||
        !allocate(geometry, 24 * (cell.triclinic ? 13 : 3)) ||
        !allocate(charge, 4 * n) || !allocate(types, 4 * n) ||
        !allocate(keys, 4 * (slots + n)) || !allocate(local, 16 * n) ||
        !allocate(packed, 16 * slots) || !allocate(packedTypes, 4 * slots) ||
        !allocate(ids, 4 * slots) || !allocate(inverse, 8 * n) ||
        !allocate(spans, 8 * entries) || !allocate(state, 8) ||
        !allocate(failed, 8) || !allocate(words, 24 * n) ||
        !allocate(low, 24 * n) || !allocate(high, 24 * n) ||
        !allocate(arrivals, 4) || !allocate(virial, 72) ||
        !allocate(output, 24 * n))
      return false;
    if (exclusions && (!allocate(bonds, 8ULL * exclusions, edges.data()) ||
                       !allocate(index, 8 * n, indexRows.data())))
      return false;
    parameters.resize(buffers.values.size());
    for (size_t i = 0; i < parameters.size(); ++i)
      if (!allocate(parameters[i], buffers.values[i].size(),
                    buffers.values[i].data()))
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
  enum Kernel {
    Setup,
    Inputs,
    Insert,
    Pack,
    Force,
    Finish,
    CountRetries,
    Size
  };
  CUfunction fn[Size]{};
  int threads, shared, split;
  bool capturing = false, pressure = false;
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
    std::vector<void *> args{
        &f.positions,     &f.edgesBox, &f.raw,   &f.box,    &f.geometry,
        &f.rebuildStatus, &f.charge,   &f.types, &f.counts, &f.keys,
        &f.local,         &f.words,    &f.state, &f.failed, &f.arrivals};
    for (size_t i = 0; i < f.fields; ++i)
      args.push_back(&f.parameters[i]);
    if (!launch(Inputs, (f.atoms + 127) / 128, 128, args))
      return false;
    if (f.cells > f.atoms &&
        (!launch(Setup, (f.cells + 127) / 128, 128,
                 {&f.map, &f.counts, &f.rebuildStatus, &f.box, &f.geometry}) ||
         !launch(Insert, (f.atoms + 127) / 128, 128,
                 {&f.raw, &f.charge, &f.geometry, &f.counts, &f.keys, &f.local,
                  &forward})))
      return false;
    return launch(Pack, (f.cells + 3) / 4, 128,
                  {&f.counts, &f.keys, &f.local, &f.types, &f.map, &f.packed,
                   &f.packedTypes, &f.ids, &f.inverse, &f.spans,
                   &f.rebuildStatus});
  }
  bool pair(bool count = false) {
    float scale = 262144;
    std::vector<void *> args{&f.spans, &f.packed, &f.ids, &f.geometry,
                             &f.words, &f.state,  &scale, &f.failed};
    if (f.exclusions)
      args.insert(args.end(),
                  {&f.bonds, &f.inverse, f.cartesian ? &f.positions : &f.raw,
                   &f.box, &f.index});
    for (size_t i = 0; i < f.fields; ++i)
      args.push_back(&f.parameters[i]);
    args.push_back(&f.packedTypes);
    for (size_t i = f.fields; i < f.parameters.size(); ++i)
      args.push_back(&f.parameters[i]);
    auto n = f.atoms;
    auto blocks =
        kernels::towersForceBlocks(f.cells, split, f.exclusions, threads);
    if (pressure) {
      args.push_back(&f.virial);
      if (!f.check(api().cuMemsetD8Async(f.virial, 0, 72, f.stream),
                   "clear virial"))
        return false;
    }
    if (count) {
      args.push_back(&f.retries);
      return launch(CountRetries, blocks, threads, args, shared);
    }
    if (!launch(Force, blocks, threads, args, shared))
      return false;
    if (pressure)
      args.pop_back();
    args[4] = &f.low;
    args.insert(args.end(), {&f.high, &f.words, &f.arrivals, &f.output});
    int grid =
        std::min<int64_t>(f.sms, kernels::towersFinishBlocks(n, threads));
    if (grid == 1)
      return launch(Finish, 1, threads, args, shared);
    return f.check(api().cuLaunchCooperativeKernel(fn[Finish], grid, 1, 1,
                                                   threads, 1, 1, shared,
                                                   f.stream, args.data()),
                   "launch towers_finish");
  }
  bool prepare(const void *image, Kernel end) {
    const char *names[] = {"towers_setup",
                           "towers_trial_inputs",
                           "towers_insert",
                           "towers_pack",
                           pressure ? "towers_force_virial" : "towers_force",
                           "towers_finish",
                           "towers_count_stripe_retries"};
    if (!f.check(api().cuModuleLoadData(&module, image),
                 "load trial specialization"))
      return false;
    for (int k = 0; k < end; ++k)
      if (!f.check(api().cuModuleGetFunction(&fn[k], module, names[k]),
                   names[k]))
        return false;
    for (int k = Force; k < end; ++k)
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
    return true;
  }
  bool measure(const void *image, bool canRetry, TowersTiming &timing) {
    if (!prepare(image, canRetry ? Size : CountRetries))
      return false;
    if (!f.check(api().cuStreamBeginCapture(
                     f.stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
                 "begin trial graph"))
      return false;
    capturing = true;
    auto mark = [&](int i) {
      return f.check(api().cuEventRecordWithFlags(f.boundaries[i], f.stream,
                                                  CU_EVENT_RECORD_EXTERNAL),
                     "record trial boundary");
    };
    if (!mark(0))
      return false;
    for (int i = 0; i < f.iterations; ++i)
      if (!rebuild() || !mark(2 * i + 1) || !pair() || !mark(2 * i + 2))
        return false;
    capturing = false;
    if (!f.check(api().cuStreamEndCapture(f.stream, &graph),
                 "end trial graph") ||
        !f.check(api().cuGraphInstantiateWithFlags(&exec, graph, 0),
                 "instantiate trial graph") ||
        !f.check(api().cuGraphLaunch(exec, f.stream), "warm trial graph") ||
        !f.check(api().cuGraphLaunch(exec, f.stream), "launch trial graph") ||
        !f.check(api().cuStreamSynchronize(f.stream), "wait for trial"))
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
    for (int i = 0; i < 2 * f.iterations; ++i) {
      float ms;
      if (!f.check(api().cuEventElapsedTime(&ms, f.boundaries[i],
                                            f.boundaries[i + 1]),
                   "read trial interval"))
        return false;
      (i % 2 ? timing.pairUs : timing.rebuildUs) +=
          ms * (1000.0f / f.iterations);
    }
    if (canRetry) {
      std::vector<int64_t> counts(size_t(f.cells) * split);
      size_t bytes = counts.size() * sizeof(int64_t);
      if (!f.check(api().cuMemsetD8Async(f.retries, 0, bytes, f.stream),
                   "clear retry counts") ||
          !rebuild() || !pair(true) ||
          !f.check(api().cuStreamSynchronize(f.stream),
                   "wait for retry count") ||
          !f.check(api().cuMemcpyDtoH(counts.data(), f.retries, bytes),
                   "read stripe retries"))
        return false;
      for (auto count : counts)
        timing.stripeRetries += count;
    }
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

llvm::Expected<TowersPopulation> readTowersPopulation(TowersTrial &f,
                                                      const std::string &image) {
  if (!f.check(api().cuCtxPushCurrent(f.context), "enter trial context"))
    return llvm::createStringError("towers_shape: %s", f.error.c_str());
  auto restore = llvm::scope_exit([] {
    CUcontext old;
    api().cuCtxPopCurrent(&old);
  });
  Sample sample{f};
  if (!sample.prepare(image.data(), Sample::Force))
    return llvm::createStringError("towers_shape: %s", f.error.c_str());
  // Field zero is the owner's own cell. Packing clears the insertion counts
  // and publishes the full population in the second word of each span.
  std::vector<int> counts(f.cells);
  CUDA_MEMCPY2D copy{};
  copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.srcDevice = f.spans + sizeof(int);
  copy.srcPitch = 2 * sizeof(int) * (5 + 2 * f.reach);
  copy.dstMemoryType = CU_MEMORYTYPE_HOST;
  copy.dstHost = counts.data();
  copy.dstPitch = copy.WidthInBytes = sizeof(int);
  copy.Height = f.cells;
  if (!f.check(api().cuMemcpy2DUnaligned(&copy), "read setup cell populations"))
    return llvm::createStringError("towers_shape: %s", f.error.c_str());
  int64_t total = 0;
  for (int count : counts)
    total += count;
  if (total != f.atoms)
    return llvm::createStringError("towers_shape: incomplete setup population");
  return TowersPopulation{*std::max_element(counts.begin(), counts.end()),
                          double(total) / f.cells};
}

llvm::Expected<TowersTiming>
measureTowersTrial(TowersTrial &f, const std::string &image,
                   const driver::TowersShape &shape, bool pressure) {
  if (!f.check(api().cuCtxPushCurrent(f.context), "enter trial context"))
    return llvm::createStringError("towers_shape: %s", f.error.c_str());
  auto restore = llvm::scope_exit([] {
    CUcontext old;
    api().cuCtxPopCurrent(&old);
  });
  Sample sample{f};
  sample.pressure = pressure;
  sample.threads = shape.threads;
  sample.shared = 8 * shape.records;
  sample.split = shape.split;
  TowersTiming us;
  if (!sample.measure(image.data(),
                      shape.records < shape.stripe * (shape.panel / 32), us))
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
      for (auto event : f->boundaries)
        if (event)
          api().cuEventDestroy(event);
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
