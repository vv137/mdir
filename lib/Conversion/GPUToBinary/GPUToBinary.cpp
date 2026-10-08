//===- GPUToBinary.cpp - Serialize GPU modules in parallel ----------------===//
//
// `mdir-gpu-module-to-binary` does what upstream `gpu-module-to-binary`
// (mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp) does, one
// `gpu.binary` for each `gpu.module`, but serializes the modules on the
// threads of the context, takes their PTX and cubins from the compile
// cache, and runs ptxas itself (D214).
// `mdir-gpu-lower-to-nvvm` is upstream `gpu-lower-to-nvvm-pipeline`
// (mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp), pass for pass,
// with this pass in place of upstream's serialization.
//
//===----------------------------------------------------------------------===//

#include "mdir/Conversion/GPUToBinary.h"
#include "mdir/Conversion/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVM/NVVM/Utils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/Targets.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Triple.h"
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>
#include <chrono>
#include <limits>
#include <map>
#include <mutex>

namespace mdir {
#define GEN_PASS_DEF_GPUMODULETOBINARY
#include "mdir/Conversion/Passes.h.inc"
} // namespace mdir

using namespace mlir;
using mdir::compiler::CompileCacheConfig;
using mdir::compiler::CompileStats;

namespace {

double now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string hashText(StringRef text) {
  return llvm::toHex(llvm::BLAKE3::hash(llvm::arrayRefFromStringRef(text)),
                     /*LowerCase=*/true);
}

//===----------------------------------------------------------------------===//
// What the pass did, per context
//===----------------------------------------------------------------------===//

std::mutex &getStatsMutex() {
  static std::mutex mutex;
  return mutex;
}
llvm::DenseMap<MLIRContext *, CompileStats> &getStatsTable() {
  static llvm::DenseMap<MLIRContext *, CompileStats> table;
  return table;
}

//===----------------------------------------------------------------------===//
// Files and tools, once per process
//===----------------------------------------------------------------------===//

/// The BLAKE3 of the file at `path`, read once per process for each size
/// and time of modification; nullopt if it cannot be read.
std::optional<std::string> hashFile(StringRef path) {
  static std::mutex mutex;
  static std::map<std::string, std::string> hashes;
  llvm::sys::fs::file_status status;
  if (llvm::sys::fs::status(path, status))
    return std::nullopt;
  std::string name =
      (path + "\n" + Twine(status.getSize()) + "\n" +
       Twine(status.getLastModificationTime().time_since_epoch().count()))
          .str();
  std::lock_guard<std::mutex> lock(mutex);
  if (auto found = hashes.find(name); found != hashes.end())
    return found->second;
  auto file = llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                          /*RequiresNullTerminator=*/false);
  if (!file)
    return std::nullopt;
  return hashes[name] = hashText((*file)->getBuffer());
}

/// Runs `program` with `arguments`; its standard output and error go to
/// `log`. Returns whether it exited with 0.
bool runTool(StringRef program, ArrayRef<StringRef> arguments,
             std::string &log) {
  SmallString<128> logPath;
  if (llvm::sys::fs::createTemporaryFile("mdir-ptxas", "log", logPath))
    return false;
  llvm::FileRemover removeLog(logPath);
  std::optional<StringRef> redirects[] = {std::nullopt, StringRef(logPath),
                                          StringRef(logPath)};
  std::string message;
  int status = llvm::sys::ExecuteAndWait(program, arguments, std::nullopt,
                                         redirects, 0, 0, &message);
  if (auto text = llvm::MemoryBuffer::getFile(logPath))
    log = (*text)->getBuffer().str();
  if (!message.empty())
    log += message;
  return status == 0;
}

/// The ptxas that compiles the PTX of the modules: that of the toolkit
/// whose libdevice the modules link (CUDA_ROOT, CUDA_HOME, or CUDA_PATH,
/// else the toolkit of the build); else that of NVIDIA's wheel
/// `nvidia-cuda-nvcc` in the site-packages that holds the Python package
/// (D228); else the one on PATH; and the text of its
/// `--version`, which is part of the key of a cubin. Empty if there is none
/// that runs.
struct Ptxas {
  std::string path;
  std::string version;
};
/// `site-packages/nvidia/cu13/bin/ptxas` beside the package `mdir` whose
/// extension holds this code (`site-packages/mdir/_core*.so`); empty when
/// this code is not in such a package or the wheel is not installed.
static std::string findWheelPtxas() {
  Dl_info info;
  if (!dladdr(reinterpret_cast<void *>(&findWheelPtxas), &info) ||
      !info.dli_fname)
    return "";
  SmallString<256> path(llvm::sys::path::parent_path(
      llvm::sys::path::parent_path(info.dli_fname)));
  llvm::sys::path::append(path, "nvidia", "cu13", "bin", "ptxas");
  return llvm::sys::fs::can_execute(path) ? std::string(path) : "";
}
const Ptxas &getPtxas() {
  static std::mutex mutex;
  static std::map<std::string, Ptxas> found;
  std::string toolkit = NVVM::getCUDAToolkitPath().str();
  std::lock_guard<std::mutex> lock(mutex);
  auto [entry, inserted] = found.try_emplace(toolkit);
  if (!inserted)
    return entry->second;
  SmallString<256> path(toolkit);
  llvm::sys::path::append(path, "bin", "ptxas");
  std::string program;
  if (!toolkit.empty() && llvm::sys::fs::can_execute(path))
    program = std::string(path);
  else if (std::string wheel = findWheelPtxas(); !wheel.empty())
    program = wheel;
  else if (auto onPath = llvm::sys::findProgramByName("ptxas"))
    program = *onPath;
  std::string log;
  if (!program.empty() && runTool(program, {"ptxas", "--version"}, log))
    entry->second = Ptxas{program, log};
  return entry->second;
}

/// Architectures that ptxas could not compile for in this process: their
/// modules stay PTX, which the driver compiles at load.
struct Unsupported {
  std::mutex mutex;
  std::map<std::string, bool> chips;
  bool contains(StringRef chip) {
    std::lock_guard<std::mutex> lock(mutex);
    return chips.count(chip.str());
  }
  /// Records `chip`; returns whether it is new.
  bool add(StringRef chip) {
    std::lock_guard<std::mutex> lock(mutex);
    return chips.emplace(chip.str(), true).second;
  }
};
Unsupported &getUnsupported() {
  static Unsupported unsupported;
  return unsupported;
}

/// At most this many ptxas run at once in a process. Beyond it, starting
/// processes costs the kernel more than it gains: lowering the dipeptide
/// (253 modules) on 128 threads of a 128-core host took 4.3 s wall either
/// way, with 17.6 s of system time unbounded and 4.0 s at 32.
constexpr int kPtxasSlots = 32;
class Slots {
public:
  void acquire() {
    std::unique_lock<std::mutex> lock(mutex);
    free.wait(lock, [&] { return used < kPtxasSlots; });
    ++used;
  }
  void release() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      --used;
    }
    free.notify_one();
  }

private:
  std::mutex mutex;
  std::condition_variable free;
  int used = 0;
};
Slots &getPtxasSlots() {
  static Slots slots;
  return slots;
}

/// Compiles `ptx` to a cubin for `chip` with ptxas, as upstream's
/// NVPTXSerializer::compileToBinary does, with its arguments
/// (`-arch <chip> --opt-level <O>`). Returns the cubin, or nullopt with the
/// log of ptxas.
std::optional<std::string> compileCubin(const Ptxas &ptxas, StringRef ptx,
                                        ArrayRef<std::string> arguments,
                                        std::string &log) {
  SmallString<128> input, output;
  if (llvm::sys::fs::createTemporaryFile("mdir-kernels", "ptx", input)) {
    log = "cannot create a temporary file for the PTX";
    return std::nullopt;
  }
  llvm::FileRemover removeInput(input);
  if (llvm::sys::fs::createTemporaryFile("mdir-kernels", "cubin", output)) {
    log = "cannot create a temporary file for the cubin";
    return std::nullopt;
  }
  llvm::FileRemover removeOutput(output);
  {
    std::error_code error;
    llvm::raw_fd_ostream stream(input, error);
    if (error) {
      log = error.message();
      return std::nullopt;
    }
    stream << ptx;
  }
  SmallVector<StringRef> argv = {"ptxas"};
  for (const std::string &argument : arguments)
    argv.push_back(argument);
  argv.append({input, "-o", output});
  getPtxasSlots().acquire();
  bool ran = runTool(ptxas.path, argv, log);
  getPtxasSlots().release();
  if (!ran)
    return std::nullopt;
  auto cubin = llvm::MemoryBuffer::getFile(output, /*IsText=*/false,
                                           /*RequiresNullTerminator=*/false);
  if (!cubin) {
    log = "ptxas wrote no cubin";
    return std::nullopt;
  }
  return (*cubin)->getBuffer().str();
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

/// The format tags of the keys; a change of what a key holds changes them.
constexpr StringLiteral kPtxFormat = "mdir-gpu-ptx-cache 1";
constexpr StringLiteral kCubinFormat = "mdir-gpu-cubin-cache 1";

bool isPtx(StringRef data) {
  return data.contains(".version") && data.contains(".target");
}
bool isCubin(StringRef data) { return data.starts_with("\x7f" "ELF"); }

/// The counters of one run of the pass, added up across threads.
struct Counters {
  std::atomic<unsigned> ptxCompiled{0}, ptxHits{0}, cubinCompiled{0},
      cubinHits{0}, rejected{0}, stored{0}, unstored{0};
  /// The bytes of the entries stored.
  std::atomic<uint64_t> storedBytes{0};
  std::mutex mutex;
  double compileSeconds = 0.0, savedSeconds = 0.0, lookupSeconds = 0.0;
  void addTimes(double compile, double saved, double lookup) {
    std::lock_guard<std::mutex> lock(mutex);
    compileSeconds += compile;
    savedSeconds += saved;
    lookupSeconds += lookup;
  }
};

/// The GPU entries of the compile cache (D212): `<directory>/gpu/`.
class GpuCache {
public:
  explicit GpuCache(std::optional<CompileCacheConfig> config)
      : config(std::move(config)) {
    if (this->config) {
      SmallString<256> path(this->config->directory);
      llvm::sys::path::append(path, "gpu");
      directory = std::string(path);
    }
  }
  bool enabled() const { return config.has_value(); }

  /// The data of the entry of `key`, or nullopt; `saved` is the time its
  /// generation took.
  std::optional<std::string> read(StringRef key, StringRef extension,
                                  bool (*isValid)(StringRef), double &saved,
                                  Counters &counters) {
    std::string path = getPath(key, extension);
    bool rejected = false;
    auto entry = mdir::compiler::readCacheEntry(path, key, saved, rejected,
                                                isValid);
    if (rejected)
      ++counters.rejected;
    if (!entry)
      return std::nullopt;
    mdir::compiler::touchCacheEntry(path);
    return entry->getBuffer().str();
  }

  void write(StringRef key, StringRef extension, StringRef data,
             double seconds, Counters &counters) {
    // A cache that cannot be written is no cache; the run goes on.
    std::call_once(made, [&] {
      writable = !llvm::sys::fs::create_directories(directory);
    });
    uint64_t written = 0;
    if (writable &&
        !mdir::compiler::writeCacheEntry(getPath(key, extension), key, data,
                                         seconds, &written)) {
      ++counters.stored;
      counters.storedBytes += written;
      return;
    }
    // writeCacheEntry's error, if any, is dropped with the entry.
    ++counters.unstored;
  }

  /// Adds what one run of the pass stored to the cache's total, once for
  /// all its modules; the directory is listed only when the bound may be
  /// exceeded (D[compile-cache-size-file]).
  void noteStores(uint64_t bytes) {
    if (config)
      mdir::compiler::noteCacheStores(config->directory, bytes,
                                      config->maxBytes);
  }

private:
  std::string getPath(StringRef key, StringRef extension) const {
    SmallString<256> path(directory);
    llvm::sys::path::append(path,
                            mdir::compiler::getCacheEntryName(key, extension));
    return std::string(path);
  }
  std::optional<CompileCacheConfig> config;
  std::string directory;
  std::once_flag made;
  bool writable = false;
};

/// The key of the PTX of `module` for `target`: a hash of the module's IR,
/// the target, the libraries it links (libdevice), and the LLVM version.
/// The MDIR build is not part of it: the module holds only upstream ops of
/// the LLVM and NVVM dialects. Nullopt if the module cannot be cached.
std::optional<std::string> getPtxKey(gpu::GPUModuleOp module,
                                     NVVM::NVVMTargetAttr target,
                                     StringRef binary) {
  std::string text;
  llvm::raw_string_ostream os(text);
  // The generic form with every attribute in full: nothing elided, no
  // aliases, and no locations, which do not reach the PTX.
  OpPrintingFlags flags;
  flags.printGenericOpForm()
      .useLocalScope()
      .assumeVerified()
      .enableDebugInfo(false)
      .elideLargeElementsAttrs(std::numeric_limits<int64_t>::max())
      .elideLargeResourceString(std::numeric_limits<int64_t>::max());
  module->print(os, flags);
  // A blob of a resource is printed by name, not by content.
  if (StringRef(text).contains("dense_resource<"))
    return std::nullopt;
  std::string key;
  llvm::raw_string_ostream k(key);
  k << kPtxFormat << '\n'
    << "llvm " << LLVM_VERSION_STRING << '\n'
    << "target " << target << '\n'
    << "binary " << binary << '\n';
  SmallString<256> libdevice(NVVM::getCUDAToolkitPath());
  llvm::sys::path::append(libdevice, "nvvm", "libdevice", "libdevice.10.bc");
  auto hash = hashFile(libdevice);
  if (!hash)
    return std::nullopt;
  k << "libdevice " << *hash << '\n';
  if (ArrayAttr links = target.getLink())
    for (Attribute link : links) {
      auto path = dyn_cast<StringAttr>(link);
      std::optional<std::string> linked;
      if (path)
        linked = hashFile(path.getValue());
      if (!linked)
        return std::nullopt;
      k << "link " << *linked << '\n';
    }
  k << "module " << hashText(text) << '\n';
  return key;
}

class GpuModuleToBinary
    : public mdir::impl::GpuModuleToBinaryBase<GpuModuleToBinary> {
public:
  using Base::Base;
  void runOnOperation() final;
};

void GpuModuleToBinary::runOnOperation() {
  double started = now();
  auto format = llvm::StringSwitch<std::optional<gpu::CompilationTarget>>(
                    this->format.getValue())
                    .Cases({"assembly", "isa"}, gpu::CompilationTarget::Assembly)
                    .Cases({"binary", "bin"}, gpu::CompilationTarget::Binary)
                    .Default(std::nullopt);
  if (!format) {
    getOperation()->emitError() << "unknown format '" << this->format
                                << "' (expected isa or bin)";
    return signalPassFailure();
  }
  if (binary != "auto" && binary != "cubin" && binary != "ptx") {
    getOperation()->emitError() << "unknown binary '" << binary
                                << "' (expected auto, cubin, or ptx)";
    return signalPassFailure();
  }
  // Under cubin, a module that cannot become a cubin is an error.
  bool required = binary == "cubin";
  // The modules in the order of the blocks that hold them, as upstream.
  SmallVector<gpu::GPUModuleOp> modules;
  for (Region &region : getOperation()->getRegions())
    for (Block &block : region.getBlocks())
      for (auto module : block.getOps<gpu::GPUModuleOp>())
        modules.push_back(module);

  // The symbol table is built before the threads start; upstream builds it
  // lazily, which is not safe on several threads.
  SymbolTable table(getOperation());
  auto getTable = [&]() -> SymbolTable * { return &table; };
  gpu::TargetOptions ptxOptions({}, {}, {}, {},
                                gpu::CompilationTarget::Assembly, getTable);
  gpu::TargetOptions cubinOptions({}, {}, {}, {},
                                  gpu::CompilationTarget::Binary, getTable);
  const Ptxas *ptxas = nullptr;
  if (*format == gpu::CompilationTarget::Binary) {
    ptxas = &getPtxas();
    if (ptxas->path.empty() && required) {
      getOperation()->emitError()
          << "MDIR_GPU_BINARY=cubin, but no ptxas was found in the CUDA "
             "toolkit (CUDA_ROOT, CUDA_HOME, CUDA_PATH) or on PATH";
      return signalPassFailure();
    }
    if (ptxas->path.empty()) {
      static std::once_flag warned;
      std::call_once(warned, [] {
        llvm::errs() << "mdir: no ptxas was found in the CUDA toolkit or on "
                        "PATH; the kernels are loaded as PTX, which the "
                        "driver compiles\n";
      });
      ptxas = nullptr;
    }
  }
  // Without `cache`, no entry is read or written (D217).
  GpuCache cache(this->cache ? CompileCacheConfig::fromEnvironment()
                             : std::nullopt);
  Counters counters;

  // Each module is translated to LLVM IR in an LLVM context of its own and
  // read, not changed, so modules are serialized side by side. The
  // diagnostics are ordered by module.
  std::vector<SmallVector<Attribute>> results(modules.size());
  LogicalResult serialized = failableParallelForEachN(
      &getContext(), 0, modules.size(), [&](size_t i) -> LogicalResult {
        gpu::GPUModuleOp module = modules[i];
        if (!module.getTargetsAttr())
          return module.emitError("the module has no target attributes");
        for (Attribute attribute : module.getTargetsAttr()) {
          auto target = cast<gpu::TargetAttrInterface>(attribute);
          auto nvvm = dyn_cast<NVVM::NVVMTargetAttr>(attribute);
          double compile = 0.0, saved = 0.0, lookup = 0.0;
          // The PTX: from the cache, or generated by LLVM.
          double begin = now();
          std::optional<std::string> key;
          if (nvvm && cache.enabled())
            key = getPtxKey(module, nvvm, binary);
          std::optional<std::string> ptx;
          if (key) {
            double seconds = 0.0;
            ptx = cache.read(*key, ".ptx", isPtx, seconds, counters);
            if (ptx) {
              ++counters.ptxHits;
              saved += seconds;
            }
          }
          lookup += now() - begin;
          if (!ptx) {
            double generating = now();
            std::optional<gpu::SerializedObject> object =
                target.serializeToObject(module, ptxOptions);
            if (!object)
              return module.emitError(
                  "An error happened while serializing the module.");
            const SmallVector<char, 0> &bytes = object->getObject();
            ptx = std::string(bytes.begin(), bytes.end());
            double seconds = now() - generating;
            compile += seconds;
            ++counters.ptxCompiled;
            if (key)
              cache.write(*key, ".ptx", *ptx, seconds, counters);
          }

          // The cubin, for an NVVM target when ptxas is found and takes
          // the architecture: from the cache, or generated by ptxas.
          std::optional<std::string> cubin;
          if (ptxas && nvvm &&
              (required || !getUnsupported().contains(nvvm.getChip()))) {
            SmallVector<std::string> arguments = {
                "-arch", nvvm.getChip().str(), "--opt-level",
                std::to_string(nvvm.getO())};
            std::string cubinKey;
            llvm::raw_string_ostream k(cubinKey);
            k << kCubinFormat << '\n'
              << "ptxas " << ptxas->version << '\n'
              << "binary " << binary << '\n'
              << "arguments " << llvm::join(arguments, " ") << '\n'
              << "ptx " << hashText(*ptx) << '\n';
            begin = now();
            if (cache.enabled()) {
              double seconds = 0.0;
              cubin = cache.read(cubinKey, ".cubin", isCubin, seconds,
                                 counters);
              if (cubin) {
                ++counters.cubinHits;
                saved += seconds;
              }
            }
            lookup += now() - begin;
            if (!cubin) {
              double generating = now();
              std::string log;
              cubin = compileCubin(*ptxas, *ptx, arguments, log);
              if (!cubin && required)
                return module.emitError()
                       << "MDIR_GPU_BINARY=cubin, but ptxas could not compile "
                          "the module for "
                       << nvvm.getChip() << ":\n"
                       << log;
              if (!cubin) {
                // An architecture that this ptxas does not take, or a
                // failure of ptxas: the module stays PTX.
                if (getUnsupported().add(nvvm.getChip()))
                  llvm::errs() << "mdir: ptxas could not compile the kernels "
                                  "for "
                               << nvvm.getChip()
                               << "; they are loaded as PTX, which the "
                                  "driver compiles. The log of ptxas:\n"
                               << log << "\n";
              } else {
                double seconds = now() - generating;
                compile += seconds;
                ++counters.cubinCompiled;
                if (cache.enabled())
                  cache.write(cubinKey, ".cubin", *cubin, seconds, counters);
              }
            }
          }
          counters.addTimes(compile, saved, lookup);

          // Without the times of the serialization, which upstream records
          // as properties: they would differ from run to run.
          StringRef bytes = cubin ? *cubin : *ptx;
          gpu::SerializedObject object(
              SmallVector<char, 0>(bytes.begin(), bytes.end()),
              DictionaryAttr::get(&getContext()));
          Attribute created = target.createObject(
              module, object, cubin ? cubinOptions : ptxOptions);
          if (!created)
            return module.emitError(
                "An error happened while creating the object.");
          results[i].push_back(created);
        }
        return success();
      });
  if (failed(serialized))
    return signalPassFailure();

  // The binaries replace the modules on this thread, in their order.
  OpBuilder builder(&getContext());
  for (auto [module, objects] : llvm::zip_equal(modules, results)) {
    builder.setInsertionPointAfter(module);
    auto handler =
        dyn_cast_or_null<gpu::OffloadingLLVMTranslationAttrInterface>(
            module.getOffloadingHandlerAttr());
    gpu::BinaryOp::create(builder, module.getLoc(), module.getName(), handler,
                          builder.getArrayAttr(objects));
    module->erase();
  }
  if (counters.stored)
    cache.noteStores(counters.storedBytes);

  std::lock_guard<std::mutex> lock(getStatsMutex());
  CompileStats &stats = getStatsTable()[&getContext()];
  stats.gpuModules += modules.size();
  stats.gpuSerializeSeconds += now() - started;
  stats.gpuPtxCompiled += counters.ptxCompiled;
  stats.gpuPtxHits += counters.ptxHits;
  stats.gpuCubinCompiled += counters.cubinCompiled;
  stats.gpuCubinHits += counters.cubinHits;
  stats.gpuCompileSeconds += counters.compileSeconds;
  stats.gpuSavedSeconds += counters.savedSeconds;
  stats.gpuLookupSeconds += counters.lookupSeconds;
  stats.gpuRejected += counters.rejected;
  stats.gpuStored += counters.stored;
  stats.gpuUnstored += counters.unstored;
}

/// Upstream's buildLowerToNVVMPassPipeline, with mdir-gpu-module-to-binary.
/// The options of upstream's pipeline, MDIR_GPU_BINARY, and whether the
/// compile cache is used.
struct GpuLowerToNVVMOptions : public gpu::GPUToNVVMPipelineOptions {
  PassOptions::Option<std::string> binary{
      *this, "binary",
      llvm::cl::desc("auto, cubin, or ptx (MDIR_GPU_BINARY)"),
      llvm::cl::init("auto")};
  PassOptions::Option<bool> cache{
      *this, "cache",
      llvm::cl::desc("use the compile cache that the environment names; "
                     "false bypasses it (D217)"),
      llvm::cl::init(true)};
};

void buildGpuLowerToNVVM(OpPassManager &pm,
                         const GpuLowerToNVVMOptions &options) {
  // The common part.
  pm.addPass(createConvertNVGPUToNVVMPass());
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createConvertVectorToSCFPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertNVVMToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  GpuNVVMAttachTargetOptions target;
  target.triple = options.cubinTriple;
  target.chip = options.cubinChip;
  target.features = options.cubinFeatures;
  target.optLevel = options.optLevel;
  target.cmdOptions = options.cmdOptions;
  pm.addPass(createGpuNVVMAttachTarget(target));
  pm.addPass(createLowerAffinePass());
  pm.addPass(createArithToLLVMConversionPass());
  ConvertIndexToLLVMPassOptions index;
  index.indexBitwidth = options.indexBitWidth;
  pm.addPass(createConvertIndexToLLVMPass(index));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // The GPU modules.
  ConvertGpuOpsToNVVMOpsOptions nvvm;
  nvvm.useBarePtrCallConv = options.kernelUseBarePtrCallConv;
  nvvm.indexBitwidth = options.indexBitWidth;
  nvvm.allowPatternRollback = options.allowPatternRollback;
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToNVVMOps(nvvm));
  pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());

  // The host, after the GPU modules.
  GpuToLLVMConversionPassOptions host;
  host.hostBarePtrCallConv = options.hostUseBarePtrCallConv;
  host.kernelBarePtrCallConv = options.kernelUseBarePtrCallConv;
  pm.addPass(createGpuToLLVMConversionPass(host));
  mdir::GpuModuleToBinaryOptions binary;
  binary.format = options.cubinFormat;
  binary.binary = options.binary;
  binary.cache = options.cache;
  pm.addPass(mdir::createGpuModuleToBinary(binary));
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

} // namespace

#if LLVM_HAS_NVPTX_TARGET
extern "C" void LLVMInitializeNVPTXTargetInfo();
extern "C" void LLVMInitializeNVPTXTarget();
extern "C" void LLVMInitializeNVPTXTargetMC();
#endif

/// Whether LLVM's NVPTX backend knows `chip`.
static bool isKnownChip(StringRef chip) {
#if LLVM_HAS_NVPTX_TARGET
  static std::once_flag initialized;
  std::call_once(initialized, [] {
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
  });
  llvm::Triple triple("nvptx64-nvidia-cuda");
  std::string error;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target)
    return false;
  std::unique_ptr<llvm::MCSubtargetInfo> info(
      target->createMCSubtargetInfo(triple, chip, ""));
  return info && info->isCPUStringValid(chip);
#else
  return false;
#endif
}

/// The compute capability of a visible device, from NVML, which needs no
/// CUDA context, so that a process may fork after it compiles. The
/// device is the `device`-th entry of CUDA_VISIBLE_DEVICES, or CUDA's
/// device `device` without it. An entry `GPU-<uuid>` names its device. An
/// index is CUDA's, whose order is NVML's under CUDA_DEVICE_ORDER=PCI_BUS_ID
/// and the fastest first otherwise; in that case the architecture is known
/// only when every device has the same. Empty when it cannot be told.
static std::string queryChip(int64_t device) {
  void *nvml = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!nvml)
    return "";
  using Init = int (*)();
  using Count = int (*)(unsigned *);
  using Handle = int (*)(unsigned, void **);
  using Uuid = int (*)(void *, char *, unsigned);
  using Capability = int (*)(void *, int *, int *);
  auto init = reinterpret_cast<Init>(dlsym(nvml, "nvmlInit_v2"));
  auto shutdown = reinterpret_cast<Init>(dlsym(nvml, "nvmlShutdown"));
  auto count = reinterpret_cast<Count>(dlsym(nvml, "nvmlDeviceGetCount_v2"));
  auto handle =
      reinterpret_cast<Handle>(dlsym(nvml, "nvmlDeviceGetHandleByIndex_v2"));
  auto uuid = reinterpret_cast<Uuid>(dlsym(nvml, "nvmlDeviceGetUUID"));
  auto capability = reinterpret_cast<Capability>(
      dlsym(nvml, "nvmlDeviceGetCudaComputeCapability"));
  if (!init || !shutdown || !count || !handle || !uuid || !capability ||
      init() != 0)
    return "";
  struct Device {
    std::string uuid, chip;
  };
  std::vector<Device> devices;
  unsigned n = 0;
  if (count(&n) == 0)
    for (unsigned i = 0; i < n; ++i) {
      void *h = nullptr;
      char name[96] = {};
      int major = 0, minor = 0;
      if (handle(i, &h) != 0 || uuid(h, name, sizeof(name)) != 0 ||
          capability(h, &major, &minor) != 0) {
        devices.clear();
        break;
      }
      devices.push_back(
          {name, "sm_" + std::to_string(major) + std::to_string(minor)});
    }
  shutdown();
  if (devices.empty() || device < 0)
    return "";

  // The entry of CUDA_VISIBLE_DEVICES that CUDA numbers `device`.
  std::string entry = std::to_string(device);
  if (const char *visible = std::getenv("CUDA_VISIBLE_DEVICES")) {
    SmallVector<StringRef> entries;
    StringRef(visible).split(entries, ',', -1, /*KeepEmpty=*/false);
    if (device >= static_cast<int64_t>(entries.size()))
      return "";
    entry = entries[device].trim().str();
  }
  if (StringRef(entry).starts_with("GPU-")) {
    for (const Device &d : devices)
      if (StringRef(d.uuid).starts_with(entry))
        return d.chip;
    return "";
  }
  unsigned index = 0;
  const char *order = std::getenv("CUDA_DEVICE_ORDER");
  if (!StringRef(entry).getAsInteger(10, index) && order &&
      StringRef(order) == "PCI_BUS_ID")
    return index < devices.size() ? devices[index].chip : "";
  // An index in the order of speed, or a MIG instance: known only when
  // every device has the same architecture.
  for (const Device &d : devices)
    if (d.chip != devices.front().chip)
      return "";
  return devices.front().chip;
}

/// The architecture for `device`: MDIR_GPU_ARCH, else NVML's (queryChip),
/// once per process and environment. Empty if neither is known.
static llvm::Expected<std::string> getChip(int64_t device) {
  if (const char *given = std::getenv("MDIR_GPU_ARCH"); given && *given) {
    if (!StringRef(given).starts_with("sm_") || !isKnownChip(given))
      return llvm::createStringError(
          "MDIR_GPU_ARCH='" + std::string(given) +
          "' is not an architecture that LLVM's NVPTX back end knows, such "
          "as sm_86");
    return std::string(given);
  }
  // The runtime takes MDRT_DEVICE over the device it is given.
  if (const char *chosen = std::getenv("MDRT_DEVICE"); chosen && *chosen)
    device = std::strtol(chosen, nullptr, 10);
  auto text = [](const char *name) {
    const char *value = std::getenv(name);
    return value ? std::string("=") + value : std::string();
  };
  std::string key = std::to_string(getpid()) + " " + std::to_string(device) +
                    " " + text("CUDA_VISIBLE_DEVICES") + " " +
                    text("CUDA_DEVICE_ORDER");
  static std::mutex mutex;
  static std::map<std::string, std::string> chips;
  std::lock_guard<std::mutex> lock(mutex);
  auto [entry, inserted] = chips.try_emplace(key);
  if (inserted) {
    std::string chip = queryChip(device);
    entry->second = !chip.empty() && isKnownChip(chip) ? chip : "";
  }
  return entry->second;
}

llvm::Expected<std::string> mdir::getGpuPipelineOptions(int64_t device) {
  std::string binary = "auto";
  if (const char *given = std::getenv("MDIR_GPU_BINARY"); given && *given)
    binary = StringRef(given).trim().lower();
  if (binary != "auto" && binary != "cubin" && binary != "ptx")
    return llvm::createStringError("MDIR_GPU_BINARY='" + binary +
                                   "' is not auto, cubin, or ptx");
  auto chip = getChip(device);
  if (!chip)
    return chip.takeError();
  // PTX for the architecture given, else for the default one, as before.
  if (binary == "ptx") {
    const char *given = std::getenv("MDIR_GPU_ARCH");
    return (given && *given ? "cubin-chip=" + *chip + " " : std::string()) +
           "cubin-format=isa binary=ptx";
  }
  if (chip->empty()) {
    if (binary == "cubin")
      return llvm::createStringError(
          "MDIR_GPU_BINARY=cubin, but the architecture of the device is not "
          "known: NVML cannot tell it for the device that "
          "CUDA_VISIBLE_DEVICES and MDRT_DEVICE select; set MDIR_GPU_ARCH, "
          "for example sm_86");
    return std::string("cubin-format=isa binary=auto");
  }
  return "cubin-chip=" + *chip + " cubin-format=bin binary=" + binary;
}

CompileStats mdir::takeGpuModuleStats(MLIRContext &context) {
  std::lock_guard<std::mutex> lock(getStatsMutex());
  CompileStats stats;
  auto &table = getStatsTable();
  if (auto found = table.find(&context); found != table.end()) {
    stats = found->second;
    table.erase(found);
  }
  return stats;
}

void mdir::registerGpuLowerToNVVMPipeline() {
  // A pipeline may be registered only once in a process.
  static std::once_flag once;
  std::call_once(once, [] {
    PassPipelineRegistration<GpuLowerToNVVMOptions>(
        "mdir-gpu-lower-to-nvvm",
        "gpu-lower-to-nvvm-pipeline, whose GPU modules are serialized in "
        "parallel by mdir-gpu-module-to-binary",
        buildGpuLowerToNVVM);
  });
}
