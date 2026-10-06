//===- GPUToBinary.cpp - Serialize GPU modules in parallel ----------------===//
//
// `mdir-gpu-module-to-binary` does what upstream `gpu-module-to-binary`
// (mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp) does, one
// `gpu.binary` for each `gpu.module`, but serializes the modules on the
// threads of the context, takes their PTX and cubins from the compile
// cache, and runs ptxas itself (D[gpu-module-compile]).
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
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
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
/// else the toolkit of the build), else the one on PATH; and the text of
/// its `--version`, which is part of the key of a cubin. Empty if there is
/// none that runs.
struct Ptxas {
  std::string path;
  std::string version;
};
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
  if (!runTool(ptxas.path, argv, log))
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
    if (writable && !mdir::compiler::writeCacheEntry(
                         getPath(key, extension), key, data, seconds)) {
      ++counters.stored;
      return;
    }
    // writeCacheEntry's error, if any, is dropped with the entry.
    ++counters.unstored;
  }

  void evict() {
    if (config)
      mdir::compiler::evictCache(config->directory, config->maxBytes);
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
                                     NVVM::NVVMTargetAttr target) {
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
    << "target " << target << '\n';
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
  GpuCache cache(CompileCacheConfig::fromEnvironment());
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
            key = getPtxKey(module, nvvm);
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
          if (ptxas && nvvm && !getUnsupported().contains(nvvm.getChip())) {
            SmallVector<std::string> arguments = {
                "-arch", nvvm.getChip().str(), "--opt-level",
                std::to_string(nvvm.getO())};
            std::string cubinKey;
            llvm::raw_string_ostream k(cubinKey);
            k << kCubinFormat << '\n'
              << "ptxas " << ptxas->version << '\n'
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
    cache.evict();

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
void buildGpuLowerToNVVM(OpPassManager &pm,
                         const gpu::GPUToNVVMPipelineOptions &options) {
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
  pm.addPass(mdir::createGpuModuleToBinary(binary));
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

} // namespace

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
    PassPipelineRegistration<gpu::GPUToNVVMPipelineOptions>(
        "mdir-gpu-lower-to-nvvm",
        "gpu-lower-to-nvvm-pipeline, whose GPU modules are serialized in "
        "parallel by mdir-gpu-module-to-binary",
        buildGpuLowerToNVVM);
  });
}
