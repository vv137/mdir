//===- GPUToBinary.cpp - Serialize GPU modules in parallel ----------------===//
//
// `mdir-gpu-module-to-binary` does what upstream `gpu-module-to-binary`
// (mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp) does, one
// `gpu.binary` for each `gpu.module`, but serializes the modules on the
// threads of the context (D[gpu-module-compile]). `mdir-gpu-lower-to-nvvm`
// is upstream `gpu-lower-to-nvvm-pipeline`
// (mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp), pass for pass,
// with this pass in place of upstream's serialization.
//
//===----------------------------------------------------------------------===//

#include "mdir/Conversion/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include <chrono>
#include <mutex>

namespace mdir {
#define GEN_PASS_DEF_GPUMODULETOBINARY
#include "mdir/Conversion/Passes.h.inc"
} // namespace mdir

using namespace mlir;

namespace {

/// What one module serializes to: the object of each of its targets.
struct Serialized {
  SmallVector<Attribute> objects;
};

class GpuModuleToBinary
    : public mdir::impl::GpuModuleToBinaryBase<GpuModuleToBinary> {
public:
  using Base::Base;
  void runOnOperation() final;
};

void GpuModuleToBinary::runOnOperation() {
  auto started = std::chrono::steady_clock::now();
  auto target = llvm::StringSwitch<std::optional<gpu::CompilationTarget>>(
                    format.getValue())
                    .Cases({"assembly", "isa"}, gpu::CompilationTarget::Assembly)
                    .Cases({"binary", "bin"}, gpu::CompilationTarget::Binary)
                    .Default(std::nullopt);
  if (!target) {
    getOperation()->emitError() << "unknown format '" << format
                                << "' (expected isa or bin)";
    return signalPassFailure();
  }
  // The modules in the order of the blocks that hold them, as upstream.
  SmallVector<gpu::GPUModuleOp> modules;
  for (Region &region : getOperation()->getRegions())
    for (Block &block : region.getBlocks())
      for (auto module : block.getOps<gpu::GPUModuleOp>())
        modules.push_back(module);
  numModules += modules.size();

  // The symbol table is built before the threads start; upstream builds it
  // lazily, which is not safe on several threads.
  SymbolTable table(getOperation());
  auto getTable = [&]() -> SymbolTable * { return &table; };
  gpu::TargetOptions options({}, {}, {}, {}, *target, getTable);

  // Each module is translated to LLVM IR in an LLVM context of its own and
  // read, not changed, so modules are serialized side by side. The
  // diagnostics are ordered by module.
  std::vector<Serialized> results(modules.size());
  LogicalResult serialized = failableParallelForEachN(
      &getContext(), 0, modules.size(), [&](size_t i) -> LogicalResult {
        gpu::GPUModuleOp module = modules[i];
        if (!module.getTargetsAttr())
          return module.emitError("the module has no target attributes");
        for (Attribute attribute : module.getTargetsAttr()) {
          auto target = cast<gpu::TargetAttrInterface>(attribute);
          std::optional<gpu::SerializedObject> object =
              target.serializeToObject(module, options);
          if (!object)
            return module.emitError(
                "An error happened while serializing the module.");
          // Without the times of the serialization, which upstream records
          // as properties: they would differ from run to run.
          gpu::SerializedObject timeless(std::move(object->getObject()),
                                         DictionaryAttr::get(&getContext()));
          Attribute created = target.createObject(module, timeless, options);
          if (!created)
            return module.emitError(
                "An error happened while creating the object.");
          results[i].objects.push_back(created);
        }
        return success();
      });
  if (failed(serialized))
    return signalPassFailure();

  // The binaries replace the modules on this thread, in their order.
  OpBuilder builder(&getContext());
  for (auto [module, result] : llvm::zip_equal(modules, results)) {
    builder.setInsertionPointAfter(module);
    auto handler = dyn_cast_or_null<gpu::OffloadingLLVMTranslationAttrInterface>(
        module.getOffloadingHandlerAttr());
    gpu::BinaryOp::create(builder, module.getLoc(), module.getName(), handler,
                          builder.getArrayAttr(result.objects));
    module->erase();
  }
  serializeMicroseconds +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - started)
          .count();
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
