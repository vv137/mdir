// The dialects, interfaces, translations, and passes that the lowerings of
// `mdir` and of the Python package use, and no others (#655). The
// `registerAll*` functions of MLIR referenced every upstream dialect and
// pass, so that the linker kept all of MLIR in each binary. A program text
// that names a dialect outside this list fails to parse ("unregistered
// dialect"), and a pipeline that names a pass outside it fails to parse
// ("does not refer to a registered pass"): extend the shared dialect set in
// lib/Dialect/ProgramDialects.cpp and its libraries in lib/CMakeLists.txt, or
// the passes/interfaces below and lib/Compiler/CMakeLists.txt, together.
// `mdir-opt` keeps all of MLIR, for replaying and developing pipelines
// (docs/debugging.md).
//
// What is listed, and why:
// - Dialects: the shared set and its rationale are in
//   lib/Dialect/ProgramDialects.cpp; construction and lowering use it.
// - Interfaces of those dialects that MLIR attaches from outside the
//   dialect (bufferization, value bounds, inlining, the conversion of
//   `gpu-to-llvm` to the LLVM dialect, the NVVM target), as
//   `registerAllDialects` and `registerAllExtensions` attach them; the
//   sharding and transform-dialect extensions are left out, since no
//   lowering uses them.
// - Translations to LLVM IR of the builtin, LLVM, NVVM, OpenMP, and GPU
//   dialects, with the offloading of GPU binaries.
// - Passes: those that the textual pipelines of Compile.cpp and Run.cpp
//   name. The passes that the lowerings add in C++ (GPUToBinary.cpp,
//   StageBodies.cpp) need no registration.
#include "mdir/Compiler/Compile.h"
#include "mdir/Conversion/Passes.h"
#include "mdir/Dialect/Dyn/DynDialect.h"
#include "mdir/Dialect/MD/MDDialect.h"
#include "mdir/Dialect/MD/Transforms/DerivativeInterface.h"
#include "mdir/Dialect/MD/Transforms/Passes.h"
#include "mdir/Dialect/MDExec/MDExecDialect.h"
#include "mdir/Dialect/MDExec/Transforms/Passes.h"
#include "mdir/Dialect/MDRT/MDRTDialect.h"
#include "mdir/Dialect/ProgramDialects.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/GPU/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/GPU/Transforms/IndexedAccessOpInterfaceImpl.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRefMemorySlot.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/MemoryAccessOpInterfacesImpl.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/RuntimeOpVerification.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/IndexedAccessOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"

#include <mutex>

using namespace mlir;
using namespace mdir;

namespace {
/// The upstream passes that the textual pipelines of the lowerings name.
void registerPipelinePasses() {
  registerCanonicalizerPass();
  registerCSEPass();
  registerSCFToControlFlowPass();
  registerConvertSCFToOpenMPPass();
  registerConvertMathToLLVMPass();
  registerConvertMathToLibmPass();
  registerConvertVectorToLLVMPass();
  memref::registerExpandStridedMetadataPass();
  registerFinalizeMemRefToLLVMConversionPass();
  registerArithToLLVMConversionPass();
  registerConvertFuncToLLVMPass();
  registerConvertControlFlowToLLVMPass();
  registerUBToLLVMConversionPass();
  registerConvertOpenMPToLLVMPass();
  registerReconcileUnrealizedCastsPass();
}

/// The interfaces of those dialects that MLIR attaches from outside them.
void registerPipelineInterfaces(DialectRegistry &registry) {
  affine::registerValueBoundsOpInterfaceExternalModels(registry);
  arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  arith::registerBufferViewFlowOpInterfaceExternalModels(registry);
  arith::registerValueBoundsOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  builtin::registerCastOpInterfaceExternalModels(registry);
  cf::registerBufferizableOpInterfaceExternalModels(registry);
  cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  gpu::registerBufferDeallocationOpInterfaceExternalModels(registry);
  gpu::registerIndexedAccessOpInterfaceExternalModels(registry);
  gpu::registerValueBoundsOpInterfaceExternalModels(registry);
  LLVM::registerInlinerInterface(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerSubsetOpInterfaceExternalModels(registry);
  linalg::registerTilingInterfaceExternalModels(registry);
  linalg::registerValueBoundsOpInterfaceExternalModels(registry);
  linalg::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  memref::registerAllocationOpInterfaceExternalModels(registry);
  memref::registerBufferViewFlowOpInterfaceExternalModels(registry);
  memref::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  memref::registerValueBoundsOpInterfaceExternalModels(registry);
  memref::registerMemorySlotExternalModels(registry);
  nvgpu::registerMemoryAccessOpInterfacesExternalModels(registry);
  scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerValueBoundsOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
  tensor::registerRuntimeVerifiableOpInterfaceExternalModels(registry);
  tensor::registerSubsetOpInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);
  tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerIndexedAccessOpInterfaceExternalModels(registry);
  vector::registerSubsetOpInterfaceExternalModels(registry);
  vector::registerValueBoundsOpInterfaceExternalModels(registry);
  NVVM::registerNVVMTargetInterfaceExternalModels(registry);

  // The conversions to the LLVM dialect that a pass such as `gpu-to-llvm`
  // takes from the dialects it meets, and the inliner of func.
  arith::registerConvertArithToLLVMInterface(registry);
  registerConvertComplexToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);
  func::registerInlinerExtension(registry);
  registerConvertFuncToLLVMInterface(registry);
  index::registerConvertIndexToLLVMInterface(registry);
  registerConvertMathToLLVMInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
  registerConvertNVVMToLLVMInterface(registry);
  registerConvertOpenMPToLLVMInterface(registry);
  ub::registerConvertUBToLLVMInterface(registry);
  gpu::registerConvertGpuToLLVMInterface(registry);
  NVVM::registerConvertGpuToNVVMInterface(registry);
  vector::registerConvertVectorToLLVMInterface(registry);
}

/// The translations to LLVM IR.
void registerPipelineTranslations(DialectRegistry &registry) {
  registerBuiltinDialectTranslation(registry);
  registerGPUDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerNVVMDialectTranslation(registry);
  registerOpenMPDialectTranslation(registry);
  gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
}
} // namespace

DialectRegistry mdir::compiler::getRegistry() {
  // Process-wide registries are initialized once; each lowering owns a context.
  static std::once_flag registration;
  std::call_once(registration, [] {
    registerPipelinePasses();
    md::registerMDPasses();
    md_exec::registerMDExecPasses();
    registerMDIRConversionPasses();
    registerGpuLowerToNVVMPipeline();
  });
  DialectRegistry registry;
  insertProgramDialects(registry);
  registerPipelineInterfaces(registry);
  md::registerDerivativeInterfaces(registry);
  registerPipelineTranslations(registry);
  return registry;
}
