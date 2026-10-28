# The compile cache (D212): the host objects of the JIT engines and the
# binaries of the GPU modules (D214).
add_mlir_library(MDIRCompileCache CompileCache.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_COMPONENTS BitWriter Core Object Support
)

add_mlir_library(MDIRCompiler Compile.cpp Registry.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_COMPONENTS OrcJIT Passes
  LINK_LIBS PUBLIC
  MDIRDriver MDIRDyn MDIRMD MDIRMDExec MDIRMDExecToGPU MDIRGPUToBinary MDIRMDExecToLoops MDIRMDDistToLoops
  MDIRMDExecTransforms MDIRMDRT MDIRMDToMDExec MDIRMDTransforms
  # The MLIR of the lowerings alone (Registry.cpp, #655): its dialects, the
  # interfaces attached to them, its conversions, and its translations.
  MDIRProgramDialects
  # Interfaces attached from outside the dialects
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRArithTransforms
  MLIRBufferizationTransforms
  MLIRCastInterfaces
  MLIRControlFlowTransforms
  MLIRGPUTransforms
  MLIRLLVMIRTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIRNVGPUTransforms
  MLIRSCFTransforms
  MLIRTensorTransforms
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRTensorTilingInterfaceImpl
  MLIRVectorTransforms
  MLIRNVVMTarget
  MLIRValueBoundsOpInterface
  # Conversions and the passes of the textual pipelines
  MLIRArithToLLVM
  MLIRComplexToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncInlinerExtension
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIRMemRefToLLVM
  MLIRNVVMToLLVM
  MLIROpenMPToLLVM
  MLIRUBToLLVM
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUToNVVMTransforms
  MLIRVectorToLLVMPass
  MLIRVectorToLLVM
  MLIRSCFToControlFlow
  MLIRSCFToOpenMP
  MLIRReconcileUnrealizedCasts
  MLIRTransforms
  # Translations to LLVM IR
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
)
foreach(target MDIRCompiler obj.MDIRCompiler)
  if(TARGET ${target})
    target_compile_definitions(${target} PRIVATE MDIR_HAS_CUDA=$<BOOL:${MDIR_CUDA_FOUND}>
      MDIR_CUDA_ROOT="${MDIR_CUDA_ROOT}")
  endif()
endforeach()

# A simulation that persists across runs (D196): it compiles
# and runs its programs in the process that embeds it.
add_mlir_library(MDIRSimulation Simulation.cpp JITEngine.cpp JITMemory.cpp Skins.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_COMPONENTS BitWriter Core Object Support nativecodegen native
    DebugInfoDWARF
  LINK_LIBS PUBLIC MDIRCompiler MDIRCompileCache MLIRExecutionEngine
)
foreach(target MDIRSimulation obj.MDIRSimulation)
  if(TARGET ${target})
    target_compile_definitions(${target} PRIVATE
      MDIR_BUILD_LIBRARY_DIR="${CMAKE_BINARY_DIR}/lib"
      MDIR_LLVM_LIBRARY_DIR="${LLVM_LIBRARY_DIR}"
      MDIR_CUDA_ROOT="${MDIR_CUDA_ROOT}"
      MDIR_VERSION="${PROJECT_VERSION}")
  endif()
endforeach()
