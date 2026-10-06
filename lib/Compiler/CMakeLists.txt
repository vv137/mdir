# The compile cache (D212): the host objects of the JIT engines and the
# binaries of the GPU modules (D[gpu-module-compile]).
add_mlir_library(MDIRCompileCache CompileCache.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_COMPONENTS BitWriter Core Object Support
)

add_mlir_library(MDIRCompiler Compile.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_COMPONENTS OrcJIT
  LINK_LIBS PUBLIC
  MDIRDriver MDIRDyn MDIRMD MDIRMDExec MDIRMDExecToGPU MDIRGPUToBinary MDIRMDExecToLoops
  MDIRMDExecTransforms MDIRMDRT MDIRMDToMDExec MDIRMDTransforms
  MLIRRegisterAllDialects MLIRRegisterAllExtensions MLIRRegisterAllPasses
  MLIRToLLVMIRTranslationRegistration
)
foreach(target MDIRCompiler obj.MDIRCompiler)
  if(TARGET ${target})
    target_compile_definitions(${target} PRIVATE MDIR_HAS_CUDA=$<BOOL:${MDIR_CUDA_FOUND}>)
  endif()
endforeach()

# A simulation that persists across runs (D196): it compiles
# and runs its programs in the process that embeds it.
add_mlir_library(MDIRSimulation Simulation.cpp JITEngine.cpp JITMemory.cpp
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
      MDIR_CUDA_ROOT="${MDIR_CUDA_ROOT}")
  endif()
endforeach()
