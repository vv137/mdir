add_mlir_library(MDIRCompiler Compile.cpp
  PARTIAL_SOURCES_INTENDED
  LINK_COMPONENTS OrcJIT
  LINK_LIBS PUBLIC
  MDIRDriver MDIRDyn MDIRMD MDIRMDExec MDIRMDExecToGPU MDIRMDExecToLoops
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
  LINK_COMPONENTS Core Support nativecodegen native DebugInfoDWARF
  LINK_LIBS PUBLIC MDIRCompiler MLIRExecutionEngine
)
foreach(target MDIRSimulation obj.MDIRSimulation)
  if(TARGET ${target})
    target_compile_definitions(${target} PRIVATE
      MDIR_BUILD_LIBRARY_DIR="${CMAKE_BINARY_DIR}/lib"
      MDIR_LLVM_LIBRARY_DIR="${LLVM_LIBRARY_DIR}"
      MDIR_CUDA_ROOT="${MDIR_CUDA_ROOT}")
  endif()
endforeach()
