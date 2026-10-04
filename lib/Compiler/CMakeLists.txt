add_mlir_library(MDIRCompiler Compile.cpp
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
