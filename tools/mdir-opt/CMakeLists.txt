set(LIBS
  MDIRDyn
  MDIRMD
  MDIRMDExec
  MDIRMDExecTransforms
  MDIRMDRT
  MDIRMDExecToGPU
  MDIRGPUToBinary
  MDIRMDExecToLoops
  MDIRMDToMDExec
  MLIRMemRefDialect
  MDIRMDTransforms
  MDIRTestPasses
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRMathDialect
  MLIROptLib
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  )

add_llvm_executable(mdir-opt mdir-opt.cpp)
llvm_update_compile_flags(mdir-opt)
target_link_libraries(mdir-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(mdir-opt)
