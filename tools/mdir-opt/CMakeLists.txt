set(LIBS
  MDIRMD
  MDIRMDTransforms
  MDIRTestPasses
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIROptLib
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
  )

add_llvm_executable(mdir-opt mdir-opt.cpp)
llvm_update_compile_flags(mdir-opt)
target_link_libraries(mdir-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(mdir-opt)
