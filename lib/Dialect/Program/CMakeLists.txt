# Shared accepted dialects for construction, canonical printing, and lowering.
add_mlir_library(MDIRProgramDialects ProgramDialects.cpp
  LINK_LIBS PUBLIC MDIRDyn MDIRMD MDIRMDExec MDIRMDRT
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRComplexDialect
  MLIRControlFlowDialect
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIndexDialect
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRNVVMDialect
  MLIRNVGPUDialect
  MLIROpenMPDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRUBDialect
  MLIRVectorDialect
)
