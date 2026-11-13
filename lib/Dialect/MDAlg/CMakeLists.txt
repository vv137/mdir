add_mlir_dialect_library(MDIRMDAlg
  CollectionAnalysis.cpp
  MDAlgDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MDAlg

  DEPENDS
  MDIRMDAlgIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRCallInterfaces
  MLIRFunctionInterfaces
  MLIRLoopLikeInterface
  MLIRIR
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRInferTypeOpInterface
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  )
