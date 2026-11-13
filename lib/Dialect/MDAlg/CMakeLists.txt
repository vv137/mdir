add_mlir_dialect_library(MDIRMDAlg
  MDAlgDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MDAlg

  DEPENDS
  MDIRMDAlgIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  )
