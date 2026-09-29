add_mlir_dialect_library(MDIRMD
  MDDialect.cpp
  MDOps.cpp
  MDTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MD

  DEPENDS
  MDIRMDIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRSideEffectInterfaces
  )
