add_mlir_dialect_library(MDIRMDExecTransforms
  FuseLoops.cpp
  ReuseNeighbors.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MDExec/Transforms

  DEPENDS
  MDIRMDExecPassIncGen

  LINK_LIBS PUBLIC
  MDIRMDExec
  MDIRMDRT
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  )
