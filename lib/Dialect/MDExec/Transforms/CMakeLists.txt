add_mlir_dialect_library(MDIRMDExecTransforms
  AssignPrecision.cpp
  FuseLoops.cpp
  ReuseNeighbors.cpp
  SimplifyDistance.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MDExec/Transforms

  DEPENDS
  MDIRMDExecPassIncGen

  LINK_LIBS PUBLIC
  MDIRMDExec
  MDIRMDRT
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRSCFDialect
  )
