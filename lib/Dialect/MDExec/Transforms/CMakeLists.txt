add_mlir_dialect_library(MDIRMDExecTransforms
  AccumulateDestinations.cpp
  Approximate.cpp
  AssignPrecision.cpp
  AssignStorage.cpp
  AssignStreams.cpp
  ChooseNeighbors.cpp
  ExposeValidity.cpp
  FoldTables.cpp
  FuseLoops.cpp
  NarrowSums.cpp
  RebuildAtInterval.cpp
  ReuseNeighbors.cpp
  SimplifyDistance.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MDExec/Transforms

  DEPENDS
  MDIRMDExecPassIncGen

  LINK_LIBS PUBLIC
  MDIRMD
  MDIRMDExec
  MDIRMDRT
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRVectorDialect
  )
