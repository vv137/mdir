add_mlir_dialect_library(MDIRMDExec
  Algebra.cpp
  HostPairs.cpp
  MDExecDialect.cpp
  Independence.cpp
  LoopEffects.cpp
  MDExecOps.cpp
  NeighborContract.cpp
  NeighborCoverage.cpp
  ProfileMap.cpp
  Scatter.cpp
  StorageSet.cpp
  Transfers.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MDExec

  DEPENDS
  MDIRMDExecIncGen
  MDIRMDExecEnumsIncGen
  MDIRMDExecAttrsIncGen
  MDIRMDExecLoopEffectsIncGen

  LINK_LIBS PUBLIC
  MDIRMD
  MDIREffects
  MDIRMDDist
  MDIRMDRT
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRIR
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRLoopLikeInterface
  MLIRMathDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRVectorDialect
  MLIRViewLikeInterface
  )
