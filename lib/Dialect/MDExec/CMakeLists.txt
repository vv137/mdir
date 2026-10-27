add_mlir_dialect_library(MDIRMDExec
  Algebra.cpp
  MDExecDialect.cpp
  Independence.cpp
  LoopEffects.cpp
  MDExecOps.cpp
  NeighborCoverage.cpp
  Scatter.cpp
  StorageSet.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MDExec

  DEPENDS
  MDIRMDExecIncGen
  MDIRMDExecEnumsIncGen
  MDIRMDExecAttrsIncGen
  MDIRMDExecLoopEffectsIncGen

  LINK_LIBS PUBLIC
  MDIRMD
  MDIRMDDist
  MDIRMDRT
  MLIRControlFlowInterfaces
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  )
