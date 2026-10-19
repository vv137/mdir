add_mlir_dialect_library(MDIRMDExec
  Algebra.cpp
  MDExecDialect.cpp
  Independence.cpp
  MDExecOps.cpp
  Scatter.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MDExec

  DEPENDS
  MDIRMDExecIncGen
  MDIRMDExecEnumsIncGen
  MDIRMDExecAttrsIncGen

  LINK_LIBS PUBLIC
  MDIRMD
  MDIRMDDist
  MDIRMDRT
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  )
