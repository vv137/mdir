add_mlir_dialect_library(MDIRMDExec
  MDExecDialect.cpp
  Independence.cpp
  MDExecOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MDExec

  DEPENDS
  MDIRMDExecIncGen
  MDIRMDExecEnumsIncGen

  LINK_LIBS PUBLIC
  MDIRMD
  MDIRMDRT
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  )
