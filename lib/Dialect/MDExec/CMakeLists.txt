add_mlir_dialect_library(MDIRMDExec
  MDExecDialect.cpp
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
  MLIRIR
  MLIRSideEffectInterfaces
  )
