add_mlir_dialect_library(MDIRDyn
  DynDialect.cpp
  DynOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/Dyn

  DEPENDS
  MDIRDynIncGen

  LINK_LIBS PUBLIC
  MDIRMD
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  )
