add_mlir_dialect_library(MDIRMDDist
  MDDistDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MDDist

  DEPENDS
  MDIRMDDistIncGen

  LINK_LIBS PUBLIC
  MDIRMD
  MLIRIR
  MLIRSideEffectInterfaces
  )
