add_mlir_dialect_library(MDIRMDRT
  MDRTDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MDRT

  DEPENDS
  MDIRMDRTIncGen

  LINK_LIBS PUBLIC
  MDIRMD
  MLIRIR
  MLIRSideEffectInterfaces
  )
