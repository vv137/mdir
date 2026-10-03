add_mlir_dialect_library(MDIRMDDist MDDistDialect.cpp
  DEPENDS MDIRMDDistIncGen
  LINK_LIBS PUBLIC MDIRMDRT MLIRIR MLIRFuncDialect)
