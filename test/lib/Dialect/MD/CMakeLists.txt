# Passes that exist only to test the md dialect.
add_mlir_library(MDIRTestPasses
  TestOutlineKernels.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MDIRMD
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  )
