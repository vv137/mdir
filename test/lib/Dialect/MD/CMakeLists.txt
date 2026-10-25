# Passes that exist only to test the md dialect.
add_mlir_library(MDIRTestPasses
  TestOutlineKernels.cpp
  TestLoopEffects.cpp
  TestStageAdjoint.cpp
  TestTranspose.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MDIRMD
  MDIRMDExec
  MDIRMDTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  )
