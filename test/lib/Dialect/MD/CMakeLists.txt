# Passes that exist only to test the md dialect.
add_mlir_library(MDIRTestPasses
  TestOutlineKernels.cpp
  TestTemplateWindows.cpp
  TestExecTemplateWindows.cpp
  TestLoopEffects.cpp
  TestStageAdjoint.cpp
  TestTranspose.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MDIRMD
  MDIRMDExec
  MDIRMDTransforms
  MDIRMDExecKernels
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  )
