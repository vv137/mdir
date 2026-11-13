# Passes that exist only to test the md dialect and the emitters of the
# device lowering.
add_mlir_library(MDIRTestPasses
  TestOutlineKernels.cpp
  TestLoopEffects.cpp
  TestStageAdjoint.cpp
  TestTranspose.cpp
  TestSegmentWindows.cpp
  TestWarpHelpers.cpp
  TestCollectionAnalysis.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MDIRMD
  MDIRMDAlg
  MDIRMDExecKernels
  MDIRMDExec
  MDIRMDTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRMathDialect
  MLIRIR
  MLIRPass
  )
