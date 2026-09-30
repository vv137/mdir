add_mlir_conversion_library(MDIRMDExecKernels
  MDExecKernels.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Conversion

  DEPENDS
  MDIRMDExecIncGen

  LINK_LIBS PUBLIC
  MDIRMDExec
  MLIRArithDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  )
