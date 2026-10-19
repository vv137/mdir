add_mlir_conversion_library(MDIRMDExecKernels
  CompactionEmitter.cpp
  ExpandPrimitives.cpp
  MDExecKernels.cpp
  ScatterEmitter.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Conversion

  DEPENDS
  MDIRConversionPassIncGen
  MDIRMDExecIncGen

  LINK_LIBS PUBLIC
  MDIRMDExec
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRVectorDialect
  )
