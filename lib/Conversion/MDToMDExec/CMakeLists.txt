add_mlir_conversion_library(MDIRMDToMDExec
  MDToMDExec.cpp
  LinearTransformValues.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Conversion

  DEPENDS
  MDIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MDIRDyn
  MDIRMD
  MDIRMDExec
  MDIRMDRT
  MLIRArithDialect
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRVectorDialect
  )
