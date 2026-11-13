add_mlir_conversion_library(MDIRMDExecKernels
  CompactionEmitter.cpp
  GatherEmitter.cpp
  GroupsBuild.cpp
  HostPairLanes.cpp
  MDAlgLowering.cpp
  MDExecKernels.cpp
  RaggedRows.cpp
  ReductionEmitter.cpp
  ScatterEmitter.cpp
  SinglePassScan.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Conversion

  DEPENDS
  MDIRConversionPassIncGen
  MDIRMDAlgIncGen
  MDIRMDExecIncGen

  LINK_LIBS PUBLIC
  MDIRMDAlg
  MDIRMDExec
  MDIRMDRT
  MLIRArithAttrToLLVMConversion
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRVectorDialect
  )
