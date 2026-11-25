add_mlir_conversion_library(MDIRMDExecKernels
  AxisFFTEmitter.cpp
  BondedUnits.cpp
  CompactionEmitter.cpp
  GatherEmitter.cpp
  GroupsBuild.cpp
  TowersBuild.cpp
  TowersForce.cpp
  MatrixBuild.cpp
  HostPairLanes.cpp
  MDAlgLowering.cpp
  MDExecKernels.cpp
  TraversePairs.cpp
  TraverseSegments.cpp
  TraverseSegmentsSIMT.cpp
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
