add_mlir_conversion_library(MDIRMDExecKernels
  CompactionEmitter.cpp
  ExpandPrimitives.cpp
  GatherEmitter.cpp
  MDExecKernels.cpp
  RaggedRows.cpp
  ReductionEmitter.cpp
  ScatterEmitter.cpp
  SinglePassScan.cpp
  TemplateWindowKernels.cpp
  TemplateWorkKernels.cpp
  TemplateBuildKernels.cpp
  TemplateSpatialKernels.cpp
  TemplateStageKernels.cpp
  TemplateSearchKernels.cpp
  TemplateParticleKernels.cpp
  TemplateRefreshKernels.cpp
  TemplateFrameKernels.cpp
  TemplateValidityKernels.cpp
  TemplateWriterKernels.cpp
  TemplatePruneKernels.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Conversion

  DEPENDS
  MDIRConversionPassIncGen
  MDIRMDExecIncGen

  LINK_LIBS PUBLIC
  MDIRMDExec
  MDIRMDRT
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRVectorDialect
  )
