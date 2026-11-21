# Compile the PME templates into the library as strings.
set(PME_GPU_TEMPLATE_FILE
  ${PROJECT_SOURCE_DIR}/lib/Runtime/Templates/PMEGPU.mlir)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${PME_GPU_TEMPLATE_FILE})
file(READ ${PME_GPU_TEMPLATE_FILE} PME_GPU_TEMPLATE)
set(PME_GPU_TRICLINIC_TEMPLATE_FILE
  ${PROJECT_SOURCE_DIR}/lib/Runtime/Templates/PMEGPUTriclinic.mlir)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${PME_GPU_TRICLINIC_TEMPLATE_FILE})
file(READ ${PME_GPU_TRICLINIC_TEMPLATE_FILE} PME_GPU_TRICLINIC_TEMPLATE)
set(PME_GPU_PROLATE_TEMPLATE_FILE
  ${PROJECT_SOURCE_DIR}/lib/Runtime/Templates/PMEGPUProlate.mlir)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${PME_GPU_PROLATE_TEMPLATE_FILE})
file(READ ${PME_GPU_PROLATE_TEMPLATE_FILE} PME_GPU_PROLATE_TEMPLATE)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Templates.cpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/Templates.cpp
  @ONLY)

add_mlir_conversion_library(MDIRMDExecToGPU
  MDExecToGPU.cpp
  MDExecToMDAlg.cpp
  StageBodies.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/Templates.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Conversion

  DEPENDS
  MDIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MDIRMD
  MDIRMDExec
  MDIRMDExecKernels
  MDIRMDRT
  MLIRArithDialect
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncTransforms
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTransforms
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRNVVMDialect
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRTransformUtils
  MLIRVectorDialect
  )
