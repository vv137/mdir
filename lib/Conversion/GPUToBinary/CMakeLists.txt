add_mlir_conversion_library(MDIRGPUToBinary
  GPUToBinary.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Conversion

  DEPENDS
  MDIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MDIRCompileCache
  MLIRAffineToStandard
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRGPUDialect
  MLIRGPUPipelines
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUToNVVMTransforms
  MLIRGPUTransforms
  MLIRIndexToLLVM
  MLIRIR
  MLIRMathToLLVM
  MLIRMemRefTransforms
  MLIRNVGPUToNVVM
  MLIRNVVMDialect
  MLIRNVVMTarget
  MLIRNVVMToLLVM
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRTransforms
  MLIRVectorToSCF
  )
