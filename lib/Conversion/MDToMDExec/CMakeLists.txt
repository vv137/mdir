add_mlir_conversion_library(MDIRMDToMDExec
  MDToMDExec.cpp

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
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRVectorDialect
  )
