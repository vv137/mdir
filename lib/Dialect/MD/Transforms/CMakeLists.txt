add_mlir_dialect_library(MDIRMDTransforms
  CheckExchange.cpp
  Differentiate.cpp
  ExpandTruncation.cpp
  ScalarDerivative.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MD/Transforms

  DEPENDS
  MDIRMDPassIncGen

  LINK_LIBS PUBLIC
  MDIRMD
  MLIRArithDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRVectorDialect
  )
