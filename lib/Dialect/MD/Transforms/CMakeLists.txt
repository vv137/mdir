add_mlir_dialect_library(MDIRMDTransforms
  BypassUpdates.cpp
  CheckExchange.cpp
  Activity.cpp
  Differentiate.cpp
  ExpandTruncation.cpp
  Inline.cpp
  ScalarDerivative.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MD/Transforms

  DEPENDS
  MDIRMDPassIncGen

  LINK_LIBS PUBLIC
  MDIRDyn
  MDIRMD
  MLIRArithDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRVectorDialect
  )
