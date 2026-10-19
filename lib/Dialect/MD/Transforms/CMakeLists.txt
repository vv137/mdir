add_mlir_dialect_library(MDIRMDTransforms
  BypassUpdates.cpp
  CheckExchange.cpp
  CheckDerivatives.cpp
  CheckDerivativeCoverage.cpp
  CheckTransposes.cpp
  Activity.cpp
  Adjoints.cpp
  Differentiate.cpp
  Distribute.cpp
  ExpandTruncation.cpp
  Inline.cpp
  OwnerComputes.cpp
  ScalarDerivative.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Dialect/MD/Transforms

  DEPENDS
  MDIRMDPassIncGen
  MDIRDerivativeInterfaceIncGen
  MDIRAdjointInterfaceIncGen

  LINK_LIBS PUBLIC
  MDIRDyn
  MDIRMD
  MDIRMDDist
  MDIRMDRT
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRArithDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRVectorDialect
  )
