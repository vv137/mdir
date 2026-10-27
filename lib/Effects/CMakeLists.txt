add_mlir_library(MDIREffects
  Effects.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  )
