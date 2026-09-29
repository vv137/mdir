set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_executable(mdir-checkpoint mdir-checkpoint.cpp)
llvm_update_compile_flags(mdir-checkpoint)
target_link_libraries(mdir-checkpoint PRIVATE MDIRDriver)
