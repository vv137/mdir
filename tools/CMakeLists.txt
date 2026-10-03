add_subdirectory(mdir)
add_subdirectory(mdir-opt)

option(MDIR_ENABLE_MPI "Build the experimental fixed-layout CPU LJ tool" OFF)
if(MDIR_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  add_subdirectory(mdir-cpu-lj)
endif()
