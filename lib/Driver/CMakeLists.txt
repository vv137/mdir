# Checkpoints need HDF5. Without it the driver is built all the same, and
# a control file that asks for a checkpoint is an error.
set(MDIR_ENABLE_HDF5 "AUTO" CACHE STRING
  "Build the driver with checkpoints, which need HDF5: ON, OFF, or AUTO")
set(MDIR_HDF5_FOUND OFF)
if(MDIR_ENABLE_HDF5)
  find_path(MDIR_HDF5_INCLUDE_DIR hdf5.h
    HINTS ${HDF5_ROOT} ENV HDF5_ROOT
    PATH_SUFFIXES include include/hdf5/serial)
  find_library(MDIR_HDF5_LIBRARY NAMES hdf5 hdf5_serial
    HINTS ${HDF5_ROOT} ENV HDF5_ROOT
    PATH_SUFFIXES lib lib64)
  if(MDIR_HDF5_INCLUDE_DIR AND MDIR_HDF5_LIBRARY)
    set(MDIR_HDF5_FOUND ON)
  elseif(NOT MDIR_ENABLE_HDF5 STREQUAL "AUTO")
    message(FATAL_ERROR
      "MDIR_ENABLE_HDF5 is ON, but HDF5 was not found. Set HDF5_ROOT.")
  endif()
endif()

# Embed the standard pipeline so templates work without the source tree.
# The examples remain the source of the stage settings (D147).
function(read_pipeline_template variable file)
  set(path "${PROJECT_SOURCE_DIR}/examples/ala3/${file}.toml")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
  file(READ "${path}" contents)
  string(REPLACE "ala3.prmtop" "system.prmtop" contents "${contents}")
  string(REPLACE "ala3.inpcrd" "system.inpcrd" contents "${contents}")
  set(${variable} "${contents}" PARENT_SCOPE)
endfunction()
read_pipeline_template(MDIR_TEMPLATE_MINIMIZE 1-min)
read_pipeline_template(MDIR_TEMPLATE_NVT 2-nvt)
read_pipeline_template(MDIR_TEMPLATE_NPT 3-npt)
read_pipeline_template(MDIR_TEMPLATE_PRODUCTION 4-md)
configure_file(Templates.cpp.in Templates.cpp @ONLY)

add_mlir_library(MDIRDriver
  Amber.cpp
  Builder.cpp
  Cell.cpp
  Charmm.cpp
  CMap.cpp
  Checkpoint.cpp
  Control.cpp
  Expression.cpp
  Fingerprint.cpp
  Gromacs.cpp
  H5MD.cpp
  Model.cpp
  ModelFingerprint.cpp
  Tunables.cpp
  Output.cpp
  Selection.cpp
  System.cpp
  Trajectory.cpp
  Validation.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/Templates.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mdir/Driver

  LINK_COMPONENTS
  Support
  )
# The sources are compiled in an object library of their own.
set(MDIR_DRIVER_TARGETS MDIRDriver)
if(TARGET obj.MDIRDriver)
  list(APPEND MDIR_DRIVER_TARGETS obj.MDIRDriver)
endif()

foreach(target ${MDIR_DRIVER_TARGETS})
  target_include_directories(${target} SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/tomlplusplus)
  if(MDIR_HDF5_FOUND)
    target_include_directories(${target} SYSTEM PRIVATE
      ${MDIR_HDF5_INCLUDE_DIR})
    target_compile_definitions(${target} PRIVATE MDIR_HAS_HDF5=1)
  else()
    target_compile_definitions(${target} PRIVATE MDIR_HAS_HDF5=0)
  endif()
endforeach()

if(MDIR_HDF5_FOUND)
  message(STATUS "HDF5: ${MDIR_HDF5_LIBRARY}")
  target_link_libraries(MDIRDriver PUBLIC ${MDIR_HDF5_LIBRARY})
else()
  message(STATUS "HDF5: not found; the driver writes no checkpoints")
endif()
set_property(GLOBAL PROPERTY MDIR_HDF5_FOUND ${MDIR_HDF5_FOUND})
