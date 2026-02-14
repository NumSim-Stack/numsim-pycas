cmake_minimum_required(VERSION 3.22)
project(numsim_cas_python LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# numsim-cas (suppress its tests, build as STATIC to bundle into _core.so)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(numsim_cas
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../numsim-cas"
)
FetchContent_MakeAvailable(numsim_cas)

# pybind11
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# Disable LTO: numsim-cas relies on implicit template instantiation in .cpp
# files via inline virtual functions; LTO defers this and breaks linkage.
pybind11_add_module(_core NO_EXTRAS
  src/module.cpp
  src/bind_scalar.cpp
  src/bind_tensor.cpp
  src/bind_tensor_to_scalar.cpp
  src/bind_cross_domain.cpp
)
target_link_libraries(_core PRIVATE NumSim_CAS)
target_link_options(_core PRIVATE
  "LINKER:--whole-archive"
  $<TARGET_FILE:NumSim_CAS>
  "LINKER:--no-whole-archive"
)
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

install(TARGETS _core DESTINATION numsim_cas)
