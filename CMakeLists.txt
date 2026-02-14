cmake_minimum_required(VERSION 3.22)
project(numsim_cas_python LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# numsim-cas: use local path for development, git for CI/packaging
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

set(NUMSIM_CAS_SOURCE_DIR "" CACHE PATH
    "Local path to numsim-cas source tree (leave empty to fetch from git)")
set(NUMSIM_CAS_GIT_REPO "https://github.com/OWNER/numsim-cas.git" CACHE STRING
    "Git repository URL for numsim-cas")
set(NUMSIM_CAS_GIT_TAG "move_to_virtual" CACHE STRING
    "Git tag/branch for numsim-cas")

if(NUMSIM_CAS_SOURCE_DIR)
  FetchContent_Declare(numsim_cas SOURCE_DIR "${NUMSIM_CAS_SOURCE_DIR}")
else()
  FetchContent_Declare(numsim_cas
    GIT_REPOSITORY "${NUMSIM_CAS_GIT_REPO}"
    GIT_TAG "${NUMSIM_CAS_GIT_TAG}"
  )
endif()
FetchContent_MakeAvailable(numsim_cas)

# pybind11
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_core NO_EXTRAS
  src/module.cpp
  src/bind_scalar.cpp
  src/bind_tensor.cpp
  src/bind_tensor_to_scalar.cpp
  src/bind_cross_domain.cpp
)
target_link_libraries(_core PRIVATE NumSim_CAS)
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# MSVC-specific: silence C4251 warnings from pybind11/STL
if(MSVC)
  target_compile_options(_core PRIVATE /W3 /wd4251)
endif()

install(TARGETS _core DESTINATION numsim_cas)
