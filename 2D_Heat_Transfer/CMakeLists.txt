cmake_minimum_required(VERSION 2.8)

project(twoD_heattransfer)

set(CMAKE_MODULE_PATH "/home/koorosh/Tools/petsc-3.6.3/cmake_files_for_PETSC")
set(PETSC_DIR "/home/koorosh/Tools/petsc-3.6.3")
set(PETSC_ARCH "arch-linux2-c-debug")

set(CMAKE_CXX_COMPILER "/usr/bin/mpicxx") # Set up MPI compilers C++
set(CMAKE_C_COMPILER "/usr/bin/mpicc") # Set up MPI compilers C

FIND_PACKAGE(PETSc REQUIRED)
INCLUDE_DIRECTORIES(${PETSC_INCLUDES})

ADD_EXECUTABLE(twoD_heattransfer main.cpp)
TARGET_LINK_LIBRARIES(twoD_heattransfer ${PETSC_LIBRARIES})