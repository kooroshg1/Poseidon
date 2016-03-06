cmake_minimum_required(VERSION 2.8)

project(NSsolver)

set(CMAKE_MODULE_PATH "/home/koorosh/Tools/cmake_modules_for_petsc")
set(PETSC_DIR "/home/koorosh/Tools/petsc-3.6.3")
set(PETSC_ARCH "arch-linux2-c-debug")

set(CMAKE_CXX_COMPILER "/home/koorosh/Tools/bin/mpicxx") # Set up MPI compilers C++
set(CMAKE_C_COMPILER "/home/koorosh/Tools/bin/mpicc") # Set up MPI compilers C

FIND_PACKAGE(PETSc REQUIRED)
INCLUDE_DIRECTORIES(${PETSC_INCLUDES})

ADD_EXECUTABLE(NSsolver solver.cpp)
TARGET_LINK_LIBRARIES(NSsolver ${PETSC_LIBRARIES})
