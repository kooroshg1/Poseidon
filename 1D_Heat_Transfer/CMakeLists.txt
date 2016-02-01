cmake_minimum_required(VERSION 2.8)

project(oneD_heattransfer)

set(CMAKE_MODULE_PATH "/home/koorosh/Tools/petsc-3.6.3/cmake_files_for_PETSC")
set(PETSC_DIR "/home/koorosh/Tools/petsc-3.6.3")
set(PETSC_ARCH "arch-linux2-c-debug")
FIND_PACKAGE(PETSc REQUIRED)
INCLUDE_DIRECTORIES(${PETSC_INCLUDES})

ADD_EXECUTABLE(oneD_heattransfer oneD_heattransfer.cpp)
TARGET_LINK_LIBRARIES(oneD_heattransfer ${PETSC_LIBRARIES})
