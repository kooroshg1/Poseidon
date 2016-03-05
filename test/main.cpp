#include <iostream>
#include <petscksp.h>
using namespace std;

static char help[] = "just a test!";

void vecExtract(Vec &V, PetscInt *nodeIndex, PetscInt n);

int main(int argc, char **args) {
    PetscMPIInt size;
    PetscInt number = 10;
    PetscBool nonzeroguess = PETSC_FALSE;
    PetscInitialize(&argc, &args, (char*)0, help);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    PetscOptionsGetInt(NULL, "-n", &number, NULL);
    PetscOptionsGetBool(NULL, "-nonzero_guess", &nonzeroguess, NULL);
    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
    Vec V;
    VecCreate(PETSC_COMM_WORLD, &V); VecSetSizes(V, PETSC_DECIDE, 7); VecSetFromOptions(V); VecSet(V, 0);
    PetscInt vectorIndex[3];
    PetscScalar vectorValue[3];
    vectorIndex[0] = 1; vectorIndex[1] = 3; vectorIndex[2] = 6;
    vectorValue[0] = 0.1; vectorValue[1] = 0.3; vectorValue[2] = 0.5;
    VecSetValues(V, 3, vectorIndex, vectorValue, INSERT_VALUES);
    VecView(V, PETSC_VIEWER_STDOUT_WORLD);
    vecExtract(V, vectorIndex, 3);
    VecView(V, PETSC_VIEWER_STDOUT_WORLD);
    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
    PetscFinalize();
    return 0;
}

void vecExtract(Vec &V, PetscInt *nodeIndex, PetscInt n) {
    PetscScalar *V_ = new PetscScalar[n];
    VecGetValues(V, n, nodeIndex, V_);
//    VecDestroy(&V);
    VecCreate(PETSC_COMM_WORLD, &V); VecSetSizes(V, PETSC_DECIDE, n); VecSetFromOptions(V); VecSet(V, 0);

    PetscInt *vectorIndex = new PetscInt[n];
    for (int i = 0; i < n; i++) {
        vectorIndex[i] = i;
    }
    VecSetValues(V, n, vectorIndex, V_, INSERT_VALUES);
    VecAssemblyBegin(V);
    VecAssemblyEnd(V);
}