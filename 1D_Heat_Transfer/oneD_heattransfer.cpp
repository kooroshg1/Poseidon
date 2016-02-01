static char help[] = "Solves 1D heat transfer problem.\n\n";

#include <petscksp.h>
#include <iostream>
using namespace std;

int main(int argc, char **args) {
    Vec         x, b, u; // Approximate solution, RHS, exact solution
    Mat         A; // Linear system matrix
    KSP         ksp; // Linear solver
    PetscInt    i, n = 10, col[3], its;
    PetscMPIInt size;

    PetscScalar value[3];

    PetscInitialize(&argc, &args, NULL, help);

    /*
     * Create Vectors.
     */
    VecCreate(PETSC_COMM_WORLD, &x);
    PetscObjectSetName((PetscObject) x, "solution");
    VecSetSizes(x, PETSC_DECIDE, n);
    VecSetFromOptions(x);
    VecDuplicate(x, &b);
    VecDuplicate(x, &u);

    /*
     * Create matrices
     */
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);
    MatSetFromOptions(A);
    MatSetUp(A);

    /*
     * Assemble matrix
     */
    value[0] = 1.0;     value[1] = -2.0;    value[2] = 1.0;
    for (i=1; i<n-1; i++) {
        col[0] = i - 1;     col[1] = i;     col[2] = i + 1;
        MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES);
    }
    i = n - 1; col[0] = n - 2; col[1] = n - 1;
    MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES);
    i = 0; col[0] = 0; col[1] = 1; value[0] = -2; value[1] = 1.0;
    MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES);
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    /*
     * Assigning boundary conditions
     */
    PetscScalar BC[2];
    PetscInt iBC[2];
    BC[0] = -3; BC[1] = -1;
    iBC[0] = 0; iBC[1] = 9;
    i = 2;
    VecSetValues(b, i, iBC, BC, INSERT_VALUES);
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    /*
     * Create linear solver
     */
    KSPCreate(PETSC_COMM_WORLD, &ksp);

    /*
     * Set operators
     */
    KSPSetOperators(ksp, A, A);

    KSPSolve(ksp, b, x);
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    PetscFinalize();
    return 0;
}
