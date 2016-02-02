static char help[] = "Solves the 2D incompressible laminar Navier-Stokes equations.\n";

#include <petscts.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
    /*
     * Define variables for PETSc
     */
    TS ts; /* Nonlinear solver */
    PetscMPIInt size;

    /*
     * Define variable for problem
     */
    Mat A; // Jacobian matrix
    PetscInt nx = 5, ny = 3; // Number of nodes in x and y directions (pressure nodes)
//    PetscScalar* U = new PetscScalar[(nx - 1) * (ny)]; // u-velocity
//    PetscScalar* V = new PetscScalar[(nx) * (ny - 1)]; // v-velocity
//    PetscScalar* P = new PetscScalar[(nx) * (ny)]; // pressure
    PetscInt N = (nx - 1) * (ny) + (nx) * (ny - 1) + (nx) * (ny); // Total number of computational points
    Vec UVP; // Holds all the variables
    PetscReal ftime = 0.5;
    PetscInt steps; // Number of timesteps used
    /*
     * Initialize problem
     */
    PetscInitialize(&argc, &argv, NULL, help);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    /*
     * Generate necessary matrix and vectors, solve same ODE on every process
     */
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);
    MatSetFromOptions(A);
    MatSetUp(A);

    /*
     * Create timestepping solver context
     */
    TSCreate(PETSC_COMM_WORLD, &ts);
    TSSetType(ts, TSBEULER);
//    TSSetRHSFunction(ts, RHSFunction, NULL);
//    TSSetIFunction(ts, NULL, IFunction, NULL);
//    TSSetIJacobian(ts, A, A, IJacobian, NULL);
    TSSetDuration(ts, PETSC_DEFAULT, ftime);

    /*
     * Set initial conditions
     */
    VecCreate(PETSC_COMM_WORLD, &UVP);
    VecSetSizes(UVP, PETSC_DECIDE, N);
    VecSetFromOptions(UVP);
    VecSet(UVP, 0);

    TSSetInitialTimeStep(ts, 0.0, 0.01);

    /*
     * Set runtime options
     */
    TSSetFromOptions(ts);

    /*
     * Solve nonlinear system
     */
    TSSolve(ts, UVP);
    TSGetSolveTime(ts, &ftime);
    TSGetTimeStepNumber(ts, &steps);
    VecView(UVP, PETSC_VIEWER_STDOUT_WORLD);

    /*
     * Free work space
     */
    MatDestroy(&A);
    VecDestroy(&UVP);
    TSDestroy(&ts);

    PetscFinalize();
    return 0;
}