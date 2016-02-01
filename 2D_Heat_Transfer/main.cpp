#include <iostream>
#include <petscksp.h>
#include <ctime>

using namespace std;

void PetscLaplacianOperator(PetscInt n, Mat &A);

int main(int argc, char **args) {
    clock_t begin = clock();

    Vec RHS, sol; // Solution, RHS, and place holder
    Mat A; // Linear system matrix
    KSP ksp; // Solver
    PetscInt n = 100; // Number of nodes in x and y directions
    PetscInt i;
    PetscInt col[5]; // Place holnder for column index
    PetscScalar lx = 1.0, ly = 1.0; // Domain length in x and y directions
    PetscScalar value[5]; // Place holder for coefficients in A matrix
    PetscMPIInt size;
    PetscViewer matlabViewer;

    // Initialize MPI
    size = 1;
    PetscInitialize(&argc, &args, NULL, NULL);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    /*
     * Create matrices
     */
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, pow(n, 2), pow(n, 2));
    MatSetFromOptions(A);
    MatSetUp(A);
    /*
     * Generate Laplacian operator
     */
    PetscLaplacianOperator(n, A);

    /*
     * Track indices
     */
    PetscInt northIndex[n - 2], southIndex[n - 2], eastIndex[n - 2], westIndex[n - 2], interiorIndex[(n - 2) * (n - 2)];
    int northIndexTracker = 0, southIndexTracker = 0, eastIndexTracker = 0, westIndexTracker = 0, interiorIndexTracker = 0;
    for (i=0; i < n*n; i++) {
        if ((i < n) && (i != 0) && (i != n - 1)) {
            southIndex[southIndexTracker] = i;
            southIndexTracker++;
        }
        else if ((i%n == 0) && (i > 0) && (i < n * (n - 1))) {
            westIndex[westIndexTracker] = i;
            westIndexTracker++;
        }
        else if ((i%n == 4) && (i > n - 1) && (i < n * (n - 1))) {
            eastIndex[eastIndexTracker] = i;
            eastIndexTracker++;
        }
        else if ((i > n * (n - 1)) && (i < n * n - 1)) {
            northIndex[northIndexTracker] = i;
            northIndexTracker++;
        }
        else if ((i != 0) && (i != n - 1) && (i != n * (n - 1)) && (i != n * n - 1)) {
            interiorIndex[interiorIndexTracker] = i;
            interiorIndexTracker++;
        }
    }

    /*
     * Boundary conditions value place holders
     */

    PetscScalar* southBoundaryValue = new PetscScalar[n - 2]; fill_n(southBoundaryValue, n - 2, 1);
    PetscScalar* northBoundaryValue = new PetscScalar[n - 2]; fill_n(northBoundaryValue, n - 2, 10);
    /*
     * Create vectors
     */
    VecCreate(MPI_COMM_WORLD, &RHS);
    VecSetSizes(RHS, PETSC_DECIDE, pow(n, 2));
    VecSetFromOptions(RHS);

    VecSetValues(RHS, n - 2, southIndex, southBoundaryValue, INSERT_VALUES);
    VecSetValues(RHS, n - 2, northIndex, northBoundaryValue, INSERT_VALUES);
    // Setting values at the corners
    VecSetValue(RHS, 0, southBoundaryValue[0], INSERT_VALUES);
    VecSetValue(RHS, n - 1, southBoundaryValue[0], INSERT_VALUES);
    VecSetValue(RHS, n * (n - 1), northBoundaryValue[0], INSERT_VALUES);
    VecSetValue(RHS, n * n - 1, northBoundaryValue[0], INSERT_VALUES);
    VecAssemblyBegin(RHS);
    VecAssemblyEnd(RHS);

    /*
     * Modify the stiffness matrix for zero gradients on left and right boundaries.
     */
    value[0] = 1;
    PetscInt cornerIndex[4] = {0, n - 1, n * (n - 1), n * n - 1};
    for (i=0; i < n - 2; i++) { // West and east boundaries
        MatSetValues(A, 1, &westIndex[i], 1, &westIndex[i], value, ADD_VALUES);
        MatSetValues(A, 1, &eastIndex[i], 1, &eastIndex[i], value, ADD_VALUES);
    }
    for (i=0; i < 4; i++) { // Corner points
        MatSetValues(A, 1, &cornerIndex[i], 1, &cornerIndex[i], value, ADD_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    /*
     * Define solver
     */
    VecScale(RHS, -1);
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetFromOptions(ksp);
    VecDuplicate(RHS, &sol); // Place holder for solution
    KSPSolve(ksp, RHS, sol);

    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "solution.output", &matlabViewer);
    PetscViewerSetFormat(matlabViewer, PETSC_VIEWER_ASCII_MATLAB);
    VecView(sol, matlabViewer);

    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "operator.output", &matlabViewer);
    PetscViewerSetFormat(matlabViewer, PETSC_VIEWER_ASCII_MATLAB);
    MatView(A, matlabViewer);
    // Finalize the MPI secion
    KSPDestroy(&ksp);
    VecDestroy(&RHS);
    VecDestroy(&sol);
    MatDestroy(&A);
    PetscFinalize();

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << elapsed_secs << endl;

    return 0;
}

void PetscLaplacianOperator(PetscInt n, Mat &A) {
    PetscInt col[5], i;
    PetscScalar value[5];

    /*
     * Assemble matrix
     */
    value[0] = 1.0; value[1] = 1; value[2] = -4; value[3] = 1; value[4] = 1;
    for (i=0; i<n*n; i++) {
        if ((i % n == 0) && (i < n - 1)) { // Element at south-west corner
            col[0] = i;
            col[1] = i + 1;
            col[2] = i + n;
            value[0] = -4;
            value[1] = 1;
            value[2] = 1;
            MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES);
        }
        else if ((i < n) && (i != n - 1)) { // Elements on south row
            col[0] = i - 1;
            col[1] = i;
            col[2] = i + 1;
            col[3] = i + n;
            value[0] = 1;
            value[1] = -4;
            value[2] = 1;
            value[3] = 1;
            MatSetValues(A, 1, &i, 4, col, value, INSERT_VALUES);
        }
        else if (i == n - 1) { // Element at south-east corner
            col[0] = i - 1;
            col[1] = i;
            col[2] = i + n;
            value[0] = 1;
            value[1] = -4;
            value[2] = 1;
            MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES);
        }
        else if ((i % n == 0) && (i >= n * (n - 1))) { // Element at north-west corner
            col[0] = i - n;
            col[1] = i;
            col[2] = i + 1;
            value[0] = 1;
            value[1] = -4;
            value[2] = 1;
            MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES);
        }
        else if ((i > n * n - n) && (i < n * n - 1)) { // Elements on north boundary
            col[0] = i - n;
            col[1] = i - 1;
            col[2] = i;
            col[3] = i + 1;
            value[0] = 1;
            value[1] = 1;
            value[2] = -4;
            value[3] = 1;
            MatSetValues(A, 1, &i, 4, col, value, INSERT_VALUES);
        }
        else if (i == n * n - 1) { // Elements on North east corner
            col[0] = i - n;
            col[1] = i - 1;
            col[2] = i;
            value[0] = 1;
            value[1] = 1;
            value[2] = -4;
            MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES);
        }
        else if ((i % n == 0) && (i != 0) && (i != n * (n - 1))) { // West wall
            col[0] = i - n;
            col[1] = i;
            col[2] = i + 1;
            col[3] = i + n;
            value[0] = 1;
            value[1] = -4;
            value[2] = 1;
            value[3] = 1;
            MatSetValues(A, 1, &i, 4, col, value, INSERT_VALUES);
        }
        else if ((i % n == n - 1) && (i != 4) && (i != n * n - 1)) { // East wall
            col[0] = i - n;
            col[1] = i - 1;
            col[2] = i;
            col[3] = i + n;
            value[0] = 1;
            value[1] = 1;
            value[2] = -4;
            value[3] = 1;
            MatSetValues(A, 1, &i, 4, col, value, INSERT_VALUES);
        }
        else {
            col[0] = i - n;
            col[1] = i - 1;
            col[2] = i;
            col[3] = i + 1;
            col[4] = i + n;
            value[0] = 1;
            value[1] = 1;
            value[2] = -4;
            value[3] = 1;
            value[4] = 1;
            MatSetValues(A, 1, &i, 5, col, value, INSERT_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

