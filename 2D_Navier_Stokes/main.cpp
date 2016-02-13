static char help[] = "Solves the 2D incompressible laminar Navier-Stokes equations.\n";

#include <petscts.h>
#include <iostream>
#include <bits/algorithmfwd.h>
#include <cmath>

using namespace std;

void poseidonGetUIndex(PetscInt nUx, PetscInt nUy, PetscInt *uBoundaryIndex, PetscInt *uInteriorIndex);
void poseidonGetVIndex(PetscInt nVx, PetscInt nVy, PetscInt *vBoundaryIndex, PetscInt *vInteriorIndex);
void poseidonGetPIndex(PetscInt nPx, PetscInt nPy, PetscInt *pBoundaryIndex, PetscInt *pInteriorIndex);
void poseidonUhbarOperator(PetscInt nUx, PetscInt nUy, PetscInt nUinterior, PetscInt *uInteriorIndex, Mat &UhbarOperator);
void poseidonUhtildeOperator(PetscInt nUx, PetscInt nUy, PetscInt nUinterior, PetscInt *uInteriorIndex, Mat &UhtildeOperator);
void poseidonUvbarOperator(PetscInt nUx, PetscInt nUy, PetscInt nUinterior, PetscInt *uInteriorIndex, Mat &UvbarOperator);
void poseidonUvtildeOperator(PetscInt nUx, PetscInt nUy, PetscInt nUinterior, PetscInt *uInteriorIndex, Mat &UvtildeOperator);
void poseidonVhbarOperator(PetscInt nVx, PetscInt nVy, PetscInt nVinterior, PetscInt *vInteriorIndex, Mat &VhbarOperator);
void poseidonVhtildeOperator(PetscInt nVx, PetscInt nVy, PetscInt nVinterior, PetscInt *vInteriorIndex, Mat &VhtildeOperator);
void poseidonVvbarOperator(PetscInt nVx, PetscInt nVy, PetscInt nVinterior, PetscInt *vInteriorIndex, Mat &VvbarOperator);
void poseidonVvtildeOperator(PetscInt nVx, PetscInt nVy, PetscInt nVinterior, PetscInt *vInteriorIndex, Mat &VvtildeOperator);
void poseidondUdxOperator(PetscInt nUx, PetscInt nUy, PetscReal dx, PetscReal dy, Mat &dUdxOperator);
void poseidondUdyOperator(PetscInt nUx, PetscInt nUy, PetscReal dx, PetscReal dy, Mat &dUdyOperator);
void poseidondVdxOperator(PetscInt nVx, PetscInt nVy, PetscReal dx, PetscReal dy, Mat &dVdxOperator);
void poseidondVdyOperator(PetscInt nVx, PetscInt nVy, PetscReal dx, PetscReal dy, Mat &dVdyOperator);

int main(int argc, char **args) {
    /*
     * Petsc MPI variables
     */
    PetscMPIInt size;
    PetscInitialize(&argc,&args,(char*)0,help);
    MPI_Comm_size(PETSC_COMM_WORLD,&size);

    /*
     * Physical Properties
     */
    PetscReal dt = 0.1; // Time step

    /*
     * Domain dimensions and properties
     */
    PetscInt nx = 5, ny = 3; // Number pressure nodes in the domain, without boundaries
    PetscReal lx = 1.0, ly = 1.0; // Domain dimension in x and y directions

    PetscReal dx = lx / nx, dy = ly / ny; // Grid spacing in x and y directions
    PetscInt nUx = nx + 1, nUy = ny + 2,
             nUinterior = nUx * nUy - 2 * nUx - 2 * (nUy - 2),
             uUboundary = 2 * nUx + 2 * (nUy - 2); // Number of nodes for u-velocity variable
    PetscInt nVx = nx + 2, nVy = ny + 1,
             nVinterior = nVx * nVy - 2 * nVx - 2 * (nVy - 2),
             uVboundary = 2 * nVx + 2 * (nVy - 2); // Number of nodes for u-velocity variable
    PetscInt nPx = nx + 2, nPy = ny + 2,
             nPinterior = nPx * nPy - 2 * nPx - 2 * (nPy - 2),
             uPboundary = 2 * nPx + 2 * (nPy - 2); // Number of nodes for u-velocity variable

    /*
     * Initializing variables for u-velocity (U), v-velocity (V), and pressure (P)
     */
//    PetscScalar* U = new PetscScalar[nUx * nUy]; // u-velocity with boundaries
//    PetscScalar* V = new PetscScalar[nVx * nVy]; // v-velocity with boundaries
//    PetscScalar* P = new PetscScalar[nPx * nPy]; // pressure with boundaries
    Vec U, V, P;
    Vec Uhbar, Vhbar, Phbar;
    Vec Uvbar, Vvbar, Pvbar;
    Vec Uhtilde, Vhtilde, Phtilde;
    Vec Uvtilde, Vvtilde, Pvtilde;
    VecCreate(PETSC_COMM_WORLD, &U); VecSetSizes(U, PETSC_DECIDE, nUx * nUy); VecSetFromOptions(U);
    VecCreate(PETSC_COMM_WORLD, &V); VecSetSizes(V, PETSC_DECIDE, nVx * nVy); VecSetFromOptions(V);
    VecCreate(PETSC_COMM_WORLD, &P); VecSetSizes(P, PETSC_DECIDE, nPx * nPy); VecSetFromOptions(P);

    VecCreate(PETSC_COMM_WORLD, &Uhbar); VecSetSizes(Uhbar, PETSC_DECIDE, (nUx - 1) * (nUy - 2)); VecSetFromOptions(Uhbar);  // fixed
    VecCreate(PETSC_COMM_WORLD, &Uvbar); VecSetSizes(Uvbar, PETSC_DECIDE, nUx * (nUy - 1)); VecSetFromOptions(Uvbar); // fixed
    VecCreate(PETSC_COMM_WORLD, &Uhtilde); VecSetSizes(Uhtilde, PETSC_DECIDE, (nUx - 1) * (nUy - 2)); VecSetFromOptions(Uhtilde);  // fixed
    VecCreate(PETSC_COMM_WORLD, &Uvtilde); VecSetSizes(Uvtilde, PETSC_DECIDE, nUx * (nUy - 1)); VecSetFromOptions(Uvtilde); // fixed

    VecCreate(PETSC_COMM_WORLD, &Vhbar); VecSetSizes(Vhbar, PETSC_DECIDE, (nVx - 1) * nVy); VecSetFromOptions(Vhbar); // fixed
    VecCreate(PETSC_COMM_WORLD, &Vvbar); VecSetSizes(Vvbar, PETSC_DECIDE, (nVx - 2) * (nVy - 1)); VecSetFromOptions(Vvbar);  // fixed
    VecCreate(PETSC_COMM_WORLD, &Vhtilde); VecSetSizes(Vhtilde, PETSC_DECIDE, (nVx - 1) * nVy); VecSetFromOptions(Vhtilde);  // fixed
    VecCreate(PETSC_COMM_WORLD, &Vvtilde); VecSetSizes(Vvtilde, PETSC_DECIDE, (nVx - 2) * (nVy - 1)); VecSetFromOptions(Vvtilde);  // fixed
    /*
     * Track the boundary and interior indices for different variables
     */
    PetscInt* uBoundaryIndex = new PetscInt[2 * nUx + 2 * (nUy - 2)];
    PetscInt* uInteriorIndex = new PetscInt[nUx * nUy - 2 * nUx - 2 * (nUy - 2)];
    PetscInt* vBoundaryIndex = new PetscInt[2 * nVx + 2 * (nVy - 2)];
    PetscInt* vInteriorIndex = new PetscInt[nVx * nVy - 2 * nVx - 2 * (nVy - 2)];
    PetscInt* pBoundaryIndex = new PetscInt[2 * nPx + 2 * (nPy - 2)];
    PetscInt* pInteriorIndex = new PetscInt[nPx * nPy - 2 * nPx - 2 * (nPy - 2)];

    poseidonGetUIndex(nUx, nUy, uBoundaryIndex, uInteriorIndex);
    poseidonGetVIndex(nVx, nVy, vBoundaryIndex, vInteriorIndex);
    poseidonGetPIndex(nPx, nPy, pBoundaryIndex, pInteriorIndex);
//    PetscIntView(2 * (nx + 1) + 2 * ny, uBoundaryIndex, PETSC_VIEWER_STDOUT_WORLD);
//    PetscIntView((nx + 1) * (ny + 2) - 2 * (nx + 1) - 2 * ny, uInteriorIndex, PETSC_VIEWER_STDOUT_WORLD);
//    PetscIntView(2 * nVx + 2 * (nVy - 2), vBoundaryIndex, PETSC_VIEWER_STDOUT_WORLD);
//    PetscIntView(nVx * nVy - 2 * nVx - 2 * (nVy - 2), vInteriorIndex, PETSC_VIEWER_STDOUT_WORLD);
//    PetscIntView(2 * nPx + 2 * (nPy - 2), pBoundaryIndex, PETSC_VIEWER_STDOUT_WORLD);
//    PetscIntView(nPx * nPy - 2 * nPx - 2 * (nPy - 2), pInteriorIndex, PETSC_VIEWER_STDOUT_WORLD);

    /*
     * Calcualte Uhbar, Uhtitlde, ... operators for discretising the convective terms.
     */
    // Initializing matrices for operators
    Mat UhbarOperator; Mat UhtildeOperator;
    Mat UvbarOperator; Mat UvtildeOperator;
    Mat VhbarOperator; Mat VhtildeOperator;
    Mat VvbarOperator; Mat VvtildeOperator;

    MatCreate(PETSC_COMM_WORLD, &UhbarOperator); MatCreate(PETSC_COMM_WORLD, &UhtildeOperator);
    MatCreate(PETSC_COMM_WORLD, &UvbarOperator); MatCreate(PETSC_COMM_WORLD, &UvtildeOperator);
    MatCreate(PETSC_COMM_WORLD, &VhbarOperator); MatCreate(PETSC_COMM_WORLD, &VhtildeOperator);
    MatCreate(PETSC_COMM_WORLD, &VvbarOperator); MatCreate(PETSC_COMM_WORLD, &VvtildeOperator);

    MatSetSizes(UhbarOperator, PETSC_DECIDE, PETSC_DECIDE, (nUx - 1) * (nUy - 2), nUx * nUy);  // fixed
    MatSetSizes(UhtildeOperator, PETSC_DECIDE, PETSC_DECIDE, (nUx - 1) * (nUy - 2), nUx * nUy);  // fixed

    MatSetSizes(UvbarOperator, PETSC_DECIDE, PETSC_DECIDE, nUx * (nUy - 1), nUx * nUy); //fixed
    MatSetSizes(UvtildeOperator, PETSC_DECIDE, PETSC_DECIDE, nUx * (nUy - 1), nUx * nUy);  //fixed

    MatSetSizes(VhbarOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 1) * nVy, nVx * nVy); // fixed
    MatSetSizes(VhtildeOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 1) * nVy, nVx * nVy); // fixed

    MatSetSizes(VvbarOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 1), nVx * nVy);  // fixed
    MatSetSizes(VvtildeOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 1), nVx * nVy);  // fixed

    MatSetFromOptions(UhbarOperator); MatSetFromOptions(UhtildeOperator);
    MatSetFromOptions(UvbarOperator); MatSetFromOptions(UvtildeOperator);
    MatSetFromOptions(VhbarOperator); MatSetFromOptions(VhtildeOperator);
    MatSetFromOptions(VvbarOperator); MatSetFromOptions(VvtildeOperator);

    MatSetUp(UhbarOperator); MatSetUp(UhtildeOperator);
    MatSetUp(UvbarOperator); MatSetUp(UvtildeOperator);
    MatSetUp(VhbarOperator); MatSetUp(VhtildeOperator);
    MatSetUp(VvbarOperator); MatSetUp(VvtildeOperator);

    // Generating operators
    poseidonUhbarOperator(nUx, nUy, nUinterior, uInteriorIndex, UhbarOperator);
    poseidonUhtildeOperator(nUx, nUy, nUinterior, uInteriorIndex, UhtildeOperator);

    poseidonUvbarOperator(nUx, nUy, nUinterior, uInteriorIndex, UvbarOperator);
    poseidonUvtildeOperator(nUx, nUy, nUinterior, uInteriorIndex, UvtildeOperator);

    poseidonVhbarOperator(nVx, nVy, nVinterior, vInteriorIndex, VhbarOperator);
    poseidonVhtildeOperator(nVx, nVy, nVinterior, vInteriorIndex, VhtildeOperator);

    poseidonVvbarOperator(nVx, nVy, nVinterior, vInteriorIndex, VvbarOperator);
    poseidonVvtildeOperator(nVx, nVy, nVinterior, vInteriorIndex, VvtildeOperator);

    /*
     * Generate ddx and ddy operators
     */
    Mat dUdxOperator, dUdyOperator, dVdxOperator, dVdyOperator;

    MatCreate(PETSC_COMM_WORLD, &dUdxOperator); MatCreate(PETSC_COMM_WORLD, &dUdyOperator);
    MatCreate(PETSC_COMM_WORLD, &dVdxOperator); MatCreate(PETSC_COMM_WORLD, &dVdyOperator);

    MatSetSizes(dUdxOperator, PETSC_DECIDE, PETSC_DECIDE, (nUx - 2) * (nUy - 2), (nUx - 1) * (nUy - 2));
    MatSetSizes(dUdyOperator, PETSC_DECIDE, PETSC_DECIDE, (nUx - 2) * (nUy - 2), nUx * (nUy - 1));
    MatSetSizes(dVdxOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 2), (nVx - 1) * nVy);
    MatSetSizes(dVdyOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 2), (nVx - 2) * (nVy - 1));

    MatSetFromOptions(dUdxOperator); MatSetFromOptions(dUdyOperator);
    MatSetFromOptions(dVdxOperator); MatSetFromOptions(dVdyOperator);

    MatSetUp(dUdxOperator); MatSetUp(dUdyOperator);
    MatSetUp(dVdxOperator); MatSetUp(dVdyOperator);

    poseidondUdxOperator(nUx, nUy, dx, dy, dUdxOperator);
    poseidondUdyOperator(nUx, nUy, dx, dy, dUdyOperator);
    poseidondVdxOperator(nVx, nVy, dx, dy, dVdxOperator);
    poseidondVdyOperator(nVx, nVy, dx, dy, dVdyOperator);
    /*
     * Gamma calculation
     */
    PetscInt maxUloc, maxVloc;
    PetscReal maxUval, maxVval;
    Vec Uabs, Vabs;
    VecDuplicate(U, &Uabs); VecDuplicate(V, &Vabs);
    VecAbs(Uabs); VecAbs(Vabs);
    VecMax(Uabs, &maxUloc, &maxUval);
    VecMax(Vabs, &maxVloc, &maxVval);
    PetscReal gamma = PetscMin(1.2 * dt * PetscMax(maxUval, maxVval), 1); // Define gamma
    VecDestroy(&Uabs); VecDestroy(&Vabs);

    /*
     * Calculating the convective terms in x direction (U*)
     */
    MatMult(UhbarOperator, U, Uhbar); MatMult(UvbarOperator, U, Uvbar);
    MatMult(UhtildeOperator, U, Uhtilde); MatMult(UvtildeOperator, U, Uvtilde);
    MatMult(VhbarOperator, V, Vhbar); MatMult(VvbarOperator, V, Vvbar);
    MatMult(VhtildeOperator, V, Vhtilde); MatMult(VvtildeOperator, V, Vvtilde);

    Vec Uhbar2, UhbarXUhtitlde, UvbarXVhbar, VhbarXUvtilde, UhbarABS, VhbarABS;
    Vec Uhbar2_gammaXUhbarXUhtilde, UvbarXVhbar_gammaXVhbarUvtilde;
    Vec UvbarXVhtilde, Vvbar2, VvbarXVvtilde, UvbarABS, VvbarABS;
    Vec UvbarVhbar_gammaXUvbarVhtilde, Vvbar2_gammaXVvbarXVvtilde;

    Vec UstarRHSx, UstarRHSy;
    Vec VstarRHSx, VstarRHSy;
    Vec UstarRHS, VstarRHS;

    VecDuplicate(Uhbar, &Uhbar2);
    VecDuplicate(Uhbar, &UhbarXUhtitlde);
    VecDuplicate(Uvbar, &UvbarXVhbar);
    VecDuplicate(Vhbar, &VhbarXUvtilde);
    VecDuplicate(Uhbar, &UhbarABS);
    VecDuplicate(Vhbar, &VhbarABS);
    VecDuplicate(Uhbar, &Uhbar2_gammaXUhbarXUhtilde);
    VecDuplicate(Uvbar, &UvbarXVhbar_gammaXVhbarUvtilde);
    VecDuplicate(Uvbar, &UvbarXVhtilde);
    VecDuplicate(Vvbar, &Vvbar2);
    VecDuplicate(Vvbar, &VvbarXVvtilde);
    VecDuplicate(Uvbar, &UvbarABS);
    VecDuplicate(Vvbar, &VvbarABS);
    VecDuplicate(Uvbar, &UvbarVhbar_gammaXUvbarVhtilde);
    VecDuplicate(Vvbar, &Vvbar2_gammaXVvbarXVvtilde);

    VecCreate(PETSC_COMM_WORLD, &UstarRHSx); VecSetSizes(UstarRHSx, PETSC_DECIDE, (nUx - 2) * (nUy - 2)); VecSetFromOptions(UstarRHSx);
    VecCreate(PETSC_COMM_WORLD, &UstarRHSy); VecSetSizes(UstarRHSy, PETSC_DECIDE, (nUx - 2) * (nUy - 2)); VecSetFromOptions(UstarRHSy);
    VecCreate(PETSC_COMM_WORLD, &UstarRHS); VecSetSizes(UstarRHS, PETSC_DECIDE, (nUx - 2) * (nUy - 2)); VecSetFromOptions(UstarRHS);
    VecCreate(PETSC_COMM_WORLD, &VstarRHSx); VecSetSizes(VstarRHSx, PETSC_DECIDE, (nVx - 2) * (nVy - 2)); VecSetFromOptions(VstarRHSx);
    VecCreate(PETSC_COMM_WORLD, &VstarRHSy); VecSetSizes(VstarRHSy, PETSC_DECIDE, (nVx - 2) * (nVy - 2)); VecSetFromOptions(VstarRHSy);
    VecCreate(PETSC_COMM_WORLD, &VstarRHS); VecSetSizes(VstarRHS, PETSC_DECIDE, (nVx - 2) * (nVy - 2)); VecSetFromOptions(VstarRHS);

    VecAbs(UhbarABS);
    VecAbs(VhbarABS);
    VecAbs(UvbarABS);
    VecAbs(VvbarABS);

    VecPointwiseMult(Uhbar, Uhbar, Uhbar2);
    VecPointwiseMult(UhbarABS, Uhtilde, UhbarXUhtitlde);
    VecPointwiseMult(Uvbar, Vhbar, UvbarXVhbar);
    VecPointwiseMult(VhbarABS, Uvtilde, VhbarXUvtilde);
    VecPointwiseMult(Uvbar, Vhbar, UvbarXVhbar);
    VecPointwiseMult(UvbarABS, Vhtilde, UvbarXVhtilde);
    VecPointwiseMult(Vvbar, Vvbar, Vvbar2);
    VecPointwiseMult(VvbarABS, Vvtilde, VvbarXVvtilde);

    VecWAXPY(Uhbar2_gammaXUhbarXUhtilde, -gamma, UhbarXUhtitlde, Uhbar2);
    VecWAXPY(UvbarXVhbar_gammaXVhbarUvtilde, -gamma, VhbarXUvtilde, UvbarXVhbar);
    VecWAXPY(UvbarVhbar_gammaXUvbarVhtilde, -gamma, UvbarXVhtilde, UvbarXVhbar);
    VecWAXPY(Vvbar2_gammaXVvbarXVvtilde, -gamma, VvbarXVvtilde, Vvbar2);

    MatMult(dUdxOperator, Uhbar2_gammaXUhbarXUhtilde, UstarRHSx);
    MatMult(dUdyOperator, UvbarXVhbar_gammaXVhbarUvtilde, UstarRHSy);
    MatMult(dVdxOperator, UvbarVhbar_gammaXUvbarVhtilde, VstarRHSx);
    MatMult(dVdyOperator, Vvbar2_gammaXVvbarXVvtilde, VstarRHSy);

    VecWAXPY(UstarRHS, 1, UstarRHSx, UstarRHSy);
    VecWAXPY(VstarRHS, 1, VstarRHSx, VstarRHSy);

    VecScale(VstarRHS, -1.0 * dt);
    VecScale(UstarRHS, -1.0 * dt);

    PetscScalar *UstarRHS_array, *VstarRHS_array;

    VecGetArray(UstarRHS, &UstarRHS_array);
    VecGetArray(VstarRHS, &VstarRHS_array);
    VecSetValues(U, nUinterior, uInteriorIndex, UstarRHS_array, ADD_VALUES);
    VecSetValues(V, nVinterior, vInteriorIndex, VstarRHS_array, ADD_VALUES);
//    PetscInt vecSize, vecSize1, vecSize2;
//    VecGetSize(UstarRHS, &vecSize);
//    MatGetSize(dVdxOperator, &vecSize1, &vecSize2);
//    cout << vecSize  << endl;
//    cout << vecSize << "\t\t" << vecSize1 << "\t\t" << vecSize2 << endl;
//    VecWAXPY(UstarRHS, 1.0, Uhbar2_gammaXUhbarXUhtilde, UvbarXVhbar_gammaXVhbarUvtilde);
//    VecPointwiseMult(Uvbar, Vhbar, Vhbar);
//    VecPointwiseMult(Vhbar, Uvtilde, Uvtilde);
//    VecAXPY(Uhbar, 1.0, Uvtilde);

//    PetscInt myLoc[3] = {1, 2, 3};
//    PetscScalar *myArray;
//    VecGetArray(Uhtilde, &myArray);
//    VecSetValues(Uhbar, 3, myLoc, myArray, ADD_VALUES);

    /*
     * Destroying generated vectors and matrices
     */
    VecDestroy(&U); VecDestroy(&V); VecDestroy(&P);
    VecDestroy(&Uhbar); VecDestroy(&Vhbar);
    VecDestroy(&Uvbar); VecDestroy(&Vvbar);
    VecDestroy(&Uhtilde); VecDestroy(&Vhtilde);
    VecDestroy(&Uvtilde); VecDestroy(&Vvtilde);

    MatDestroy(&UhbarOperator); MatDestroy(&UhtildeOperator);
    MatDestroy(&UvbarOperator); MatDestroy(&UvtildeOperator);
    MatDestroy(&VhbarOperator); MatDestroy(&VhtildeOperator);
    MatDestroy(&VvbarOperator); MatDestroy(&VvtildeOperator);

    MatDestroy(&dUdxOperator); MatDestroy(&dUdyOperator);
    MatDestroy(&dVdxOperator); MatDestroy(&dVdyOperator);
    PetscFinalize();
    return 0;
}

void poseidonGetUIndex(PetscInt nUx, PetscInt nUy, PetscInt *uBoundaryIndex, PetscInt *uInteriorIndex) {
    int boundaryIndexTracker;
    int interiorIndexTracker;
    boundaryIndexTracker = 0;
    interiorIndexTracker = 0;
    // u-velocity index
    for (int i = 0; i < nUx * nUy; i++) {
        if (i < nUx) {
            uBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
        }
        else if (i%nUx == 0) {
            uBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
        }
        else if (i%nUx == nUx - 1) {
            uBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
        }
        else if ((i > nUx * (nUy - 1)) && (i < nUx * nUy)) {
            uBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
        }
        else {
            uInteriorIndex[interiorIndexTracker] = i;
            interiorIndexTracker++;
        }
    }
}

void poseidonGetVIndex(PetscInt nVx, PetscInt nVy, PetscInt *vBoundaryIndex, PetscInt *vInteriorIndex) {
    int boundaryIndexTracker;
    int interiorIndexTracker;
    boundaryIndexTracker = 0;
    interiorIndexTracker = 0;
    for (int i = 0; i < nVx * nVy; i++) {
        if (i < nVx) {
            vBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
        }
        else if (i%nVx == 0) {
            vBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
        }
        else if (i%nVx == nVx - 1) {
            vBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
        }
        else if ((i > nVx * (nVy - 1)) && (i < nVx * nVy)) {
            vBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
        }
        else {
            vInteriorIndex[interiorIndexTracker] = i;
            interiorIndexTracker++;
        }
    }
}

void poseidonGetPIndex(PetscInt nPx, PetscInt nPy, PetscInt *pBoundaryIndex, PetscInt *pInteriorIndex) {
    int boundaryIndexTracker;
    int interiorIndexTracker;
    boundaryIndexTracker = 0;
    interiorIndexTracker = 0;
    for (int i = 0; i < nPx * nPy; i++) {
        if (i < nPx) {
            pBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
        }
        else if (i%nPx == 0) {
            pBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
        }
        else if (i%nPx == nPx - 1) {
            pBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
        }
        else if ((i > nPx * (nPy - 1)) && (i < nPx * nPy)) {
            pBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
        }
        else {
            pInteriorIndex[interiorIndexTracker] = i;
            interiorIndexTracker++;
        }
    }
}

void poseidonUhbarOperator(PetscInt nUx, PetscInt nUy, PetscInt nUinterior, PetscInt *uInteriorIndex, Mat &UhbarOperator) {
    PetscInt barOperatorPosition[2]; // values that go inside the $\bar{ }$ operator
    PetscScalar barOperatorValue[2];
    PetscInt rowIndex = 0;
    barOperatorValue[0] = 0.5;
    barOperatorValue[1] = 0.5;
    for (int i = 0; i < nUinterior; i++) {
        barOperatorPosition[0] = uInteriorIndex[i] - 1;
        barOperatorPosition[1] = uInteriorIndex[i];
//        cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
        MatSetValues(UhbarOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        if (uInteriorIndex[i + 1] - uInteriorIndex[i] != 1) {
            rowIndex++;
            barOperatorPosition[0] = uInteriorIndex[i];
            barOperatorPosition[1] = uInteriorIndex[i] + 1;
//            cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
            MatSetValues(UhbarOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        }
        rowIndex++;
    }
    MatAssemblyBegin(UhbarOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(UhbarOperator, MAT_FINAL_ASSEMBLY);
}  // fixed

void poseidonUhtildeOperator(PetscInt nUx, PetscInt nUy, PetscInt nUinterior, PetscInt *uInteriorIndex, Mat &UhtildeOperator) {
    PetscInt barOperatorPosition[2]; // values that go inside the $\bar{ }$ operator
    PetscScalar barOperatorValue[2];
    PetscInt rowIndex = 0;
    barOperatorValue[0] = -0.5;
    barOperatorValue[1] = 0.5;
    for (int i = 0; i < nUinterior; i++) {
        barOperatorPosition[0] = uInteriorIndex[i] - 1;
        barOperatorPosition[1] = uInteriorIndex[i];
//        cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
        MatSetValues(UhtildeOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        if (uInteriorIndex[i + 1] - uInteriorIndex[i] != 1) {
            rowIndex++;
            barOperatorPosition[0] = uInteriorIndex[i];
            barOperatorPosition[1] = uInteriorIndex[i] + 1;
//            cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
            MatSetValues(UhtildeOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        }
        rowIndex++;
    }
    MatAssemblyBegin(UhtildeOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(UhtildeOperator, MAT_FINAL_ASSEMBLY);
}

void poseidonUvbarOperator(PetscInt nUx, PetscInt nUy, PetscInt nUinterior, PetscInt *uInteriorIndex, Mat &UvbarOperator) {
    PetscInt barOperatorPosition[2]; // values that go inside the $\bar{ }$ operator
    PetscScalar barOperatorValue[2];
    PetscInt rowIndex = 0;
    barOperatorValue[0] = 0.5;
    barOperatorValue[1] = 0.5;
    for (int i = 0; i < nUx * (nUy - 1); i++) {
        barOperatorPosition[0] = i;
        barOperatorPosition[1] = i + nUx;
//        cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
        MatSetValues(UvbarOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        rowIndex++;
    }
    MatAssemblyBegin(UvbarOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(UvbarOperator, MAT_FINAL_ASSEMBLY);
} // fixed

void poseidonUvtildeOperator(PetscInt nUx, PetscInt nUy, PetscInt nUinterior, PetscInt *uInteriorIndex, Mat &UvtildeOperator) {
    PetscInt barOperatorPosition[2]; // values that go inside the $\bar{ }$ operator
    PetscScalar barOperatorValue[2];
    PetscInt rowIndex = 0;
    barOperatorValue[0] = -0.5;
    barOperatorValue[1] = 0.5;
    for (int i = 0; i < nUx * (nUy - 1); i++) {
        barOperatorPosition[0] = i;
        barOperatorPosition[1] = i + nUx;
//        cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
        MatSetValues(UvtildeOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        rowIndex++;
    }
    MatAssemblyBegin(UvtildeOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(UvtildeOperator, MAT_FINAL_ASSEMBLY);
} // fixed

void poseidonVhbarOperator(PetscInt nVx, PetscInt nVy, PetscInt nVinterior, PetscInt *vInteriorIndex, Mat &VhbarOperator) {
    PetscInt barOperatorPosition[2]; // values that go inside the $\bar{ }$ operator
    PetscScalar barOperatorValue[2];
    PetscInt rowIndex = 0;
    barOperatorValue[0] = 0.5;
    barOperatorValue[1] = 0.5;
    for (PetscInt i = 0; i < nVx * nVy; i++) {
        if (fmod(i, nVx) == nVx - 1) {
            continue;
        }
        barOperatorPosition[0] = i;
        barOperatorPosition[1] = i + 1;
//        cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
        MatSetValues(VhbarOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        rowIndex++;
    }
    MatAssemblyBegin(VhbarOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(VhbarOperator, MAT_FINAL_ASSEMBLY);
} // fixed

void poseidonVhtildeOperator(PetscInt nVx, PetscInt nVy, PetscInt nVinterior, PetscInt *vInteriorIndex, Mat &VhtildeOperator) {
    PetscInt barOperatorPosition[2]; // values that go inside the $\bar{ }$ operator
    PetscScalar barOperatorValue[2];
    PetscInt rowIndex = 0;
    barOperatorValue[0] = -0.5;
    barOperatorValue[1] = 0.5;
    for (int i = 0; i < nVx * nVy; i++) {
        if (fmod(i, nVx) == nVx - 1) {
            continue;
        }
        barOperatorPosition[0] = i;
        barOperatorPosition[1] = i + 1;
//        cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
        MatSetValues(VhtildeOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        rowIndex++;
    }
    MatAssemblyBegin(VhtildeOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(VhtildeOperator, MAT_FINAL_ASSEMBLY);
}  // fixed

void poseidonVvbarOperator(PetscInt nVx, PetscInt nVy, PetscInt nVinterior, PetscInt *vInteriorIndex, Mat &VvbarOperator) {
    PetscInt barOperatorPosition[2]; // values that go inside the $\bar{ }$ operator
    PetscScalar barOperatorValue[2];
    PetscInt rowIndex = 0;
    barOperatorValue[0] = 0.5;
    barOperatorValue[1] = 0.5;
    for (int i = 0; i < nVinterior; i++) {
        barOperatorPosition[0] = vInteriorIndex[i] - nVx;
        barOperatorPosition[1] = vInteriorIndex[i];
//        cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
        MatSetValues(VvbarOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        if (vInteriorIndex[i] + 2 * nVx > nVx * nVy) {
            rowIndex++;
            barOperatorPosition[0] = vInteriorIndex[i];
            barOperatorPosition[1] = vInteriorIndex[i] + nVx;
//            cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
            MatSetValues(VvbarOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        }
        rowIndex++;
    }
    MatAssemblyBegin(VvbarOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(VvbarOperator, MAT_FINAL_ASSEMBLY);
}  // fixed

void poseidonVvtildeOperator(PetscInt nVx, PetscInt nVy, PetscInt nVinterior, PetscInt *vInteriorIndex, Mat &VvtildeOperator) {
    PetscInt barOperatorPosition[2]; // values that go inside the $\bar{ }$ operator
    PetscScalar barOperatorValue[2];
    PetscInt rowIndex = 0;
    barOperatorValue[0] = -0.5;
    barOperatorValue[1] = 0.5;
    for (int i = 0; i < nVinterior; i++) {
        barOperatorPosition[0] = vInteriorIndex[i] - nVx;
        barOperatorPosition[1] = vInteriorIndex[i];
//        cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
        MatSetValues(VvtildeOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        if (vInteriorIndex[i] + 2 * nVx > nVx * nVy) {
            rowIndex++;
            barOperatorPosition[0] = vInteriorIndex[i];
            barOperatorPosition[1] = vInteriorIndex[i] + nVx;
//            cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
            MatSetValues(VvtildeOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        }
        rowIndex++;
    }
    MatAssemblyBegin(VvtildeOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(VvtildeOperator, MAT_FINAL_ASSEMBLY);
}  // fixed

void poseidondUdxOperator(PetscInt nUx, PetscInt nUy, PetscReal dx, PetscReal dy, Mat &dUdxOperator) {
    PetscInt operatorPosition[2];
    PetscScalar operatorValue[2];
    PetscInt matRow, matColumn;
    PetscInt rowIndex = 0;
    MatGetSize(dUdxOperator, &matRow, &matColumn);
    operatorValue[0] = -1.0 / dx;
    operatorValue[1] = 1.0 / dx;
    for (int i = 0; i < matColumn; i++) {
        if ((fmod(i, nUx - 1) == nUx - 2)) {
            continue;
        }
        operatorPosition[0] = i;
        operatorPosition[1] = i + 1;
//        cout << rowIndex << "\t" << operatorPosition[0] << "\t" << operatorPosition[1] << endl;
        MatSetValues(dUdxOperator, 1, &rowIndex, 2, operatorPosition, operatorValue, INSERT_VALUES);
        rowIndex++;
    }
    MatAssemblyBegin(dUdxOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(dUdxOperator, MAT_FINAL_ASSEMBLY);
}

void poseidondUdyOperator(PetscInt nUx, PetscInt nUy, PetscReal dx, PetscReal dy, Mat &dUdyOperator) {
    PetscInt operatorPosition[2];
    PetscScalar operatorValue[2];
    PetscInt matRow, matColumn;
    PetscInt rowIndex = 0;
    MatGetSize(dUdyOperator, &matRow, &matColumn);
    operatorValue[0] = -1.0 / dy;
    operatorValue[1] = 1.0 / dy;
    for (int i = 0; i < matColumn; i++) {
        if (i + nUx > matColumn - 1) {
            continue;
        }
        if (fmod(i, nUx) == 0) {
            continue;
        }
        if (fmod(i, nUx) == nUx - 1) {
            continue;
        }
        operatorPosition[0] = i;
        operatorPosition[1] = i + nUx;
//        cout << rowIndex << "\t" << operatorPosition[0] << "\t" << operatorPosition[1] << endl;
        MatSetValues(dUdyOperator, 1, &rowIndex, 2, operatorPosition, operatorValue, INSERT_VALUES);
        rowIndex++;
    }
    MatAssemblyBegin(dUdyOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(dUdyOperator, MAT_FINAL_ASSEMBLY);
}

void poseidondVdxOperator(PetscInt nVx, PetscInt nVy, PetscReal dx, PetscReal dy, Mat &dVdxOperator) {
    PetscInt operatorPosition[2];
    PetscScalar operatorValue[2];
    PetscInt matRow, matColumn;
    PetscInt rowIndex = 0;
    MatGetSize(dVdxOperator, &matRow, &matColumn);
    operatorValue[0] = -1.0 / dx;
    operatorValue[1] = 1.0 / dx;
    for (int i = 0; i < matColumn; i++) {
        if ((fmod(i, nVx - 1) == nVx - 2)) {
            continue;
        }
        if (i < nVx - 1) {
            continue;
        }
        if (i > (nVx - 1) * (nVy - 1) - 1) {
            continue;
        }
        operatorPosition[0] = i;
        operatorPosition[1] = i + 1;
//        cout << rowIndex << "\t" << operatorPosition[0] << "\t" << operatorPosition[1] << endl;
        MatSetValues(dVdxOperator, 1, &rowIndex, 2, operatorPosition, operatorValue, INSERT_VALUES);
        rowIndex++;
    }
    MatAssemblyBegin(dVdxOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(dVdxOperator, MAT_FINAL_ASSEMBLY);
}

void poseidondVdyOperator(PetscInt nVx, PetscInt nVy, PetscReal dx, PetscReal dy, Mat &dVdyOperator) {
    PetscInt operatorPosition[2];
    PetscScalar operatorValue[2];
    PetscInt matRow, matColumn;
    PetscInt rowIndex = 0;
    MatGetSize(dVdyOperator, &matRow, &matColumn);
    operatorValue[0] = -1.0 / dy;
    operatorValue[1] = 1.0 / dy;
    for (int i = 0; i < matColumn; i++) {
        if (i + nVx - 1 > matColumn - 1) {
            continue;
        }
        operatorPosition[0] = i;
        operatorPosition[1] = i + nVx - 1;
//        cout << rowIndex << "\t" << operatorPosition[0] << "\t" << operatorPosition[1] << endl;
        MatSetValues(dVdyOperator, 1, &rowIndex, 2, operatorPosition, operatorValue, INSERT_VALUES);
        rowIndex++;
    }
    MatAssemblyBegin(dVdyOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(dVdyOperator, MAT_FINAL_ASSEMBLY);
}