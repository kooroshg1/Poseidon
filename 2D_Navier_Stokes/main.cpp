static char help[] = "Solves the 2D incompressible laminar Navier-Stokes equations.\n";

#include <petscts.h>
#include <iostream>
#include <bits/algorithmfwd.h>

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

    VecCreate(PETSC_COMM_WORLD, &Uhbar); VecSetSizes(Uhbar, PETSC_DECIDE, (nUx - 1) * (nUy - 2)); VecSetFromOptions(Uhbar);
    VecCreate(PETSC_COMM_WORLD, &Uvbar); VecSetSizes(Uvbar, PETSC_DECIDE, (nUx - 2) * (nUy - 1)); VecSetFromOptions(Uvbar);
    VecCreate(PETSC_COMM_WORLD, &Uhtilde); VecSetSizes(Uhtilde, PETSC_DECIDE, (nUx - 1) * (nUy - 2)); VecSetFromOptions(Uhtilde);
    VecCreate(PETSC_COMM_WORLD, &Uvtilde); VecSetSizes(Uvtilde, PETSC_DECIDE, (nUx - 2) * (nUy - 1)); VecSetFromOptions(Uvtilde);

    VecCreate(PETSC_COMM_WORLD, &Vhbar); VecSetSizes(Vhbar, PETSC_DECIDE, (nVx - 1) * (nVy - 2)); VecSetFromOptions(Vhbar);
    VecCreate(PETSC_COMM_WORLD, &Vvbar); VecSetSizes(Vvbar, PETSC_DECIDE, (nVx - 2) * (nVy - 1)); VecSetFromOptions(Vvbar);
    VecCreate(PETSC_COMM_WORLD, &Vhtilde); VecSetSizes(Vhtilde, PETSC_DECIDE, (nVx - 1) * (nVy - 2)); VecSetFromOptions(Vhtilde);
    VecCreate(PETSC_COMM_WORLD, &Vvtilde); VecSetSizes(Vvtilde, PETSC_DECIDE, (nVx - 2) * (nVy - 1)); VecSetFromOptions(Vvtilde);
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
     * Generate tilde and bar operators for treating nonlinear terms
     */
    Mat UhbarOperator; Mat UhtildeOperator;
    Mat UvbarOperator; Mat UvtildeOperator;
    Mat VhbarOperator; Mat VhtildeOperator;
    Mat VvbarOperator; Mat VvtildeOperator;

    MatCreate(PETSC_COMM_WORLD, &UhbarOperator); MatCreate(PETSC_COMM_WORLD, &UhtildeOperator);
    MatCreate(PETSC_COMM_WORLD, &UvbarOperator); MatCreate(PETSC_COMM_WORLD, &UvtildeOperator);
    MatCreate(PETSC_COMM_WORLD, &VhbarOperator); MatCreate(PETSC_COMM_WORLD, &VhtildeOperator);
    MatCreate(PETSC_COMM_WORLD, &VvbarOperator); MatCreate(PETSC_COMM_WORLD, &VvtildeOperator);

    MatSetSizes(UhbarOperator, PETSC_DECIDE, PETSC_DECIDE, (nUx - 1) * (nUy - 2), nUx * nUy);
    MatSetSizes(UhtildeOperator, PETSC_DECIDE, PETSC_DECIDE, (nUx - 1) * (nUy - 2), nUx * nUy);

    MatSetSizes(UvbarOperator, PETSC_DECIDE, PETSC_DECIDE, (nUx - 2) * (nUy - 1), nUx * nUy);
    MatSetSizes(UvtildeOperator, PETSC_DECIDE, PETSC_DECIDE, (nUx - 2) * (nUy - 1), nUx * nUy);

    MatSetSizes(VhbarOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 1) * (nVy - 2), nVx * nVy);
    MatSetSizes(VhtildeOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 1) * (nVy - 2), nVx * nVy);

    MatSetSizes(VvbarOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 1), nVx * nVy);
    MatSetSizes(VvtildeOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 1), nVx * nVy);

    MatSetFromOptions(UhbarOperator); MatSetFromOptions(UhtildeOperator);
    MatSetFromOptions(UvbarOperator); MatSetFromOptions(UvtildeOperator);
    MatSetFromOptions(VhbarOperator); MatSetFromOptions(VhtildeOperator);
    MatSetFromOptions(VvbarOperator); MatSetFromOptions(VvtildeOperator);

    MatSetUp(UhbarOperator); MatSetUp(UhtildeOperator);
    MatSetUp(UvbarOperator); MatSetUp(UvtildeOperator);
    MatSetUp(VhbarOperator); MatSetUp(VhtildeOperator);
    MatSetUp(VvbarOperator); MatSetUp(VvtildeOperator);

    poseidonUhbarOperator(nUx, nUy, nUinterior, uInteriorIndex, UhbarOperator);
    poseidonUhtildeOperator(nUx, nUy, nUinterior, uInteriorIndex, UhtildeOperator);

    poseidonUvbarOperator(nUx, nUy, nUinterior, uInteriorIndex, UvbarOperator);
    poseidonUvtildeOperator(nUx, nUy, nUinterior, uInteriorIndex, UvtildeOperator);

    poseidonVhbarOperator(nVx, nVy, nVinterior, vInteriorIndex, VhbarOperator);
    poseidonVhtildeOperator(nVx, nVy, nVinterior, vInteriorIndex, VhtildeOperator);

    poseidonVvbarOperator(nVx, nVy, nVinterior, vInteriorIndex, VvbarOperator);
    poseidonVvtildeOperator(nVx, nVy, nVinterior, vInteriorIndex, VvtildeOperator);

//    MatView(VvtildeOperator, PETSC_VIEWER_STDOUT_WORLD);
    // Calculate Uhbar, Uhtilde, ...
    MatMult(UhbarOperator, U, Uhbar); MatMult(UvbarOperator, U, Uvbar);
    MatMult(UhtildeOperator, U, Uhtilde); MatMult(UvtildeOperator, U, Uvtilde);
    MatMult(VhbarOperator, V, Vhbar); MatMult(VvbarOperator, V, Vvbar);
    MatMult(VhtildeOperator, V, Vhtilde); MatMult(VvtildeOperator, V, Vvtilde);

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

    VecPointwiseMult(Uhbar, Uhbar, Uhbar);
    VecPointwiseMult(Uhbar, Uhtilde, Uhtilde);
    VecPointwiseMult(Uvbar, Uhbar, Uhbar);
    VecPointwiseMult(Vhbar, Uvtilde, Uhtilde);
    VecAXPY(Uhbar, 1.0, Uhtilde);
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
}

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
    for (int i = 0; i < nUinterior; i++) {
        barOperatorPosition[0] = uInteriorIndex[i] - nUx;
        barOperatorPosition[1] = uInteriorIndex[i];
//        cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
        MatSetValues(UvbarOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        if (uInteriorIndex[i] + 2 * nUx > nUx * nUy) {
            rowIndex++;
            barOperatorPosition[0] = uInteriorIndex[i];
            barOperatorPosition[1] = uInteriorIndex[i] + nUx;
//            cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
            MatSetValues(UvbarOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        }
        rowIndex++;
    }
    MatAssemblyBegin(UvbarOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(UvbarOperator, MAT_FINAL_ASSEMBLY);
}

void poseidonUvtildeOperator(PetscInt nUx, PetscInt nUy, PetscInt nUinterior, PetscInt *uInteriorIndex, Mat &UvtildeOperator) {
    PetscInt barOperatorPosition[2]; // values that go inside the $\bar{ }$ operator
    PetscScalar barOperatorValue[2];
    PetscInt rowIndex = 0;
    barOperatorValue[0] = -0.5;
    barOperatorValue[1] = 0.5;
    for (int i = 0; i < nUinterior; i++) {
        barOperatorPosition[0] = uInteriorIndex[i] - nUx;
        barOperatorPosition[1] = uInteriorIndex[i];
//        cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
        MatSetValues(UvtildeOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        if (uInteriorIndex[i] + 2 * nUx > nUx * nUy) {
            rowIndex++;
            barOperatorPosition[0] = uInteriorIndex[i];
            barOperatorPosition[1] = uInteriorIndex[i] + nUx;
//            cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
            MatSetValues(UvtildeOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        }
        rowIndex++;
    }
    MatAssemblyBegin(UvtildeOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(UvtildeOperator, MAT_FINAL_ASSEMBLY);
}

void poseidonVhbarOperator(PetscInt nVx, PetscInt nVy, PetscInt nVinterior, PetscInt *vInteriorIndex, Mat &VhbarOperator) {
    PetscInt barOperatorPosition[2]; // values that go inside the $\bar{ }$ operator
    PetscScalar barOperatorValue[2];
    PetscInt rowIndex = 0;
    barOperatorValue[0] = 0.5;
    barOperatorValue[1] = 0.5;
    for (int i = 0; i < nVinterior; i++) {
        barOperatorPosition[0] = vInteriorIndex[i] - 1;
        barOperatorPosition[1] = vInteriorIndex[i];
//        cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
        MatSetValues(VhbarOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        if (vInteriorIndex[i + 1] - vInteriorIndex[i] != 1) {
            rowIndex++;
            barOperatorPosition[0] = vInteriorIndex[i];
            barOperatorPosition[1] = vInteriorIndex[i] + 1;
//            cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
            MatSetValues(VhbarOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        }
        rowIndex++;
    }
    MatAssemblyBegin(VhbarOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(VhbarOperator, MAT_FINAL_ASSEMBLY);
}

void poseidonVhtildeOperator(PetscInt nVx, PetscInt nVy, PetscInt nVinterior, PetscInt *vInteriorIndex, Mat &VhtildeOperator) {
    PetscInt barOperatorPosition[2]; // values that go inside the $\bar{ }$ operator
    PetscScalar barOperatorValue[2];
    PetscInt rowIndex = 0;
    barOperatorValue[0] = -0.5;
    barOperatorValue[1] = 0.5;
    for (int i = 0; i < nVinterior; i++) {
        barOperatorPosition[0] = vInteriorIndex[i] - 1;
        barOperatorPosition[1] = vInteriorIndex[i];
//        cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
        MatSetValues(VhtildeOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        if (vInteriorIndex[i + 1] - vInteriorIndex[i] != 1) {
            rowIndex++;
            barOperatorPosition[0] = vInteriorIndex[i];
            barOperatorPosition[1] = vInteriorIndex[i] + 1;
//            cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
            MatSetValues(VhtildeOperator, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        }
        rowIndex++;
    }
    MatAssemblyBegin(VhtildeOperator, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(VhtildeOperator, MAT_FINAL_ASSEMBLY);
}

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
}

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
}