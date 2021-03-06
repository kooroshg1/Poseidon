static char help[] = "Solves the 2D incompressible laminar Navier-Stokes equations.\n";

#include <petscts.h>
#include <iostream>
#include <bits/algorithmfwd.h>
#include <cmath>

using namespace std;

void poseidonGetUIndex(PetscInt nUx, PetscInt nUy, PetscInt *uBoundaryIndex, PetscInt *uInteriorIndex,
                       PetscInt *uWestBoundaryIndex, PetscInt *uWestGhostBoundaryIndex,
                       PetscInt *uNorthBoundaryIndex, PetscInt *uNorthGhostBoundaryIndex,
                       PetscInt *uEastBoundaryIndex, PetscInt *uEastGhostBoundaryIndex,
                       PetscInt *uSouthBoundaryIndex, PetscInt *uSouthGhostBoundaryIndex);
void poseidonGetVIndex(PetscInt nVx, PetscInt nVy, PetscInt *vBoundaryIndex, PetscInt *vInteriorIndex,
                       PetscInt *vWestBoundaryIndex, PetscInt *vWestGhostBoundaryIndex,
                       PetscInt *vNorthBoundaryIndex, PetscInt *vNorthGhostBoundaryIndex,
                       PetscInt *vEastBoundaryIndex, PetscInt *vEastGhostBoundaryIndex,
                       PetscInt *vSouthBoundaryIndex, PetscInt *vSouthGhostBoundaryIndex);
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
void poseidonLaplacianOperator(Mat &A, PetscInt nx, PetscInt ny, PetscReal dx, PetscReal dy);
void poseidondUdX(PetscInt nUx, PetscInt nUy, PetscReal dx, PetscReal dy, Mat &dUdX);
void poseidondVdY(PetscInt nVx, PetscInt nVy, PetscReal dx, PetscReal dy, Mat &dVdX);
void poseidonLaplacianNeumann(PetscInt nx, PetscInt ny, PetscReal dx, PetscReal dy, Mat &L);

void showVectorSize(Vec A);
void showMatrixSize(Mat A);
void writeMat(Mat A);
void writeVec(Vec A);
void Xbar_tilde(PetscInt nx, PetscInt ny, Mat &Lhbar, PetscInt *skipCode, string indicator, string direction);
PetscScalar calculateGamma(PetscScalar dt, Vec &U, Vec &V);
void diffOperator(PetscInt nx, PetscInt ny, PetscScalar ds, Mat Ldds, PetscInt *skipCode, string direction);

void calculateStarVariable(Mat LUhbar, Mat LUhtilde, Mat LUvbar, Mat LUvtilde, Mat LdUsdX, Mat LdUsdY, PetscInt nUx, PetscInt nUy, Vec U, Vec &Us, PetscInt *uInteriorIndex,
                           Mat LVhbar, Mat LVhtilde, Mat LVvbar, Mat LVvtilde, Mat LdVsdX, Mat LdVsdY, PetscInt nVx, PetscInt nVy, Vec V, Vec &Vs, PetscInt *vInteriorIndex,
                           PetscScalar gamma, PetscReal dt);
void vecExtract(Vec &V, PetscInt *nodeIndex, PetscInt n);
void laplacianOperator(Mat &L, PetscInt nx, PetscInt ny, PetscScalar dx, PetscScalar dy, PetscInt *neumannBoundary, string variable);

void assignBoundaryCondition(Vec &X, PetscInt nx, PetscInt ny,
                             PetscInt *southBoundaryIndex, PetscInt *southBoundaryGhostIndex,
                             PetscInt *westBoundaryIndex, PetscInt *westBoundaryGhostIndex,
                             PetscInt *northBoundaryIndex, PetscInt *northBoundaryGhostIndex,
                             PetscInt *eastBoundaryIndex, PetscInt *eastBoundaryGhostIndex,
                             PetscInt *isNeumannBoundary, PetscScalar *dirichletBoundary,
                             PetscScalar dx, PetscScalar dy,
                             Vec &XssBoundary, PetscReal dt, PetscReal Re, string variableType);

void calculatePressure(Vec &P, Mat LdUdX, Vec U, Mat LdVdY, Vec V, Mat Lp, PetscReal dt);

void correctVelocity(Vec Uss, Vec Vss, Mat LdPdX, Mat LdPdY, Vec P, PetscReal dt,
                     PetscInt *uInteriorIndex, PetscInt *vInteriorIndex, Vec &U, Vec &V);

int main(int argc, char **args) {
    /*
     * Petsc MPI variables
     */
    PetscMPIInt size;
    PetscInitialize(&argc, &args, (char *) 0, help);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    /*
     * Physical Properties
     */
    PetscReal dt = 0.01; // Time step

    /*
     * Flow properties
     */
    PetscReal Re = 100;

    /*
     * Boundary condition type and values
     * xIsNeumannBoundary defined is the correponding wall has neumann (1) or dirichlet (0) boundary conditions.
     * These are ordered as {S, W, N, E} walls.
     * The code currently can handle zero neumann boundary conditions.
     * The dirichlet boundary values are stored in xDirichletBoundary.
     * The order is {S, W, N, E}
     */
    PetscInt uIsNeumannBoundary[4] = {0, 0, 0, 0};
    PetscScalar uDirichletBoundary[4] = {1, 2, 3, 4};
    PetscInt vIsNeumannBoundary[4] = {0, 0, 0, 0};
    PetscScalar vDirichletBoundary[4] = {10, 20, 30, 40};
    PetscInt pIsNeumannBoundary[4] = {1, 1, 1, 1};

    /*
     * Domain dimensions and properties
     */
    PetscInt nx = 5, ny = 3; // Number pressure nodes in the domain, without boundaries
    PetscReal lx = 1.0, ly = 1.0; // Domain dimension in x and y directions

    PetscReal dx = lx / nx, dy = ly / ny; // Grid spacing in x and y directions

    PetscInt nUx = nx + 1, nUy = ny + 2,
            nUinterior = nUx * nUy - 2 * nUx - 2 * (nUy - 2),
            nUboundary = 2 * nUx + 2 * (nUy - 2); // Number of nodes for u-velocity variable
    PetscInt nVx = nx + 2, nVy = ny + 1,
            nVinterior = nVx * nVy - 2 * nVx - 2 * (nVy - 2),
            nVboundary = 2 * nVx + 2 * (nVy - 2); // Number of nodes for u-velocity variable
    PetscInt nPx = nx + 2, nPy = ny + 2,
            nPinterior = nPx * nPy - 2 * nPx - 2 * (nPy - 2),
            nPboundary = 2 * nPx + 2 * (nPy - 2); // Number of nodes for u-velocity variable

    /*
     * Track the boundary and interior indices for different variables
     */
    PetscInt *uBoundaryIndex = new PetscInt[2 * nUx + 2 * (nUy - 2)]();
    PetscInt *uInteriorIndex = new PetscInt[nUx * nUy - 2 * nUx - 2 * (nUy - 2)]();
    PetscInt *uWestBoundaryIndex = new PetscInt[nUy - 2]();
    PetscInt *uWestGhostBoundaryIndex = new PetscInt[nUy - 2]();
    PetscInt *uNorthBoundaryIndex = new PetscInt[nUx - 2]();
    PetscInt *uNorthGhostBoundaryIndex = new PetscInt[nUx - 2]();
    PetscInt *uEastBoundaryIndex = new PetscInt[nUy - 2]();
    PetscInt *uEastGhostBoundaryIndex = new PetscInt[nUy - 2]();
    PetscInt *uSouthBoundaryIndex = new PetscInt[nUx - 2]();
    PetscInt *uSouthGhostBoundaryIndex = new PetscInt[nUx - 2]();

    PetscInt *vBoundaryIndex = new PetscInt[2 * nVx + 2 * (nVy - 2)]();
    PetscInt *vInteriorIndex = new PetscInt[nVx * nVy - 2 * nVx - 2 * (nVy - 2)]();
    PetscInt *vWestBoundaryIndex = new PetscInt[nVy - 2]();
    PetscInt *vWestGhostBoundaryIndex = new PetscInt[nVy - 2]();
    PetscInt *vNorthBoundaryIndex = new PetscInt[nVx - 2]();
    PetscInt *vNorthGhostBoundaryIndex = new PetscInt[nVx - 2]();
    PetscInt *vEastBoundaryIndex = new PetscInt[nVy - 2]();
    PetscInt *vEastGhostBoundaryIndex = new PetscInt[nVy - 2]();
    PetscInt *vSouthBoundaryIndex = new PetscInt[nVx - 2]();
    PetscInt *vSouthGhostBoundaryIndex = new PetscInt[nVx - 2]();

    PetscInt *pBoundaryIndex = new PetscInt[2 * nPx + 2 * (nPy - 2)]();
    PetscInt *pInteriorIndex = new PetscInt[nPx * nPy - 2 * nPx - 2 * (nPy - 2)]();

    poseidonGetUIndex(nUx, nUy, uBoundaryIndex, uInteriorIndex,
                      uWestBoundaryIndex, uWestGhostBoundaryIndex,
                      uNorthBoundaryIndex, uNorthGhostBoundaryIndex,
                      uEastBoundaryIndex, uEastGhostBoundaryIndex,
                      uSouthBoundaryIndex, uSouthGhostBoundaryIndex);
    poseidonGetVIndex(nVx, nVy, vBoundaryIndex, vInteriorIndex,
                      vWestBoundaryIndex, vWestGhostBoundaryIndex,
                      vNorthBoundaryIndex, vNorthGhostBoundaryIndex,
                      vEastBoundaryIndex, vEastGhostBoundaryIndex,
                      vSouthBoundaryIndex, vSouthGhostBoundaryIndex);
    poseidonGetPIndex(nPx, nPy, pBoundaryIndex, pInteriorIndex);

    /*
     * Initializing variables for u-velocity (U), v-velocity (V), and pressure (P)
     */
//    PetscScalar* U = new PetscScalar[nUx * nUy]; // u-velocity with boundaries
//    PetscScalar* V = new PetscScalar[nVx * nVy]; // v-velocity with boundaries
//    PetscScalar* P = new PetscScalar[nPx * nPy]; // pressure with boundaries
    Vec U, V, P;
    Vec Uhbar, Uvbar, Uhtilde, Uvtilde;
    Vec Vhbar, Vvbar, Vhtilde, Vvtilde;

    // Creating and initializing vectors for holding u and v velocities and pressure
    VecCreate(PETSC_COMM_WORLD, &U); VecSetSizes(U, PETSC_DECIDE, nUx * nUy); VecSetFromOptions(U); VecSet(U, 0);
    VecCreate(PETSC_COMM_WORLD, &V); VecSetSizes(V, PETSC_DECIDE, nVx * nVy); VecSetFromOptions(V); VecSet(V, 0);
    VecCreate(PETSC_COMM_WORLD, &P); VecSetSizes(P, PETSC_DECIDE, nx * ny); VecSetFromOptions(P); VecSet(P, 0);

    /*
     * Assign boundary conditions
     */
    Vec UssBoundary; VecCreate(PETSC_COMM_WORLD, &UssBoundary); VecSetSizes(UssBoundary, PETSC_DECIDE, (nUx - 2) * (nUy - 2)); VecSetFromOptions(UssBoundary); VecSet(UssBoundary, 0);
    Vec VssBoundary; VecCreate(PETSC_COMM_WORLD, &VssBoundary); VecSetSizes(VssBoundary, PETSC_DECIDE, (nVx - 2) * (nVy - 2)); VecSetFromOptions(VssBoundary); VecSet(VssBoundary, 0);
    assignBoundaryCondition(U, nUx, nUy,
                            uSouthBoundaryIndex, uSouthGhostBoundaryIndex,
                            uWestBoundaryIndex, uWestGhostBoundaryIndex,
                            uNorthBoundaryIndex, uNorthGhostBoundaryIndex,
                            uEastBoundaryIndex, uEastGhostBoundaryIndex,
                            uIsNeumannBoundary, uDirichletBoundary,
                            dx, dy,
                            UssBoundary, dt, Re, "U");

    assignBoundaryCondition(V, nVx, nVy,
                            vSouthBoundaryIndex, vSouthGhostBoundaryIndex,
                            vWestBoundaryIndex, vWestGhostBoundaryIndex,
                            vNorthBoundaryIndex, vNorthGhostBoundaryIndex,
                            vEastBoundaryIndex, vEastGhostBoundaryIndex,
                            vIsNeumannBoundary, vDirichletBoundary,
                            dx, dy,
                            VssBoundary, dt, Re, "V");

    /*
     * Calculate gamma
     */
    PetscReal gamma = calculateGamma(dt, U, V);

    /*
     * Generating operators for calculating \bar{ }^{v}, \bar{ }^{h}, \tilde{ }^{h}, and \tilde{ }^{v} operators
     */
    Mat LUhbar; MatCreate(PETSC_COMM_WORLD, &LUhbar); MatSetSizes(LUhbar, PETSC_DECIDE, PETSC_DECIDE, (nUx - 1) * (nUy - 2), nUx * nUy); MatSetFromOptions(LUhbar); MatSetUp(LUhbar);

    Mat LVhbar; MatCreate(PETSC_COMM_WORLD, &LVhbar); MatSetSizes(LVhbar, PETSC_DECIDE, PETSC_DECIDE, (nVx - 1) * nVy, nVx * nVy); MatSetFromOptions(LVhbar); MatSetUp(LVhbar);

    Mat LUhtilde; MatCreate(PETSC_COMM_WORLD, &LUhtilde); MatSetSizes(LUhtilde, PETSC_DECIDE, PETSC_DECIDE, (nUx - 1) * (nUy - 2), nUx * nUy); MatSetFromOptions(LUhtilde); MatSetUp(LUhtilde);

    Mat LVhtilde; MatCreate(PETSC_COMM_WORLD, &LVhtilde); MatSetSizes(LVhtilde, PETSC_DECIDE, PETSC_DECIDE, (nVx - 1) * nVy, nVx * nVy); MatSetFromOptions(LVhtilde); MatSetUp(LVhtilde);

    Mat LUvbar; MatCreate(PETSC_COMM_WORLD, &LUvbar); MatSetSizes(LUvbar, PETSC_DECIDE, PETSC_DECIDE, nUx * (nUy - 1), nUx * nUy); MatSetFromOptions(LUvbar); MatSetUp(LUvbar);

    Mat LVvbar; MatCreate(PETSC_COMM_WORLD, &LVvbar); MatSetSizes(LVvbar, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 1), nVx * nVy); MatSetFromOptions(LVvbar); MatSetUp(LVvbar);

    Mat LUvtilde; MatCreate(PETSC_COMM_WORLD, &LUvtilde); MatSetSizes(LUvtilde, PETSC_DECIDE, PETSC_DECIDE, nUx * (nUy - 1), nUx * nUy); MatSetFromOptions(LUvtilde); MatSetUp(LUvtilde);

    Mat LVvtilde; MatCreate(PETSC_COMM_WORLD, &LVvtilde); MatSetSizes(LVvtilde, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 1), nVx * nVy); MatSetFromOptions(LVvtilde); MatSetUp(LVvtilde);

    PetscInt skipWalls[4]; // This defines to skip north and south walls for U but not for V ==> [S W N E]
    skipWalls[0] = 1; skipWalls[1] = 0; skipWalls[2] = 1; skipWalls[3] = 1;
    Xbar_tilde(nUx, nUy, LUhbar, skipWalls, "bar", "horizontal");
    Xbar_tilde(nUx, nUy, LUhtilde, skipWalls, "tilde", "horizontal");

    skipWalls[0] = 0; skipWalls[1] = 0; skipWalls[2] = 1; skipWalls[3] = 0;
    Xbar_tilde(nUx, nUy, LUvbar, skipWalls, "bar", "vertical");
    Xbar_tilde(nUx, nUy, LUvtilde, skipWalls, "tilde", "vertical");

    skipWalls[0] = 0; skipWalls[1] = 0; skipWalls[2] = 0; skipWalls[3] = 1;
    Xbar_tilde(nVx, nVy, LVhbar, skipWalls, "bar", "horizontal");
    Xbar_tilde(nVx, nVy, LVhtilde, skipWalls, "tilde", "horizontal");

    skipWalls[0] = 0; skipWalls[1] = 1; skipWalls[2] = 1; skipWalls[3] = 1;
    Xbar_tilde(nVx, nVy, LVvbar, skipWalls, "bar", "vertical");
    Xbar_tilde(nVx, nVy, LVvtilde, skipWalls, "tilde", "vertical");

    /*
     * Differential operators definition
     */
    Mat LdUsdX; MatCreate(PETSC_COMM_WORLD, &LdUsdX); MatSetSizes(LdUsdX, PETSC_DECIDE, PETSC_DECIDE, (nUx - 2) * (nUy - 2), (nUx - 1) * (nUy - 2)); MatSetFromOptions(LdUsdX); MatSetUp(LdUsdX);
    Mat LdUsdY; MatCreate(PETSC_COMM_WORLD, &LdUsdY); MatSetSizes(LdUsdY, PETSC_DECIDE, PETSC_DECIDE, nUx * (nUy - 2), nUx * (nUy - 1)); MatSetFromOptions(LdUsdY); MatSetUp(LdUsdY);

    Mat LdVsdX; MatCreate(PETSC_COMM_WORLD, &LdVsdX); MatSetSizes(LdVsdX, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * nVy, (nVx - 1) * nVy); MatSetFromOptions(LdVsdX); MatSetUp(LdVsdX);
    Mat LdVsdY; MatCreate(PETSC_COMM_WORLD, &LdVsdY); MatSetSizes(LdVsdY, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 2), (nVx - 2) * (nVy - 1)); MatSetFromOptions(LdVsdY); MatSetUp(LdVsdY);

    skipWalls[0] = 0; skipWalls[1] = 0; skipWalls[2] = 0; skipWalls[3] = 1;
    diffOperator(nUx - 1, nUy - 2, dx, LdUsdX, skipWalls, "horizontal"); // Operator for Ustar_RHS diff w.r.t x
    skipWalls[0] = 0; skipWalls[1] = 0; skipWalls[2] = 1; skipWalls[3] = 0;
    diffOperator(nUx, nUy - 1, dy, LdUsdY, skipWalls, "vertical"); // Operator for Ustar_RHS diff w.r.t y

    skipWalls[0] = 0; skipWalls[1] = 0; skipWalls[2] = 0; skipWalls[3] = 1;
    diffOperator(nVx - 1, nVy, dx, LdVsdX, skipWalls, "horizontal"); // Operator for Vstar_RHS diff w.r.t x
    skipWalls[0] = 0; skipWalls[1] = 0; skipWalls[2] = 1; skipWalls[3] = 0;
    diffOperator(nVx - 2, nVy - 1, dy, LdVsdY, skipWalls, "vertical"); // Operator for Vstar_RHS diff w.r.t y

    /*
     * Calculating Ustar and Vstar
     */
    Vec Us; VecCreate(PETSC_COMM_WORLD, &Us); VecSetSizes(Us, PETSC_DECIDE, (nUx - 2) * (nUy - 2)); VecSetFromOptions(Us); VecSet(Us, 0);
    Vec Vs; VecCreate(PETSC_COMM_WORLD, &Vs); VecSetSizes(Vs, PETSC_DECIDE, (nVx - 2) * (nVy - 2)); VecSetFromOptions(Vs); VecSet(Vs, 0);
    calculateStarVariable(LUhbar, LUhtilde, LUvbar, LUvtilde, LdUsdX, LdUsdY, nUx, nUy, U, Us, uInteriorIndex,
                          LVhbar, LVhtilde, LVvbar, LVvtilde, LdVsdX, LdVsdY, nVx, nVy, V, Vs, vInteriorIndex,
                          gamma, dt);

    /*
     * Define Laplacian operator
     */
    Mat Lu; MatCreate(PETSC_COMM_WORLD, &Lu); MatSetSizes(Lu, PETSC_DECIDE, PETSC_DECIDE, (nUx - 2) * (nUy - 2), (nUx - 2) * (nUy - 2)); MatSetFromOptions(Lu); MatSetUp(Lu);
    Mat Lv; MatCreate(PETSC_COMM_WORLD, &Lv); MatSetSizes(Lv, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 2), (nVx - 2) * (nVy - 2)); MatSetFromOptions(Lv); MatSetUp(Lv);

    laplacianOperator(Lu, nUx - 2, nUy - 2, dx, dy, uIsNeumannBoundary, "U");
    laplacianOperator(Lv, nVx - 2, nVy - 2, dx, dy, vIsNeumannBoundary, "V");

    Vec Uss; VecCreate(PETSC_COMM_WORLD, &Uss); VecSetSizes(Uss, PETSC_DECIDE, (nUx - 2) * (nUy - 2)); VecSetFromOptions(Uss); VecSet(Uss, 0);
    Vec Vss; VecCreate(PETSC_COMM_WORLD, &Vss); VecSetSizes(Vss, PETSC_DECIDE, (nVx - 2) * (nVy - 2)); VecSetFromOptions(Vss); VecSet(Vss, 0);

    VecAXPY(UssBoundary, 1.0, Us);
    VecAXPY(VssBoundary, 1.0, Vs);

    MatScale(Lu, -dt / Re); MatShift(Lu, 1.0);
    MatScale(Lv, -dt / Re); MatShift(Lv, 1.0);

    KSP uSolver; KSPCreate(PETSC_COMM_WORLD, &uSolver); KSPSetOperators(uSolver, Lu, Lu); KSPSetFromOptions(uSolver);
    KSP vSolver; KSPCreate(PETSC_COMM_WORLD, &vSolver); KSPSetOperators(vSolver, Lv, Lv); KSPSetFromOptions(vSolver);

    /*
     * Solving for imlicit viscosity
     */
    KSPSolve(uSolver, UssBoundary, Uss);
    KSPSolve(vSolver, VssBoundary, Vss);

    /*
     *
     */
    PetscScalar *Uss_ = new PetscScalar[(nUx - 2) * (nUy - 2)]; VecGetArray(Uss, &Uss_);
    PetscScalar *Vss_ = new PetscScalar[(nVx - 2) * (nVy - 2)]; VecGetArray(Vss, &Vss_);

    VecSetValues(U, (nUx - 2) * (nUy - 2), uInteriorIndex, Uss_, INSERT_VALUES);
    VecSetValues(V, (nVx - 2) * (nVy - 2), vInteriorIndex, Vss_, INSERT_VALUES);


    // Define differential operators for calculating the velocity divergence
    Mat LdUdX; MatCreate(PETSC_COMM_WORLD, &LdUdX); MatSetSizes(LdUdX, PETSC_DECIDE, PETSC_DECIDE, (nUx - 1) * (nUy - 2), nUx * nUy); MatSetFromOptions(LdUdX); MatSetUp(LdUdX);
    Mat LdVdY; MatCreate(PETSC_COMM_WORLD, &LdVdY); MatSetSizes(LdVdY, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 1), nVx * nVy); MatSetFromOptions(LdVdY); MatSetUp(LdVdY);

    skipWalls[0] = 1; skipWalls[1] = 0; skipWalls[2] = 1; skipWalls[3] = 1;
    diffOperator(nUx, nUy, dx, LdUdX, skipWalls, "horizontal");
    skipWalls[0] = 0; skipWalls[1] = 1; skipWalls[2] = 1; skipWalls[3] = 1;
    diffOperator(nVx, nVy, dy, LdVdY, skipWalls, "vertical");

    // Define laplacian operator for pressure
    Mat Lp; MatCreate(PETSC_COMM_WORLD, &Lp); MatSetSizes(Lp, PETSC_DECIDE, PETSC_DECIDE, (nPx - 2) * (nPy - 2), (nPx - 2) * (nPy - 2)); MatSetFromOptions(Lp); MatSetUp(Lp);
    laplacianOperator(Lp, nPx - 2, nPy - 2, dx, dy, pIsNeumannBoundary, "P");

    // Solve for pressure
    calculatePressure(P, LdUdX, U, LdVdY, V, Lp, dt);

    /*
     * Corret velocities
     */
    Mat LdPdX; MatCreate(PETSC_COMM_WORLD, &LdPdX); MatSetSizes(LdPdX, PETSC_DECIDE, PETSC_DECIDE, (nx - 1) * ny, nx * ny); MatSetFromOptions(LdPdX); MatSetUp(LdPdX);
    Mat LdPdY; MatCreate(PETSC_COMM_WORLD, &LdPdY); MatSetSizes(LdPdY, PETSC_DECIDE, PETSC_DECIDE, nx * (ny - 1), nx * ny); MatSetFromOptions(LdPdY); MatSetUp(LdPdY);

    skipWalls[0] = 0; skipWalls[1] = 0; skipWalls[2] = 0; skipWalls[3] = 1;
    diffOperator(nx, ny, dx, LdPdX, skipWalls, "horizontal");
    skipWalls[0] = 0; skipWalls[1] = 0; skipWalls[2] = 1; skipWalls[3] = 0;
    diffOperator(nx, ny, dy, LdPdY, skipWalls, "vertical");

    correctVelocity(Uss, Vss, LdPdX, LdPdY, P, dt, uInteriorIndex, vInteriorIndex, U, V);

    /*
     * Destroying vectors
     */
    VecDestroy(&U); VecDestroy(&V); VecDestroy(&P);
//    VecDestroy(&Uhbar); VecDestroy(&Uvbar); VecDestroy(&Uhtilde); VecDestroy(&Uvtilde);
//    VecDestroy(&Vhbar); VecDestroy(&Vvbar); VecDestroy(&Vhtilde); VecDestroy(&Vvtilde);

    MatDestroy(&LUhbar); MatDestroy(&LUvbar); MatDestroy(&LUhtilde); MatDestroy(&LUvtilde);
    MatDestroy(&LVhbar); MatDestroy(&LVvbar); MatDestroy(&LVhtilde); MatDestroy(&LVvtilde);

    MatDestroy(&LdUsdX); MatDestroy(&LdUsdY); MatDestroy(&LdVsdX); MatDestroy(&LdVsdY);

    VecDestroy(&Us); VecDestroy(&Vs);
    MatDestroy(&Lu); MatDestroy(&Lv);
    KSPDestroy(&uSolver); KSPDestroy(&vSolver);
    VecDestroy(&Uss); VecDestroy(&Vss);

    MatDestroy(&LdUdX); MatDestroy(&LdVdY);
    MatDestroy(&Lp);

    PetscFinalize();
    return 0;
}

// Prints the size of vector A to the console
void showVectorSize(Vec A) {
    PetscInt vecSize;
    VecGetSize(A, &vecSize);
    cout << "Vector Length = " << vecSize << endl;
}

// Prints the size of matrix A to the console
void showMatrixSize(Mat A) {
    PetscInt matRow, matColumn;
    MatGetSize(A, &matRow, &matColumn);
    cout << "Row size = " << matRow << "\t Column size = " << matColumn << endl;
}

void writeMat(Mat A) {
    PetscViewer matlabViewer;

    PetscObjectSetName((PetscObject) A, "myVar");
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "operator.output", &matlabViewer);
    PetscViewerSetFormat(matlabViewer, PETSC_VIEWER_ASCII_MATLAB);
    MatView(A, matlabViewer);

    PetscViewerDestroy(&matlabViewer);
}

void writeVec(Vec A) {
    PetscViewer matlabViewer;

    PetscObjectSetName((PetscObject) A, "myVar");
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "operator.output", &matlabViewer);
    PetscViewerSetFormat(matlabViewer, PETSC_VIEWER_ASCII_MATLAB);
    VecView(A, matlabViewer);

    PetscViewerDestroy(&matlabViewer);
}

// Calculate gamma
PetscScalar calculateGamma(PetscScalar dt, Vec &U, Vec &V) {
    PetscInt maxUloc, maxVloc;
    PetscReal maxUval, maxVval;
    Vec Uabs, Vabs;
    VecDuplicate(U, &Uabs);
    VecDuplicate(V, &Vabs);
    VecAbs(Uabs);
    VecAbs(Vabs);
    VecMax(Uabs, &maxUloc, &maxUval);
    VecMax(Vabs, &maxVloc, &maxVval);
    PetscReal gamma = PetscMin(1.2 * dt * PetscMax(maxUval, maxVval), 1); // Define gamma
    VecDestroy(&Uabs);
    VecDestroy(&Vabs);
    return gamma;
}

// Operator for calculating [ ]hbar or [ ]htilde
void Xbar_tilde(PetscInt nx, PetscInt ny, Mat &Lhbar, PetscInt *skipCode, string indicator, string direction) {
    PetscInt barOperatorPosition[2]; // values that go inside the $\bar{ }$ operator
    PetscScalar barOperatorValue[2];
    PetscInt rowIndex = 0;
    if (indicator == "bar") {
        barOperatorValue[0] = 0.5;
        barOperatorValue[1] = 0.5;
    }
    else if (indicator == "tilde") {
        barOperatorValue[0] = -0.5;
        barOperatorValue[1] = 0.5;
    }

    for (int i = 0; i < nx * ny; i++) {
        if ((i < nx) && (skipCode[0] == 1)) {
            // south wall
            continue;
        }
        else if ((fmod(i, nx) == nx - 1) && (skipCode[3] == 1)) {
            // east wall
            continue;
        }
        else if ((i >= nx * (ny - 1)) && (skipCode[2] == 1)) {
            // north wall
            continue;
        }
        else if ((fmod(i, nx) == 0) && (skipCode[1] == 1)) {
            // west wall
            continue;
        }
        else {
            if (direction == "horizontal") {
                barOperatorPosition[0] = i;
                barOperatorPosition[1] = i + 1;
            }
            if (direction == "vertical") {
                barOperatorPosition[0] = i;
                barOperatorPosition[1] = i + nx;
            }
        }
//        cout << rowIndex << "\t\t" << barOperatorPosition[0] << "\t\t" << barOperatorPosition[1] << endl;
        MatSetValues(Lhbar, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        rowIndex++;
    }
    MatAssemblyBegin(Lhbar, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Lhbar, MAT_FINAL_ASSEMBLY);
}

// Operator for differentiating
void diffOperator(PetscInt nx, PetscInt ny, PetscScalar ds, Mat Ldds, PetscInt *skipCode, string direction) {
    PetscInt barOperatorPosition[2]; // values that go inside the $\bar{ }$ operator
    PetscScalar barOperatorValue[2];
    PetscInt rowIndex = 0;

    barOperatorValue[0] = -1.0 / ds;
    barOperatorValue[1] = 1.0 / ds;

    for (int i = 0; i < nx * ny; i++) {
        if ((i < nx) && (skipCode[0] == 1)) {
            // south wall
            continue;
        }
        else if ((fmod(i, nx) == nx - 1) && (skipCode[3] == 1)) {
            // east wall
            continue;
        }
        else if ((i >= nx * (ny - 1)) && (skipCode[2] == 1)) {
            // north wall
            continue;
        }
        else if ((fmod(i, nx) == 0) && (skipCode[1] == 1)) {
            // west wall
            continue;
        }
        else {
            if (direction == "horizontal") {
                barOperatorPosition[0] = i;
                barOperatorPosition[1] = i + 1;
            }
            if (direction == "vertical") {
                barOperatorPosition[0] = i;
                barOperatorPosition[1] = i + nx;
            }
        }
        MatSetValues(Ldds, 1, &rowIndex, 2, barOperatorPosition, barOperatorValue, INSERT_VALUES);
        rowIndex++;
    }
    MatAssemblyBegin(Ldds, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Ldds, MAT_FINAL_ASSEMBLY);
}

void calculateStarVariable(Mat LUhbar, Mat LUhtilde, Mat LUvbar, Mat LUvtilde, Mat LdUsdX, Mat LdUsdY, PetscInt nUx, PetscInt nUy, Vec U, Vec &Us, PetscInt *uInteriorIndex,
                           Mat LVhbar, Mat LVhtilde, Mat LVvbar, Mat LVvtilde, Mat LdVsdX, Mat LdVsdY, PetscInt nVx, PetscInt nVy, Vec V, Vec &Vs, PetscInt *vInteriorIndex,
                           PetscScalar gamma, PetscReal dt) {
    // Generating vectors
    Vec Uhbar; VecCreate(PETSC_COMM_WORLD, &Uhbar); VecSetSizes(Uhbar, PETSC_DECIDE, (nUx - 1) * (nUy - 2)); VecSetFromOptions(Uhbar); VecSet(Uhbar, 0);
    Vec Uvbar; VecCreate(PETSC_COMM_WORLD, &Uvbar); VecSetSizes(Uvbar, PETSC_DECIDE, nUx * (nUy - 1)); VecSetFromOptions(Uvbar); VecSet(Uvbar, 0);
    Vec Vhbar; VecCreate(PETSC_COMM_WORLD, &Vhbar); VecSetSizes(Vhbar, PETSC_DECIDE, (nVx - 1) * nVy); VecSetFromOptions(Vhbar); VecSet(Vhbar, 0);
    Vec Vvbar; VecCreate(PETSC_COMM_WORLD, &Vvbar); VecSetSizes(Vvbar, PETSC_DECIDE, (nVx - 2) * (nVy - 1)); VecSetFromOptions(Vvbar); VecSet(Vvbar, 0);

    Vec Uhtilde, Uvtilde, Vhtilde, Vvtilde;
    VecDuplicate(Uhbar, &Uhtilde);
    VecDuplicate(Uvbar, &Uvtilde);
    VecDuplicate(Vhbar, &Vhtilde);
    VecDuplicate(Vvbar, &Vvtilde);
    // Calculating basic variables required for calculated Ustar_RHS and Vstar_RHS
    MatMult(LUhbar, U, Uhbar);
    MatMult(LUhtilde, U, Uhtilde);
    MatMult(LUvbar, U, Uvbar);
    MatMult(LUvtilde, U, Uvtilde);
    MatMult(LVhbar, V, Vhbar);
    MatMult(LVhtilde, V, Vhtilde);
    MatMult(LVvbar, V, Vvbar);
    MatMult(LVvtilde, V, Vvtilde);

    // Multiplying basic variables
    Vec Uhbar2, UhbarABS, VhbarABS, UvbarABS, Vvbar2, VvbarABS;
    VecDuplicate(Uhbar, &Uhbar2);
    VecDuplicate(Uhbar, &UhbarABS); VecCopy(Uhbar, UhbarABS);
    VecDuplicate(Vhbar, &VhbarABS); VecCopy(Vhbar, VhbarABS);
    VecDuplicate(Uvbar, &UvbarABS); VecCopy(Uvbar, UvbarABS);
    VecDuplicate(Vvbar, &Vvbar2);
    VecDuplicate(Vvbar, &VvbarABS); VecCopy(Vvbar, VvbarABS);
    //
    VecPointwiseMult(Uhbar2, Uhbar, Uhbar);
    VecPointwiseMult(Vvbar2, Vvbar, Vvbar);
    VecAbs(UhbarABS);
    VecAbs(VhbarABS);
    VecAbs(UvbarABS);
    VecAbs(VvbarABS);

    //
    Vec UhbarABS_x_Uhtilde, Uvbar_x_Vhbar, VhbarABS_x_Uvtilde;
    Vec UvbarABS_x_Vhtilde, VvbarABS_x_Vvtilde;

    VecDuplicate(UhbarABS, &UhbarABS_x_Uhtilde);
    VecDuplicate(Uvbar, &Uvbar_x_Vhbar);
    VecDuplicate(VhbarABS, &VhbarABS_x_Uvtilde);
    VecDuplicate(UvbarABS, &UvbarABS_x_Vhtilde);
    VecDuplicate(VvbarABS, &VvbarABS_x_Vvtilde);

    VecPointwiseMult(UhbarABS_x_Uhtilde, UhbarABS, Uhtilde);
    VecPointwiseMult(Uvbar_x_Vhbar, Uvbar, Vhbar);
    VecPointwiseMult(VhbarABS_x_Uvtilde, VhbarABS, Uvtilde);
    VecPointwiseMult(UvbarABS_x_Vhtilde, UvbarABS, Vhtilde);
    VecPointwiseMult(VvbarABS_x_Vvtilde, VvbarABS, Vvtilde);

    //
    Vec Usx, Usy, Vsx, Vsy;
    VecDuplicate(Uhbar, &Usx); VecDuplicate(Uvbar, &Usy);
    VecDuplicate(Uvbar, &Vsx); VecDuplicate(Vvbar, &Vsy);

    VecWAXPY(Usx, -gamma, UhbarABS_x_Uhtilde, Uhbar2);
    VecWAXPY(Usy, -gamma, VhbarABS_x_Uvtilde, Uvbar_x_Vhbar);
    VecWAXPY(Vsx, -gamma, Uvbar_x_Vhbar, Uvbar_x_Vhbar);
    VecWAXPY(Vsy, -gamma, VvbarABS_x_Vvtilde, Vvbar2);

    //
    Vec dUsxdX; VecCreate(PETSC_COMM_WORLD, &dUsxdX); VecSetSizes(dUsxdX, PETSC_DECIDE, (nUx - 2) * (nUy - 2)); VecSetFromOptions(dUsxdX); VecSet(dUsxdX, 0);
    Vec dUsydY; VecCreate(PETSC_COMM_WORLD, &dUsydY); VecSetSizes(dUsydY, PETSC_DECIDE, nUx * (nUy - 2)); VecSetFromOptions(dUsydY); VecSet(dUsydY, 0);
    Vec dVsxdX; VecCreate(PETSC_COMM_WORLD, &dVsxdX); VecSetSizes(dVsxdX, PETSC_DECIDE, (nVx - 2) * nVy); VecSetFromOptions(dVsxdX); VecSet(dVsxdX, 0);
    Vec dVsydY; VecCreate(PETSC_COMM_WORLD, &dVsydY); VecSetSizes(dVsydY, PETSC_DECIDE, (nVx - 2) * (nVy - 2)); VecSetFromOptions(dVsydY); VecSet(dVsydY, 0);

    MatMult(LdUsdX, Usx, dUsxdX);
    MatMult(LdUsdY, Usy, dUsydY);
    MatMult(LdVsdX, Vsx, dVsxdX);
    MatMult(LdVsdY, Vsy, dVsydY);
    // Since the dimension of dUsxdX and dUsydY and dVsxdX and dVsydY do not match, we need to extract the data of these vectors
    PetscInt *dUsydYInteriorNodeIndex = new PetscInt[(nUx - 2) * (nUy - 2)];
    PetscInt index = 0;
    for (int i = 0; i < nUx * (nUy - 2); i++) {
        if (fmod(i, nUx) == 0) {
            continue;
        }
        if (fmod(i, nUx) == nUx - 1) {
            continue;
        }
        dUsydYInteriorNodeIndex[index] = i;
        index++;
    }
    //
    PetscInt *dVsxdXInteriorNodeIndex = new PetscInt[(nVx - 2) * (nVy - 2)];
    index = 0;
    for (int i = 0; i < (nVx - 2) * nVy; i++) {
        if (i < nVx - 2) {
            continue;
        }
        if (i >= (nVx - 2) * (nVy - 1)) {
            continue;
        }
        dVsxdXInteriorNodeIndex[index] = i;
        index++;
    }
    vecExtract(dUsydY, dUsydYInteriorNodeIndex, (nUx - 2) * (nUy - 2));
    vecExtract(dVsxdX, dVsxdXInteriorNodeIndex, (nVx - 2) * (nVy - 2));

    // Calculating the right-hand-side of Ustar and Vstar equations
    Vec UsRHS, VsRHS;
    VecDuplicate(dUsxdX, &UsRHS); VecDuplicate(dVsxdX, &VsRHS);
    VecWAXPY(UsRHS, 1, dUsxdX, dUsydY); VecScale(UsRHS, -1.0);
    VecWAXPY(VsRHS, 1, dVsxdX, dVsydY); VecScale(VsRHS, -1.0);

    // Calculating Ustar and Vstar
    vecExtract(U, uInteriorIndex, (nUx - 2) * (nUy - 2));
    vecExtract(V, vInteriorIndex, (nVx - 2) * (nVy - 2));

    VecWAXPY(Us, dt, UsRHS, U);
    VecWAXPY(Vs, dt, VsRHS, V);

    // Destroying vectors
    VecDestroy(&Uhbar); VecDestroy(&Uvbar); VecDestroy(&Vhbar); VecDestroy(&Vvbar);
    VecDestroy(&Uhtilde); VecDestroy(&Uvtilde); VecDestroy(&Vhtilde); VecDestroy(&Vvtilde);

    VecDestroy(&Uhbar2); VecDestroy(&UhbarABS); VecDestroy(&VhbarABS);
    VecDestroy(&UvbarABS); VecDestroy(&Vvbar2); VecDestroy(&VvbarABS);

    VecDestroy(&UhbarABS_x_Uhtilde); VecDestroy(&Uvbar_x_Vhbar); VecDestroy(&VhbarABS_x_Uvtilde);
    VecDestroy(&UvbarABS_x_Vhtilde); VecDestroy(&VvbarABS_x_Vvtilde);

    VecDestroy(&dUsxdX); VecDestroy(&dUsydY); VecDestroy(&dVsxdX); VecDestroy(&dVsydY);

    VecDestroy(&UsRHS); VecDestroy(&VsRHS);
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

void laplacianOperator(Mat &L, PetscInt nx, PetscInt ny, PetscScalar dx, PetscScalar dy, PetscInt *neumannBoundary, string variable) {
    PetscScalar rowValue[5] = {};
    PetscInt rowLocation[5] = {};
    PetscScalar dx2 = pow(dx, 2);
    PetscScalar dy2 = pow(dy, 2);
    for (int i = 0; i < nx * ny; i++) {
        if (i == 0) {
            // South-west corner
            rowValue[0] = - 2 / dx2 - 2 / dy2; rowValue[1] = 1 / dx2; rowValue[2] = 1 / dy2;
            rowLocation[0] = i; rowLocation[1] = i + 1; rowLocation[2] = i + nx;
            // Applying neumann boundary condition
            rowValue[0] = rowValue[0] + neumannBoundary[1] * 1 / dx2 + neumannBoundary[0] * 1 / dy2;
            // Fix for dirichlet boundary
            if (neumannBoundary[0] == 0) {
                if (variable == "U")
                    rowValue[0] = rowValue[0] - 1 / dy2;
                if (variable == "P")
                    rowValue[0] = rowValue[0] - 1 / dy2;
            }
            if (neumannBoundary[1] == 0) {
                if (variable == "V")
                    rowValue[0] = rowValue[0] - 1 / dx2;
                if (variable == "P")
                    rowValue[0] = rowValue[0] - 1 / dx2;
            }

            if (variable == "P")
                rowValue[0] = 1.5 * rowValue[0];

            MatSetValues(L, 1, &i, 3, rowLocation, rowValue, INSERT_VALUES);
            continue;
        }
        if (i < nx - 1) {
            // South-wall
            rowValue[0] = 1 / dx2; rowValue[1] = -2 / dx2 - 2 / dy2; rowValue[2] = 1 / dx2; rowValue[3] = 1 / dy2;
            rowLocation[0] = i - 1; rowLocation[1] = i; rowLocation[2] = i + 1; rowLocation[3] = i + nx;
            // Applying neumann boundary condition
            rowValue[1] = rowValue[1] + neumannBoundary[0] * 1 / dy2;
            if (neumannBoundary[0] == 0) {
                if (variable == "U")
                    rowValue[1] = rowValue[1] - 1 / dy2;
                if (variable == "P")
                    rowValue[1] = rowValue[1] - 1 / dy2;
            }

            MatSetValues(L, 1, &i, 4, rowLocation, rowValue, INSERT_VALUES);
            continue;
        }
        if (i == (nx - 1)) {
            // South-east corner
            rowValue[0] = 1 / dx2; rowValue[1] = - 2 / dx2 - 2 / dy2; rowValue[2] = 1 / dy2;
            rowLocation[0] = i - 1; rowLocation[1] = i; rowLocation[2] = i + nx;
            // Applying neumann boundary condition
            rowValue[1] = rowValue[1] + neumannBoundary[3] * 1 / dx2 + neumannBoundary[0] * 1 / dy2;

            if (neumannBoundary[0] == 0) {
                if (variable == "U")
                    rowValue[1] = rowValue[1] - 1 / dy2;
                if (variable == "P")
                    rowValue[1] = rowValue[1] - 1 / dy2;
            }
            if (neumannBoundary[3] == 0) {
                if (variable == "V")
                    rowValue[1] = rowValue[1] - 1 / dx2;
                if (variable == "P")
                    rowValue[1] = rowValue[1] - 1 / dx2;
            }

            MatSetValues(L, 1, &i, 3, rowLocation, rowValue, INSERT_VALUES);
            continue;
        }
        if ((fmod(i, nx) == 0) && (i < nx * (ny - 1))) {
            // west wall
            rowValue[0] = 1 / dy2; rowValue[1] = - 2 / dx2 - 2 / dy2; rowValue[2] = 1 / dx2; rowValue[3] = 1 / dy2;
            rowLocation[0] = i - nx; rowLocation[1] = i; rowLocation[2] = i + 1; rowLocation[3] = i + nx;
            // Applying neumann boundary condition
            rowValue[1] = rowValue[1] + neumannBoundary[1] * 1 / dx2;

            if (neumannBoundary[1] == 0) {
                if (variable == "V")
                    rowValue[1] = rowValue[1] - 1 / dx2;
                if (variable == "P")
                    rowValue[1] = rowValue[1] - 1 / dx2;
            }

            MatSetValues(L, 1, &i, 4, rowLocation, rowValue, INSERT_VALUES);
            continue;
        }
        if ((fmod(i, nx) == nx - 1) && (i < nx * (ny - 1))) {
            // East wall
            rowValue[0] = 1 / dy2; rowValue[1] = 1 / dx2; rowValue[2] = - 2 / dx2 - 2 / dy2; rowValue[3] = 1 / dy2;
            rowLocation[0] = i - nx; rowLocation[1] = i - 1; rowLocation[2] = i; rowLocation[3] = i + nx;
            // Applying neumann boundary condition
            rowValue[2] = rowValue[2] + neumannBoundary[3] * 1 / dx2;

            if (neumannBoundary[3] == 0) {
                if (variable == "V")
                    rowValue[2] = rowValue[2] - 1 / dx2;
                if (variable == "P")
                    rowValue[2] = rowValue[2] - 1 / dx2;
            }

            MatSetValues(L, 1, &i, 4, rowLocation, rowValue, INSERT_VALUES);
            continue;
        }
        if (fmod(i, nx) == 0) {
            // north-west corner
            rowValue[0] = 1 / dy2; rowValue[1] = - 2 / dx2 - 2 / dy2; rowValue[2] = 1 / dx2;
            rowLocation[0] = i - nx; rowLocation[1] = i; rowLocation[2] = i + 1;
            // Applying neumann boundary condition
            rowValue[1] = rowValue[1] + neumannBoundary[1] * 1 / dx2 + neumannBoundary[2] * 1 / dy2;

            if (neumannBoundary[2] == 0) {
                if (variable == "U")
                    rowValue[1] = rowValue[1] - 1 / dy2;
                if (variable == "P")
                    rowValue[1] = rowValue[1] - 1 / dy2;
            }
            if (neumannBoundary[1] == 0) {
                if (variable == "V")
                    rowValue[1] = rowValue[1] - 1 / dx2;
                if (variable == "P")
                    rowValue[1] = rowValue[1] - 1 / dx2;
            }

            MatSetValues(L, 1, &i, 3, rowLocation, rowValue, INSERT_VALUES);
            continue;
        }
        if ((i > nx * (ny - 1)) && (i < nx * ny - 1)) {
            // North-wall
            rowValue[0] = 1 / dy2; rowValue[1] = 1 / dx2; rowValue[2] = - 2 / dx2 - 2 / dy2; rowValue[3] = 1 / dx2;
            rowLocation[0] = i - nx; rowLocation[1] = i - 1; rowLocation[2] = i; rowLocation[3] = i + 1;
            // Applying neumann boundary condition
            rowValue[2] = rowValue[2] + neumannBoundary[2] * 1 / dy2;

            if (neumannBoundary[2] == 0) {
                if (variable == "U")
                    rowValue[2] = rowValue[2] - 1 / dy2;
                if (variable == "P")
                    rowValue[2] = rowValue[2] - 1 / dy2;
            }

            MatSetValues(L, 1, &i, 4, rowLocation, rowValue, INSERT_VALUES);
            continue;
        }
        if (i == nx * ny - 1) {
            // north-east corner
            rowValue[0] = 1 / dy2; rowValue[1] = 1 / dx2; rowValue[2] = - 2 / dx2 - 2 / dy2;
            rowLocation[0] = i - nx; rowLocation[1] = i - 1; rowLocation[2] = i;
            // Applying neumann boundary condition
            rowValue[2] = rowValue[2] + neumannBoundary[3] * 1 / dx2 + neumannBoundary[2] * 1 / dy2;

            if (neumannBoundary[2] == 0) {
                if (variable == "U")
                    rowValue[2] = rowValue[2] - 1 / dy2;
                if (variable == "P")
                    rowValue[2] = rowValue[2] - 1 / dy2;
            }
            if (neumannBoundary[3] == 0) {
                if (variable == "V")
                    rowValue[2] = rowValue[2] - 1 / dx2;
                if (variable == "P")
                    rowValue[2] = rowValue[2] - 1 / dx2;
            }

            MatSetValues(L, 1, &i, 3, rowLocation, rowValue, INSERT_VALUES);
            continue;
        }
        rowValue[0] = 1 / dy2; rowValue[1] = 1 / dx2; rowValue[2] = - 2 / dx2 - 2 / dy2; rowValue[3] = 1 / dx2; rowValue[4] = 1 / dy2;
        rowLocation[0] = i - nx; rowLocation[1] = i - 1; rowLocation[2] = i; rowLocation[3] = i + 1; rowLocation[4] = i + nx;
        MatSetValues(L, 1, &i, 5, rowLocation, rowValue, INSERT_VALUES);
    }
    MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY);
//    MatView(L, PETSC_VIEWER_STDOUT_WORLD);
//    PetscViewer matlabViewer;
//    PetscObjectSetName((PetscObject) L, "myMat");
//    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "operator.output", &matlabViewer);
//    PetscViewerSetFormat(matlabViewer, PETSC_VIEWER_ASCII_MATLAB);
//    MatView(L, matlabViewer);
}

void assignBoundaryCondition(Vec &X, PetscInt nx, PetscInt ny,
                             PetscInt *southBoundaryIndex, PetscInt *southBoundaryGhostIndex,
                             PetscInt *westBoundaryIndex, PetscInt *westBoundaryGhostIndex,
                             PetscInt *northBoundaryIndex, PetscInt *northBoundaryGhostIndex,
                             PetscInt *eastBoundaryIndex, PetscInt *eastBoundaryGhostIndex,
                             PetscInt *isNeumannBoundary, PetscScalar *dirichletBoundary,
                             PetscScalar dx, PetscScalar dy,
                             Vec &XssBoundary, PetscReal dt, PetscReal Re, string variableType) {
    /*
     * South boundary
     */
    PetscScalar *southBoundaryValue = new PetscScalar[nx - 2]();
    if (isNeumannBoundary[0] == 1) {
//        cout << "Applying neumann boundary to south wall for " << variableType << endl;
        VecGetValues(X, nx - 2, southBoundaryGhostIndex, southBoundaryValue);
    }
    else if (isNeumannBoundary[0] == 0) {
//        cout << "Applying dirichelt boundary to south wall for " << variableType << endl;
        VecGetValues(X, nx - 2, southBoundaryGhostIndex, southBoundaryValue);
        if (variableType == "U") {
            for (int i = 0; i < nx - 2; i++) {
                southBoundaryValue[i] = 2 * dirichletBoundary[0] - southBoundaryValue[i];
                VecSetValue(XssBoundary, southBoundaryIndex[i] - 1, dt / Re * southBoundaryValue[i] / pow(dy, 2.0), INSERT_VALUES); // for Xss boundary
            }
        }
        else if (variableType == "V") {
            std::fill_n(southBoundaryValue, nx - 2, dirichletBoundary[0]);
            for (int i = 0; i < nx - 2; i++) {
                VecSetValue(XssBoundary, southBoundaryIndex[i] - 1, dt / Re * southBoundaryValue[i] / pow(dy, 2.0), INSERT_VALUES); // for Xss boundary
            }
        }
    }
    VecSetValues(X, nx - 2, southBoundaryIndex, southBoundaryValue, INSERT_VALUES);

    /*
     * West boundary
     */
    PetscScalar *westBoundaryValue = new PetscScalar[ny - 2]();
    if (isNeumannBoundary[1] == 1) {
//        cout << "Applying neumann boundary to west wall for " << variableType << endl;
        VecGetValues(X, ny - 2, westBoundaryGhostIndex, westBoundaryValue);
    }
    else if (isNeumannBoundary[1] == 0) {
//        cout << "Applying dirichelt boundary to west wall for " << variableType << endl;
        VecGetValues(X, ny - 2, westBoundaryGhostIndex, westBoundaryValue);
        if (variableType == "V") {
            for (int i = 0; i < ny - 2; i++) {
                westBoundaryValue[i] = 2 * dirichletBoundary[1] - westBoundaryValue[i];
                VecSetValue(XssBoundary, i * (nx - 2), dt / Re * westBoundaryValue[i] / pow(dx, 2), INSERT_VALUES); // for Xss boundary
            }
        }
        else if (variableType == "U") {
            std::fill_n(westBoundaryValue, ny - 2, dirichletBoundary[1]);
            for (int i = 0; i < ny - 2; i++) {
                VecSetValue(XssBoundary, i * (nx - 2), dt / Re * westBoundaryValue[i] / pow(dx, 2), INSERT_VALUES); // for Xss boundary
            }
        }
    }
    VecSetValues(X, ny - 2, westBoundaryIndex, westBoundaryValue, INSERT_VALUES);

    /*
     * North boundary
     */
    PetscScalar *northBoundaryValue = new PetscScalar[nx - 2]();
    if (isNeumannBoundary[2] == 1) {
//        cout << "Applying neumann boundary to north wall for " << variableType << endl;
        VecGetValues(X, nx - 2, northBoundaryGhostIndex, northBoundaryValue);
    }
    else if (isNeumannBoundary[2] == 0) {
//        cout << "Applying dirichelt boundary to north wall for " << variableType << endl;
        VecGetValues(X, nx - 2, northBoundaryGhostIndex, northBoundaryValue);
        if (variableType == "U") {
            for (int i = 0; i < nx - 2; i++) {
                northBoundaryValue[i] = 2 * dirichletBoundary[2] - northBoundaryValue[i];
                VecSetValue(XssBoundary, i + (nx - 2) * (ny - 2 - 1), dt / Re * northBoundaryValue[i] / pow(dy, 2), INSERT_VALUES); // for Xss boundary
            }
        }
        else if (variableType == "V") {
            std::fill_n(northBoundaryValue, nx - 2, dirichletBoundary[2]);
            for (int i = 0; i < nx - 2; i++) {
                VecSetValue(XssBoundary, i + (nx - 2) * (ny - 2 - 1), dt / Re * northBoundaryValue[i] / pow(dy, 2), INSERT_VALUES); // for Xss boundary
            }
        }
    }
    VecSetValues(X, nx - 2, northBoundaryIndex, northBoundaryValue, INSERT_VALUES);

    /*
     * East boundary
     */
    PetscScalar *eastBoundaryValue = new PetscScalar[ny - 2]();
    if (isNeumannBoundary[3] == 1) {
//        cout << "Applying neumann boundary to east wall for " << variableType << endl;
        VecGetValues(X, ny - 2, eastBoundaryGhostIndex, eastBoundaryValue);
    }
    else if (isNeumannBoundary[3] == 0) {
//        cout << "Applying dirichelt boundary to east wall for " << variableType << endl;
        VecGetValues(X, ny - 2, eastBoundaryGhostIndex, eastBoundaryValue);
        if (variableType == "V") {
            for (int i = 0; i < ny - 2; i++) {
                eastBoundaryValue[i] = 2 * dirichletBoundary[3] - eastBoundaryValue[i];
                VecSetValue(XssBoundary, i * (nx - 2) + (nx - 2) - 1, dt / Re * eastBoundaryValue[i] / pow(dx, 2), INSERT_VALUES); // for Xss boundary
            }
        }
        else if (variableType == "U") {
            std::fill_n(eastBoundaryValue, ny - 2, dirichletBoundary[3]);
            for (int i = 0; i < ny - 2; i++) {
                VecSetValue(XssBoundary, i * (nx - 2) + (nx - 2) - 1, dt / Re * eastBoundaryValue[i] / pow(dx, 2), INSERT_VALUES); // for Xss boundary
            }
        }
    }
    VecSetValues(X, ny - 2, eastBoundaryIndex, eastBoundaryValue, INSERT_VALUES);

    /*
     * Assembling vectors
     */
    VecAssemblyBegin(X); VecAssemblyEnd(X);
    VecAssemblyBegin(XssBoundary); VecAssemblyEnd(XssBoundary);
}

void calculatePressure(Vec &P, Mat LdUdX, Vec U, Mat LdVdY, Vec V, Mat Lp, PetscReal dt) {
    PetscInt nU, nV, temp;
    MatGetSize(LdUdX, &nU, &temp);
    MatGetSize(LdVdY, &nV, &temp);

    // Define variables to hold the differentiated velocity and pressure fields
    Vec dUdX; VecCreate(PETSC_COMM_WORLD, &dUdX); VecSetSizes(dUdX, PETSC_DECIDE, nU); VecSetFromOptions(dUdX); VecSet(dUdX, 0);
    Vec dVdY; VecCreate(PETSC_COMM_WORLD, &dVdY); VecSetSizes(dVdY, PETSC_DECIDE, nV); VecSetFromOptions(dVdY); VecSet(dVdY, 0);

    MatMult(LdUdX, U, dUdX);
    MatMult(LdVdY, V, dVdY);

    // Creating the right-hand-side of the pressure correction
    Vec RHS; VecDuplicate(dUdX, &RHS);
    VecWAXPY(RHS, 1.0, dUdX, dVdY);
    VecScale(RHS, 1 / dt);

    // solving for pressure term
    KSP pSolver; KSPCreate(PETSC_COMM_WORLD, &pSolver); KSPSetOperators(pSolver, Lp, Lp); KSPSetFromOptions(pSolver);
    KSPSolve(pSolver, RHS, P);

    VecDestroy(&dUdX); VecDestroy(&dVdY);
    VecDestroy(&RHS);
    KSPDestroy(&pSolver);
}

void correctVelocity(Vec Uss, Vec Vss, Mat LdPdX, Mat LdPdY, Vec P, PetscReal dt,
                     PetscInt *uInteriorIndex, PetscInt *vInteriorIndex, Vec &U, Vec &V) {
    Vec dPdX, dPdY;
    VecDuplicate(Uss, &dPdX);
    VecDuplicate(Vss, &dPdY);

    MatMult(LdPdX, P, dPdX);
    MatMult(LdPdY, P, dPdY);

    VecScale(dPdX, -dt);
    VecScale(dPdY, -dt);

    Vec Urhs, Vrhs;
    VecDuplicate(Uss, &Urhs);
    VecDuplicate(Vss, &Vrhs);

    VecWAXPY(Urhs, 1.0, Uss, dPdX);
    VecWAXPY(Vrhs, 1.0, Vss, dPdY);

    PetscInt nUss, nVss;
    VecGetSize(Uss, &nUss);
    VecGetSize(Vss, &nVss);
    PetscScalar *Urhs_ = new PetscScalar[nUss];
    PetscScalar *Vrhs_ = new PetscScalar[nVss];

    VecGetArray(Urhs, &Urhs_);
    VecGetArray(Vrhs, &Vrhs_);

    VecSetValues(U, nUss, uInteriorIndex, Urhs_, INSERT_VALUES);
    VecSetValues(V, nVss, vInteriorIndex, Vrhs_, INSERT_VALUES);

    VecDestroy(&dPdX); VecDestroy(&dPdY);
    VecDestroy(&Urhs); VecDestroy(&Vrhs);
}
//    /*
//     * Track the boundary and interior indices for different variables
//     */
//    PetscInt *uBoundaryIndex = new PetscInt[2 * nUx + 2 * (nUy - 2)]();
//    PetscInt *uInteriorIndex = new PetscInt[nUx * nUy - 2 * nUx - 2 * (nUy - 2)]();
//    PetscInt *uWestBoundaryIndex = new PetscInt[nUy - 2]();
//    PetscInt *uWestGhostBoundaryIndex = new PetscInt[nUy - 2]();
//    PetscInt *uNorthBoundaryIndex = new PetscInt[nUx - 2]();
//    PetscInt *uNorthGhostBoundaryIndex = new PetscInt[nUx - 2]();
//    PetscInt *uEastBoundaryIndex = new PetscInt[nUy - 2]();
//    PetscInt *uEastGhostBoundaryIndex = new PetscInt[nUy - 2]();
//    PetscInt *uSouthBoundaryIndex = new PetscInt[nUx - 2]();
//    PetscInt *uSouthGhostBoundaryIndex = new PetscInt[nUx - 2]();
//
//    PetscInt *vBoundaryIndex = new PetscInt[2 * nVx + 2 * (nVy - 2)]();
//    PetscInt *vInteriorIndex = new PetscInt[nVx * nVy - 2 * nVx - 2 * (nVy - 2)]();
//    PetscInt *vWestBoundaryIndex = new PetscInt[nVy - 2]();
//    PetscInt *vWestGhostBoundaryIndex = new PetscInt[nVy - 2]();
//    PetscInt *vNorthBoundaryIndex = new PetscInt[nVx - 2]();
//    PetscInt *vNorthGhostBoundaryIndex = new PetscInt[nVx - 2]();
//    PetscInt *vEastBoundaryIndex = new PetscInt[nVy - 2]();
//    PetscInt *vEastGhostBoundaryIndex = new PetscInt[nVy - 2]();
//    PetscInt *vSouthBoundaryIndex = new PetscInt[nVx - 2]();
//    PetscInt *vSouthGhostBoundaryIndex = new PetscInt[nVx - 2]();
//
//    PetscInt *pBoundaryIndex = new PetscInt[2 * nPx + 2 * (nPy - 2)]();
//    PetscInt *pInteriorIndex = new PetscInt[nPx * nPy - 2 * nPx - 2 * (nPy - 2)]();
//
//    poseidonGetUIndex(nUx, nUy, uBoundaryIndex, uInteriorIndex,
//                      uWestBoundaryIndex, uWestGhostBoundaryIndex,
//                      uNorthBoundaryIndex, uNorthGhostBoundaryIndex,
//                      uEastBoundaryIndex, uEastGhostBoundaryIndex,
//                      uSouthBoundaryIndex, uSouthGhostBoundaryIndex);
//    poseidonGetVIndex(nVx, nVy, vBoundaryIndex, vInteriorIndex,
//                      vWestBoundaryIndex, vWestGhostBoundaryIndex,
//                      vNorthBoundaryIndex, vNorthGhostBoundaryIndex,
//                      vEastBoundaryIndex, vEastGhostBoundaryIndex,
//                      vSouthBoundaryIndex, vSouthGhostBoundaryIndex);
//    poseidonGetPIndex(nPx, nPy, pBoundaryIndex, pInteriorIndex);
//
////    PetscIntView(2 * (nx + 1) + 2 * ny, uBoundaryIndex, PETSC_VIEWER_STDOUT_WORLD);
////    PetscIntView((nx + 1) * (ny + 2) - 2 * (nx + 1) - 2 * ny, uInteriorIndex, PETSC_VIEWER_STDOUT_WORLD);
////    PetscIntView(2 * nVx + 2 * (nVy - 2), vBoundaryIndex, PETSC_VIEWER_STDOUT_WORLD);
////    PetscIntView(nVx * nVy - 2 * nVx - 2 * (nVy - 2), vInteriorIndex, PETSC_VIEWER_STDOUT_WORLD);
////    PetscIntView(2 * nPx + 2 * (nPy - 2), pBoundaryIndex, PETSC_VIEWER_STDOUT_WORLD);
////    PetscIntView(nPx * nPy - 2 * nPx - 2 * (nPy - 2), pInteriorIndex, PETSC_VIEWER_STDOUT_WORLD);
//
//    /*
//     * Gamma calculation for convective terms
//     */
//    PetscInt maxUloc, maxVloc;
//    PetscReal maxUval, maxVval;
//    Vec Uabs, Vabs;
//    VecDuplicate(U, &Uabs);
//    VecDuplicate(V, &Vabs);
//    VecAbs(Uabs);
//    VecAbs(Vabs);
//    VecMax(Uabs, &maxUloc, &maxUval);
//    VecMax(Vabs, &maxVloc, &maxVval);
//    PetscReal gamma = PetscMin(1.2 * dt * PetscMax(maxUval, maxVval), 1); // Define gamma
//    VecDestroy(&Uabs);
//    VecDestroy(&Vabs);
//
//    /*
//     * Defining boundary conditions
//     */
//    // Generating vectors
//    Vec uWestBoundaryValue, uNorthBoundaryValue, uEastBoundaryValue, uSouthBoundaryValue;
//    Vec vWestBoundaryValue, vNorthBoundaryValue, vEastBoundaryValue, vSouthBoundaryValue;
//
//    VecCreate(PETSC_COMM_WORLD, &uWestBoundaryValue); VecSetSizes(uWestBoundaryValue, PETSC_DECIDE, nUy - 2); VecSetFromOptions(uWestBoundaryValue);
//    VecCreate(PETSC_COMM_WORLD, &uNorthBoundaryValue); VecSetSizes(uNorthBoundaryValue, PETSC_DECIDE, nUx - 2); VecSetFromOptions(uNorthBoundaryValue);
//    VecCreate(PETSC_COMM_WORLD, &uEastBoundaryValue); VecSetSizes(uEastBoundaryValue, PETSC_DECIDE, nUy - 2); VecSetFromOptions(uEastBoundaryValue);
//    VecCreate(PETSC_COMM_WORLD, &uSouthBoundaryValue); VecSetSizes(uSouthBoundaryValue, PETSC_DECIDE, nUx - 2); VecSetFromOptions(uSouthBoundaryValue);
//
//    VecCreate(PETSC_COMM_WORLD, &vWestBoundaryValue); VecSetSizes(vWestBoundaryValue, PETSC_DECIDE, nVy); VecSetFromOptions(vWestBoundaryValue);
//    VecCreate(PETSC_COMM_WORLD, &vNorthBoundaryValue); VecSetSizes(vNorthBoundaryValue, PETSC_DECIDE, nVx); VecSetFromOptions(vNorthBoundaryValue);
//    VecCreate(PETSC_COMM_WORLD, &vEastBoundaryValue); VecSetSizes(vEastBoundaryValue, PETSC_DECIDE, nVy); VecSetFromOptions(vEastBoundaryValue);
//    VecCreate(PETSC_COMM_WORLD, &vSouthBoundaryValue); VecSetSizes(vSouthBoundaryValue, PETSC_DECIDE, nVx); VecSetFromOptions(vSouthBoundaryValue);
//
//    // Defining boundary conditions values
//    VecSet(uWestBoundaryValue, 0.0); VecSet(uNorthBoundaryValue, 1.0); VecSet(uEastBoundaryValue, 0.0); VecSet(uSouthBoundaryValue, 0.0);
//    VecSet(vWestBoundaryValue, 0.0); VecSet(vNorthBoundaryValue, 0.0); VecSet(vEastBoundaryValue, 0.0); VecSet(vSouthBoundaryValue, 0.0);
//
//    PetscScalar *uWestBoundaryValue_array; VecGetArray(uWestBoundaryValue, &uWestBoundaryValue_array);
//    PetscScalar *vWestBoundaryValue_array; VecGetArray(vWestBoundaryValue, &vWestBoundaryValue_array);
//    PetscScalar *uNorthBoundaryValue_array; VecGetArray(uNorthBoundaryValue, &uNorthBoundaryValue_array);
//    PetscScalar *vNorthBoundaryValue_array; VecGetArray(vNorthBoundaryValue, &vNorthBoundaryValue_array);
//    PetscScalar *uEastBoundaryValue_array; VecGetArray(uEastBoundaryValue, &uEastBoundaryValue_array);
//    PetscScalar *vEastBoundaryValue_array; VecGetArray(vEastBoundaryValue, &vEastBoundaryValue_array);
//    PetscScalar *uSouthBoundaryValue_array; VecGetArray(uSouthBoundaryValue, &uSouthBoundaryValue_array);
//    PetscScalar *vSouthBoundaryValue_array; VecGetArray(vSouthBoundaryValue, &vSouthBoundaryValue_array);
//
//    // Assigning boundary condition values
//    VecSetValues(U, nUy - 2, uWestBoundaryIndex, uWestBoundaryValue_array, INSERT_VALUES);
//    VecSetValues(V, nVy - 2, vWestBoundaryIndex, vWestBoundaryValue_array, INSERT_VALUES);
//    VecSetValues(U, nUx - 2, uNorthBoundaryIndex, uNorthBoundaryValue_array, INSERT_VALUES);
//    VecSetValues(V, nVx - 2, vNorthBoundaryIndex, vNorthBoundaryValue_array, INSERT_VALUES);
//    VecSetValues(U, nUy - 2, uEastBoundaryIndex, uEastBoundaryValue_array, INSERT_VALUES);
//    VecSetValues(V, nVy - 2, vEastBoundaryIndex, vEastBoundaryValue_array, INSERT_VALUES);
//    VecSetValues(U, nUx - 2, uSouthBoundaryIndex, uSouthBoundaryValue_array, INSERT_VALUES);
//    VecSetValues(V, nVx - 2, vSouthBoundaryIndex, vSouthBoundaryValue_array, INSERT_VALUES);
//
//    /*
//     * Calcualte Uhbar, Uhtitlde, ... operators for discretising the convective terms.
//     */
//    //Defining variables
//    VecCreate(PETSC_COMM_WORLD, &Uhbar);
//    VecSetSizes(Uhbar, PETSC_DECIDE, (nUx - 1) * (nUy - 2));
//    VecSetFromOptions(Uhbar);  // fixed
//    VecCreate(PETSC_COMM_WORLD, &Uvbar);
//    VecSetSizes(Uvbar, PETSC_DECIDE, nUx * (nUy - 1));
//    VecSetFromOptions(Uvbar); // fixed
//    VecCreate(PETSC_COMM_WORLD, &Uhtilde);
//    VecSetSizes(Uhtilde, PETSC_DECIDE, (nUx - 1) * (nUy - 2));
//    VecSetFromOptions(Uhtilde);  // fixed
//    VecCreate(PETSC_COMM_WORLD, &Uvtilde);
//    VecSetSizes(Uvtilde, PETSC_DECIDE, nUx * (nUy - 1));
//    VecSetFromOptions(Uvtilde); // fixed
//
//    VecCreate(PETSC_COMM_WORLD, &Vhbar);
//    VecSetSizes(Vhbar, PETSC_DECIDE, (nVx - 1) * nVy);
//    VecSetFromOptions(Vhbar); // fixed
//    VecCreate(PETSC_COMM_WORLD, &Vvbar);
//    VecSetSizes(Vvbar, PETSC_DECIDE, (nVx - 2) * (nVy - 1));
//    VecSetFromOptions(Vvbar);  // fixed
//    VecCreate(PETSC_COMM_WORLD, &Vhtilde);
//    VecSetSizes(Vhtilde, PETSC_DECIDE, (nVx - 1) * nVy);
//    VecSetFromOptions(Vhtilde);  // fixed
//    VecCreate(PETSC_COMM_WORLD, &Vvtilde);
//    VecSetSizes(Vvtilde, PETSC_DECIDE, (nVx - 2) * (nVy - 1));
//    VecSetFromOptions(Vvtilde);  // fixed
//    // Initializing matrices for operators
//    Mat UhbarOperator;
//    Mat UhtildeOperator;
//    Mat UvbarOperator;
//    Mat UvtildeOperator;
//    Mat VhbarOperator;
//    Mat VhtildeOperator;
//    Mat VvbarOperator;
//    Mat VvtildeOperator;
//
//    MatCreate(PETSC_COMM_WORLD, &UhbarOperator);
//    MatCreate(PETSC_COMM_WORLD, &UhtildeOperator);
//    MatCreate(PETSC_COMM_WORLD, &UvbarOperator);
//    MatCreate(PETSC_COMM_WORLD, &UvtildeOperator);
//    MatCreate(PETSC_COMM_WORLD, &VhbarOperator);
//    MatCreate(PETSC_COMM_WORLD, &VhtildeOperator);
//    MatCreate(PETSC_COMM_WORLD, &VvbarOperator);
//    MatCreate(PETSC_COMM_WORLD, &VvtildeOperator);
//
//    MatSetSizes(UhbarOperator, PETSC_DECIDE, PETSC_DECIDE, (nUx - 1) * (nUy - 2), nUx * nUy);  // fixed
//    MatSetSizes(UhtildeOperator, PETSC_DECIDE, PETSC_DECIDE, (nUx - 1) * (nUy - 2), nUx * nUy);  // fixed
//
//    MatSetSizes(UvbarOperator, PETSC_DECIDE, PETSC_DECIDE, nUx * (nUy - 1), nUx * nUy); //fixed
//    MatSetSizes(UvtildeOperator, PETSC_DECIDE, PETSC_DECIDE, nUx * (nUy - 1), nUx * nUy);  //fixed
//
//    MatSetSizes(VhbarOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 1) * nVy, nVx * nVy); // fixed
//    MatSetSizes(VhtildeOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 1) * nVy, nVx * nVy); // fixed
//
//    MatSetSizes(VvbarOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 1), nVx * nVy);  // fixed
//    MatSetSizes(VvtildeOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 1), nVx * nVy);  // fixed
//
//    MatSetFromOptions(UhbarOperator);
//    MatSetFromOptions(UhtildeOperator);
//    MatSetFromOptions(UvbarOperator);
//    MatSetFromOptions(UvtildeOperator);
//    MatSetFromOptions(VhbarOperator);
//    MatSetFromOptions(VhtildeOperator);
//    MatSetFromOptions(VvbarOperator);
//    MatSetFromOptions(VvtildeOperator);
//
//    MatSetUp(UhbarOperator);
//    MatSetUp(UhtildeOperator);
//    MatSetUp(UvbarOperator);
//    MatSetUp(UvtildeOperator);
//    MatSetUp(VhbarOperator);
//    MatSetUp(VhtildeOperator);
//    MatSetUp(VvbarOperator);
//    MatSetUp(VvtildeOperator);
//
//    // Generating operators
//    poseidonUhbarOperator(nUx, nUy, nUinterior, uInteriorIndex, UhbarOperator);
//    poseidonUhtildeOperator(nUx, nUy, nUinterior, uInteriorIndex, UhtildeOperator);
//
//    poseidonUvbarOperator(nUx, nUy, nUinterior, uInteriorIndex, UvbarOperator);
//    poseidonUvtildeOperator(nUx, nUy, nUinterior, uInteriorIndex, UvtildeOperator);
//
//    poseidonVhbarOperator(nVx, nVy, nVinterior, vInteriorIndex, VhbarOperator);
//    poseidonVhtildeOperator(nVx, nVy, nVinterior, vInteriorIndex, VhtildeOperator);
//
//    poseidonVvbarOperator(nVx, nVy, nVinterior, vInteriorIndex, VvbarOperator);
//    poseidonVvtildeOperator(nVx, nVy, nVinterior, vInteriorIndex, VvtildeOperator);
//
//    /*
//     * Generate ddx and ddy operators
//     */
//    Mat dUdxOperator, dUdyOperator, dVdxOperator, dVdyOperator;
//
//    MatCreate(PETSC_COMM_WORLD, &dUdxOperator);
//    MatCreate(PETSC_COMM_WORLD, &dUdyOperator);
//    MatCreate(PETSC_COMM_WORLD, &dVdxOperator);
//    MatCreate(PETSC_COMM_WORLD, &dVdyOperator);
//
//    MatSetSizes(dUdxOperator, PETSC_DECIDE, PETSC_DECIDE, (nUx - 2) * (nUy - 2), (nUx - 1) * (nUy - 2));
//    MatSetSizes(dUdyOperator, PETSC_DECIDE, PETSC_DECIDE, (nUx - 2) * (nUy - 2), nUx * (nUy - 1));
//    MatSetSizes(dVdxOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 2), (nVx - 1) * nVy);
//    MatSetSizes(dVdyOperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 2), (nVx - 2) * (nVy - 1));
//
//    MatSetFromOptions(dUdxOperator);
//    MatSetFromOptions(dUdyOperator);
//    MatSetFromOptions(dVdxOperator);
//    MatSetFromOptions(dVdyOperator);
//
//    MatSetUp(dUdxOperator);
//    MatSetUp(dUdyOperator);
//    MatSetUp(dVdxOperator);
//    MatSetUp(dVdyOperator);
//
//    poseidondUdxOperator(nUx, nUy, dx, dy, dUdxOperator);
//    poseidondUdyOperator(nUx, nUy, dx, dy, dUdyOperator);
//    poseidondVdxOperator(nVx, nVy, dx, dy, dVdxOperator);
//    poseidondVdyOperator(nVx, nVy, dx, dy, dVdyOperator);
//
//    /*
//     * Calculating the convective terms in x direction (U*)
//     */
//    MatMult(UhbarOperator, U, Uhbar);
//    MatMult(UvbarOperator, U, Uvbar);
//    MatMult(UhtildeOperator, U, Uhtilde);
//    MatMult(UvtildeOperator, U, Uvtilde);
//    MatMult(VhbarOperator, V, Vhbar);
//    MatMult(VvbarOperator, V, Vvbar);
//    MatMult(VhtildeOperator, V, Vhtilde);
//    MatMult(VvtildeOperator, V, Vvtilde);
//
//    Vec Uhbar2, UhbarXUhtitlde, UvbarXVhbar, VhbarXUvtilde, UhbarABS, VhbarABS;
//    Vec Uhbar2_gammaXUhbarXUhtilde, UvbarXVhbar_gammaXVhbarUvtilde;
//    Vec UvbarXVhtilde, Vvbar2, VvbarXVvtilde, UvbarABS, VvbarABS;
//    Vec UvbarVhbar_gammaXUvbarVhtilde, Vvbar2_gammaXVvbarXVvtilde;
//
//    Vec UsRHSx, UsRHSy;
//    Vec VsRHSx, VsRHSy;
//    Vec UsRHS, VsRHS;
//
//    VecDuplicate(Uhbar, &Uhbar2);
//    VecDuplicate(Uhbar, &UhbarXUhtitlde);
//    VecDuplicate(Uvbar, &UvbarXVhbar);
//    VecDuplicate(Vhbar, &VhbarXUvtilde);
//    VecDuplicate(Uhbar, &UhbarABS);
//    VecDuplicate(Vhbar, &VhbarABS);
//    VecDuplicate(Uhbar, &Uhbar2_gammaXUhbarXUhtilde);
//    VecDuplicate(Uvbar, &UvbarXVhbar_gammaXVhbarUvtilde);
//    VecDuplicate(Uvbar, &UvbarXVhtilde);
//    VecDuplicate(Vvbar, &Vvbar2);
//    VecDuplicate(Vvbar, &VvbarXVvtilde);
//    VecDuplicate(Uvbar, &UvbarABS);
//    VecDuplicate(Vvbar, &VvbarABS);
//    VecDuplicate(Uvbar, &UvbarVhbar_gammaXUvbarVhtilde);
//    VecDuplicate(Vvbar, &Vvbar2_gammaXVvbarXVvtilde);
//
//    VecCreate(PETSC_COMM_WORLD, &UsRHSx); VecSetSizes(UsRHSx, PETSC_DECIDE, (nUx - 2) * (nUy - 2)); VecSetFromOptions(UsRHSx);
//    VecCreate(PETSC_COMM_WORLD, &UsRHSy); VecSetSizes(UsRHSy, PETSC_DECIDE, (nUx - 2) * (nUy - 2)); VecSetFromOptions(UsRHSy);
//    VecCreate(PETSC_COMM_WORLD, &UsRHS); VecSetSizes(UsRHS, PETSC_DECIDE, (nUx - 2) * (nUy - 2)); VecSetFromOptions(UsRHS);
//    VecCreate(PETSC_COMM_WORLD, &VsRHSx); VecSetSizes(VsRHSx, PETSC_DECIDE, (nVx - 2) * (nVy - 2)); VecSetFromOptions(VsRHSx);
//    VecCreate(PETSC_COMM_WORLD, &VsRHSy); VecSetSizes(VsRHSy, PETSC_DECIDE, (nVx - 2) * (nVy - 2)); VecSetFromOptions(VsRHSy);
//    VecCreate(PETSC_COMM_WORLD, &VsRHS); VecSetSizes(VsRHS, PETSC_DECIDE, (nVx - 2) * (nVy - 2)); VecSetFromOptions(VsRHS);
//
//    VecAbs(UhbarABS);
//    VecAbs(VhbarABS);
//    VecAbs(UvbarABS);
//    VecAbs(VvbarABS);
//
//    VecPointwiseMult(Uhbar, Uhbar, Uhbar2);
//    VecPointwiseMult(UhbarABS, Uhtilde, UhbarXUhtitlde);
//    VecPointwiseMult(Uvbar, Vhbar, UvbarXVhbar);
//    VecPointwiseMult(VhbarABS, Uvtilde, VhbarXUvtilde);
//    VecPointwiseMult(Uvbar, Vhbar, UvbarXVhbar);
//    VecPointwiseMult(UvbarABS, Vhtilde, UvbarXVhtilde);
//    VecPointwiseMult(Vvbar, Vvbar, Vvbar2);
//    VecPointwiseMult(VvbarABS, Vvtilde, VvbarXVvtilde);
//
//    VecWAXPY(Uhbar2_gammaXUhbarXUhtilde, -gamma, UhbarXUhtitlde, Uhbar2);
//    VecWAXPY(UvbarXVhbar_gammaXVhbarUvtilde, -gamma, VhbarXUvtilde, UvbarXVhbar);
//    VecWAXPY(UvbarVhbar_gammaXUvbarVhtilde, -gamma, UvbarXVhtilde, UvbarXVhbar);
//    VecWAXPY(Vvbar2_gammaXVvbarXVvtilde, -gamma, VvbarXVvtilde, Vvbar2);
//
//    MatMult(dUdxOperator, Uhbar2_gammaXUhbarXUhtilde, UsRHSx);
//    MatMult(dUdyOperator, UvbarXVhbar_gammaXVhbarUvtilde, UsRHSy);
//    MatMult(dVdxOperator, UvbarVhbar_gammaXUvbarVhtilde, VsRHSx);
//    MatMult(dVdyOperator, Vvbar2_gammaXVvbarXVvtilde, VsRHSy);
//
//    VecWAXPY(UsRHS, 1, UsRHSx, UsRHSy);
//    VecWAXPY(VsRHS, 1, VsRHSx, VsRHSy);
//
//    VecScale(VsRHS, -1.0 * dt);
//    VecScale(UsRHS, -1.0 * dt);
//
//    PetscScalar *UsRHS_array, *VsRHS_array;
//
//    VecGetArray(UsRHS, &UsRHS_array);
//    VecGetArray(VsRHS, &VsRHS_array);
//
//    // Calculate Us and Vs for convective term
//    Vec Us, Vs;
//    VecDuplicate(U, &Us); VecDuplicate(V, &Vs);
//    VecCopy(U, Us); VecCopy(V, Vs);
//    VecSetValues(Us, nUinterior, uInteriorIndex, UsRHS_array, ADD_VALUES);
//    VecSetValues(Vs, nVinterior, vInteriorIndex, VsRHS_array, ADD_VALUES);
//
//    /*
//     * Solve implicit viscosity
//     */
//    /*
//     * Create matrices
//     */
//    Mat Lu, Lv;
//    MatCreate(PETSC_COMM_WORLD, &Lu); MatCreate(PETSC_COMM_WORLD, &Lv);
//    MatSetSizes(Lu, PETSC_DECIDE, PETSC_DECIDE, (nUx - 2) * (nUy - 2), (nUx - 2) * (nUy - 2));
//    MatSetSizes(Lv, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 2), (nVx - 2) * (nVy - 2));
//    MatSetFromOptions(Lu); MatSetFromOptions(Lv);
//    MatSetUp(Lu); MatSetUp(Lv);
//    /*
//     * Generate Laplacian operator
//     */
//    poseidonLaplacianOperator(Lu, nUx - 2, nUy - 2, dx, dy);
//    poseidonLaplacianOperator(Lv, nVx - 2, nVy - 2, dx, dy);
//
//    MatScale(Lu, -dt / Re);
//    MatShift(Lu, 1.0);
//
//    /*
//     * Setup the right-hand-size for the implicit viscosity calculation
//     */
//    // U-velocity
//    Vec UssRHS;
//    VecCreate(PETSC_COMM_WORLD, &UssRHS); VecSetSizes(UssRHS, PETSC_DECIDE, nUx * nUy); VecSetFromOptions(UssRHS);
//
//    Vec uSouthBoundaryValueScaled, uWestBoundaryValueScaled, uNorthBoundaryValueScaled, uEastBoundaryValueScaled;
//    VecDuplicate(uSouthBoundaryValue, &uSouthBoundaryValueScaled); VecCopy(uSouthBoundaryValue, uSouthBoundaryValueScaled); VecScale(uSouthBoundaryValueScaled, dt / (Re * pow(dy, 2.0)));
//    VecDuplicate(uWestBoundaryValue, &uWestBoundaryValueScaled); VecCopy(uWestBoundaryValue, uWestBoundaryValueScaled); VecScale(uWestBoundaryValueScaled, dt / (Re * pow(dx, 2.0)));
//    VecDuplicate(uNorthBoundaryValue, &uNorthBoundaryValueScaled); VecCopy(uNorthBoundaryValue, uNorthBoundaryValueScaled); VecScale(uNorthBoundaryValueScaled, dt / (Re * pow(dy, 2.0)));
//    VecDuplicate(uEastBoundaryValue, &uEastBoundaryValueScaled); VecCopy(uEastBoundaryValue, uEastBoundaryValueScaled); VecScale(uEastBoundaryValueScaled, dt / (Re * pow(dx, 2.0)));
//
//    PetscScalar *Us_array = new PetscScalar[nUx * nUy]; VecGetArray(Us, &Us_array);
//    PetscScalar *uSouthBoundaryValueScaled_array = new PetscScalar[nUx - 2]; VecGetArray(uSouthBoundaryValueScaled, &uSouthBoundaryValueScaled_array);
//    PetscScalar *uWestBoundaryValueScaled_array = new PetscScalar[nUy - 2]; VecGetArray(uWestBoundaryValueScaled, &uWestBoundaryValueScaled_array);
//    PetscScalar *uNorthBoundaryValueScaled_array = new PetscScalar[nUx - 2]; VecGetArray(uNorthBoundaryValueScaled, &uNorthBoundaryValueScaled_array);
//    PetscScalar *uEastBoundaryValueScaled_array = new PetscScalar[nUy - 2]; VecGetArray(uEastBoundaryValueScaled, &uEastBoundaryValueScaled_array);
//
//    VecSetValues(UssRHS, (nUx - 2) * (nUy - 2), uInteriorIndex, Us_array, INSERT_VALUES);
//    VecSetValues(UssRHS, (nUx - 2), uSouthGhostBoundaryIndex, uSouthBoundaryValueScaled_array, ADD_VALUES);
//    VecSetValues(UssRHS, (nUy - 2), uWestGhostBoundaryIndex, uWestBoundaryValueScaled_array, ADD_VALUES);
//    VecSetValues(UssRHS, (nUx - 2), uNorthGhostBoundaryIndex, uNorthBoundaryValueScaled_array, ADD_VALUES);
//    VecSetValues(UssRHS, (nUy - 2), uEastGhostBoundaryIndex, uEastBoundaryValueScaled_array, ADD_VALUES);
//    VecAssemblyBegin(UssRHS);
//    VecAssemblyEnd(UssRHS);
//
//    // V-velocity
//    Vec VssRHS;
//    VecCreate(PETSC_COMM_WORLD, &VssRHS); VecSetSizes(VssRHS, PETSC_DECIDE, nVx * nVy); VecSetFromOptions(VssRHS);
//
//    Vec vSouthBoundaryValueScaled, vWestBoundaryValueScaled, vNorthBoundaryValueScaled, vEastBoundaryValueScaled;
//    VecDuplicate(vSouthBoundaryValue, &vSouthBoundaryValueScaled); VecCopy(vSouthBoundaryValue, vSouthBoundaryValueScaled); VecScale(vSouthBoundaryValueScaled, dt / (Re * pow(dy, 2.0)));
//    VecDuplicate(vWestBoundaryValue, &vWestBoundaryValueScaled); VecCopy(vWestBoundaryValue, vWestBoundaryValueScaled); VecScale(vWestBoundaryValueScaled, dt / (Re * pow(dx, 2.0)));
//    VecDuplicate(vNorthBoundaryValue, &vNorthBoundaryValueScaled); VecCopy(vNorthBoundaryValue, vNorthBoundaryValueScaled); VecScale(vNorthBoundaryValueScaled, dt / (Re * pow(dy, 2.0)));
//    VecDuplicate(vEastBoundaryValue, &vEastBoundaryValueScaled); VecCopy(vEastBoundaryValue, vEastBoundaryValueScaled); VecScale(vEastBoundaryValueScaled, dt / (Re * pow(dx, 2.0)));
//
//    PetscScalar *Vs_array = new PetscScalar[nVx * nVy]; VecGetArray(Vs, &Vs_array);
//    PetscScalar *vSouthBoundaryValueScaled_array = new PetscScalar[nVx - 2]; VecGetArray(vSouthBoundaryValueScaled, &vSouthBoundaryValueScaled_array);
//    PetscScalar *vWestBoundaryValueScaled_array = new PetscScalar[nVy - 2]; VecGetArray(vWestBoundaryValueScaled, &vWestBoundaryValueScaled_array);
//    PetscScalar *vNorthBoundaryValueScaled_array = new PetscScalar[nVx - 2]; VecGetArray(vNorthBoundaryValueScaled, &vNorthBoundaryValueScaled_array);
//    PetscScalar *vEastBoundaryValueScaled_array = new PetscScalar[nVy - 2]; VecGetArray(vEastBoundaryValueScaled, &vEastBoundaryValueScaled_array);
//
//    VecSetValues(VssRHS, (nVx - 2) * (nVy - 2), vInteriorIndex, Vs_array, INSERT_VALUES);
//    VecSetValues(VssRHS, (nVx - 2), vSouthGhostBoundaryIndex, vSouthBoundaryValueScaled_array, ADD_VALUES);
//    VecSetValues(VssRHS, (nVy - 2), vWestGhostBoundaryIndex, vWestBoundaryValueScaled_array, ADD_VALUES);
//    VecSetValues(VssRHS, (nVx - 2), vNorthGhostBoundaryIndex, vNorthBoundaryValueScaled_array, ADD_VALUES);
//    VecSetValues(VssRHS, (nVy - 2), vEastGhostBoundaryIndex, vEastBoundaryValueScaled_array, ADD_VALUES);
//    VecAssemblyBegin(VssRHS);
//    VecAssemblyEnd(VssRHS);
//
//    // Get the interior nodes values
//    PetscScalar *UssRHSinterior_array = new PetscScalar[(nUx - 2) * (nUy - 2)]();
//    PetscScalar *VssRHSinterior_array = new PetscScalar[(nVx - 2) * (nVy - 2)]();
//
//    VecGetValues(UssRHS, (nUx - 2) * (nUy - 2), uInteriorIndex, UssRHSinterior_array);
//    VecGetValues(VssRHS, (nVx - 2) * (nVy - 2), vInteriorIndex, VssRHSinterior_array);
//
//    Vec UssRHSinterior; VecCreate(PETSC_COMM_WORLD, &UssRHSinterior);
//    VecSetSizes(UssRHSinterior, PETSC_DECIDE, (nUx - 2) * (nUy - 2)); VecSetFromOptions(UssRHSinterior);
//    for (int i = 0; i < (nUx - 2) * (nUy - 2); i++) {
//        VecSetValue(UssRHSinterior, i, UssRHSinterior_array[i], INSERT_VALUES);
//    }
//    VecAssemblyBegin(UssRHSinterior);
//    VecAssemblyEnd(UssRHSinterior);
//
//    Vec VssRHSinterior; VecCreate(PETSC_COMM_WORLD, &VssRHSinterior);
//    VecSetSizes(VssRHSinterior, PETSC_DECIDE, (nVx - 2) * (nVy - 2)); VecSetFromOptions(VssRHSinterior);
//    for (int i = 0; i < (nVx - 2) * (nVy - 2); i++) {
//        VecSetValue(VssRHSinterior, i, VssRHSinterior_array[i], INSERT_VALUES);
//    }
//    VecAssemblyBegin(VssRHSinterior);
//    VecAssemblyEnd(VssRHSinterior);
//
//    // Solve implicit viscosity - u velocity
//    Vec UssInterior;
//    VecDuplicate(UssRHSinterior, &UssInterior);
//    KSP viscositySolverU;
//    KSPCreate(PETSC_COMM_WORLD, &viscositySolverU);
//    KSPSetOperators(viscositySolverU, Lu, Lu);
//    KSPSetFromOptions(viscositySolverU);
//    KSPSolve(viscositySolverU, UssRHSinterior, UssInterior);
//
//    // Solve implicit viscosity - v velocity
//    Vec VssInterior;
//    VecDuplicate(VssRHSinterior, &VssInterior);
//    KSP viscositySolverV;
//    KSPCreate(PETSC_COMM_WORLD, &viscositySolverV);
//    KSPSetOperators(viscositySolverV, Lv, Lv);
//    KSPSetFromOptions(viscositySolverV);
//    KSPSolve(viscositySolverV, VssRHSinterior, VssInterior);
//
//    // Add the boundaries to the U** and V**
//    Vec Uss; VecCreate(PETSC_COMM_WORLD, &Uss); VecSetSizes(Uss, PETSC_DECIDE, nUx * nUy); VecSetFromOptions(Uss);
//    Vec Vss; VecCreate(PETSC_COMM_WORLD, &Vss); VecSetSizes(Vss, PETSC_DECIDE, nVx * nVy); VecSetFromOptions(Vss);
//
//    PetscReal *UssInterior_array = new PetscReal[(nUx - 2) * (nUy - 2)];
//    PetscReal *VssInterior_array = new PetscReal[(nVx - 2) * (nVy - 2)];
//
//    VecGetArray(UssInterior, &UssInterior_array);
//    VecGetArray(VssInterior, &VssInterior_array);
//
//    VecSetValues(Uss, (nUx - 2) * (nUy - 2), uInteriorIndex, UssInterior_array, INSERT_VALUES);
//    VecSetValues(Uss, nUx - 2, uSouthBoundaryIndex, uSouthBoundaryValue_array, INSERT_VALUES);
//    VecSetValues(Uss, nUy - 2, uWestBoundaryIndex, uWestBoundaryValue_array, INSERT_VALUES);
//    VecSetValues(Uss, nUx - 2, uNorthBoundaryIndex, uNorthBoundaryValue_array, INSERT_VALUES);
//    VecSetValues(Uss, nUy - 2, uEastBoundaryIndex, uEastBoundaryValue_array, INSERT_VALUES);
//
//    VecSetValues(Vss, (nVx - 2) * (nVy - 2), vInteriorIndex, VssRHSinterior_array, INSERT_VALUES);
//    VecSetValues(Uss, nVx - 2, vSouthBoundaryIndex, vSouthBoundaryValue_array, INSERT_VALUES);
//    VecSetValues(Uss, nVy - 2, vWestBoundaryIndex, vWestBoundaryValue_array, INSERT_VALUES);
//    VecSetValues(Uss, nVx - 2, vNorthBoundaryIndex, vNorthBoundaryValue_array, INSERT_VALUES);
//    VecSetValues(Uss, nVy - 2, vEastBoundaryIndex, vEastBoundaryValue_array, INSERT_VALUES);
//
//    /*
//     * Pressure correction step
//     */
//    Mat dUssdXoperator, dVssdYoperator;
//    MatCreate(PETSC_COMM_WORLD, &dUssdXoperator); MatCreate(PETSC_COMM_WORLD, &dVssdYoperator);
//    MatSetSizes(dUssdXoperator, PETSC_DECIDE, PETSC_DECIDE, (nUx - 1) * (nUy - 2), nUx * nUy);
//    MatSetSizes(dVssdYoperator, PETSC_DECIDE, PETSC_DECIDE, (nVx - 2) * (nVy - 1), nVx * nVy);
//    MatSetFromOptions(dUssdXoperator); MatSetFromOptions(dVssdYoperator);
//    MatSetUp(dUssdXoperator); MatSetUp(dVssdYoperator);
//
//    poseidondUdX(nUx, nUy, dx, dy, dUssdXoperator);
//    poseidondVdY(nVx, nVy, dx, dy, dVssdYoperator);
//
//    Vec dUssdX, dVssdY;
//    VecCreate(PETSC_COMM_WORLD, &dUssdX); VecSetSizes(dUssdX, PETSC_DECIDE, nPinterior); VecSetFromOptions(dUssdX);
//    VecCreate(PETSC_COMM_WORLD, &dVssdY); VecSetSizes(dVssdY, PETSC_DECIDE, nPinterior); VecSetFromOptions(dVssdY);
//    MatMult(dUssdXoperator, Uss, dUssdX);
//    MatMult(dVssdYoperator, Vss, dVssdY);
//
//    Vec Prhs;
//
//    VecDuplicate(dVssdY, &Prhs);
//    VecCopy(dVssdY, Prhs);
//    VecAXPBY(Prhs, 1 / dt, 1 / dt, dUssdX);
//
//    Mat nablaPoperator;
//    MatCreate(PETSC_COMM_WORLD, &nablaPoperator);
//    MatCreate(PETSC_COMM_WORLD, &nablaPoperator);
//    MatSetSizes(nablaPoperator, PETSC_DECIDE, PETSC_DECIDE, (nPx - 2) * (nPy - 2), (nPx - 2) * (nPy - 2));
//    MatSetFromOptions(nablaPoperator);
//    MatSetUp(nablaPoperator);
//
//    poseidonLaplacianNeumann(nPx, nPy, dx, dy, nablaPoperator);
//
////    poseidondUdxOperator(nUx, nUy, dx, dy, dUssdXoperator);
////    poseidondVdyOperator(nVx - 1, nVy - 2, dx, dy, dVssdYoperator);
//
//    PetscInt vecSize, matSize1, matSize2;
//    VecGetSize(Prhs, &vecSize);
//    cout << vecSize << endl;
////    MatGetSize(dUssdXoperator, &matSize1, &matSize2);
////    cout << matSize1 << "\t\t" << matSize2 << endl;
//
//
////
////
//
////    PetscInt vecSize, matSize1, matSize2;
////    VecGetSize(Us, &vecSize);
////    MatGetSize(Lu, &matSize1, &matSize2);
////    cout << matSize1 << "\t\t" << matSize2 << endl;
////    cout << vecSize << endl;
////    PetscObjectSetName((PetscObject) Lu, "LuMat");
////    PetscViewer matlabViewer;
////    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "solution.output", &matlabViewer);
////    PetscViewerSetFormat(matlabViewer, PETSC_VIEWER_ASCII_MATLAB);
////    MatView(Lu, matlabViewer);
//
////    PetscInt vecSize, vecSize1, vecSize2;
////    VecGetSize(UsRHS, &vecSize);
////    MatGetSize(dVdxOperator, &vecSize1, &vecSize2);
////    cout << vecSize  << endl;
////    cout << vecSize << "\t\t" << vecSize1 << "\t\t" << vecSize2 << endl;
////    VecWAXPY(UsRHS, 1.0, Uhbar2_gammaXUhbarXUhtilde, UvbarXVhbar_gammaXVhbarUvtilde);
////    VecPointwiseMult(Uvbar, Vhbar, Vhbar);
////    VecPointwiseMult(Vhbar, Uvtilde, Uvtilde);
////    VecAXPY(Uhbar, 1.0, Uvtilde);
//
////    PetscInt myLoc[3] = {1, 2, 3};
////    PetscScalar *myArray;
////    VecGetArray(Uhtilde, &myArray);
////    VecSetValues(Uhbar, 3, myLoc, myArray, ADD_VALUES);
//
//    /*
//     * Destroying generated vectors and matrices
//     */
//    VecDestroy(&U); VecDestroy(&V); VecDestroy(&P);
//    VecDestroy(&Uhbar); VecDestroy(&Vhbar);
//    VecDestroy(&Uvbar); VecDestroy(&Vvbar);
//    VecDestroy(&Uhtilde); VecDestroy(&Vhtilde);
//    VecDestroy(&Uvtilde); VecDestroy(&Vvtilde);
//
//    MatDestroy(&UhbarOperator); MatDestroy(&UhtildeOperator);
//    MatDestroy(&UvbarOperator); MatDestroy(&UvtildeOperator);
//    MatDestroy(&VhbarOperator); MatDestroy(&VhtildeOperator);
//    MatDestroy(&VvbarOperator); MatDestroy(&VvtildeOperator);
//
//    MatDestroy(&dUdxOperator); MatDestroy(&dUdyOperator);
//    MatDestroy(&dVdxOperator); MatDestroy(&dVdyOperator);
//
//    VecDestroy(&UsRHSx); VecDestroy(&UsRHSy);
//    VecDestroy(&VsRHSx); VecDestroy(&VsRHSy);
//    VecDestroy(&UsRHS); VecDestroy(&VsRHS);
//
//    VecDestroy(&Uhbar2);
//    VecDestroy(&UhbarXUhtitlde);
//    VecDestroy(&UvbarXVhbar);
//    VecDestroy(&VhbarXUvtilde);
//    VecDestroy(&UhbarABS);
//    VecDestroy(&VhbarABS);
//    VecDestroy(&Uhbar2_gammaXUhbarXUhtilde);
//    VecDestroy(&UvbarXVhbar_gammaXVhbarUvtilde);
//    VecDestroy(&UvbarXVhtilde);
//    VecDestroy(&Vvbar2);
//    VecDestroy(&VvbarXVvtilde);
//    VecDestroy(&UvbarABS);
//    VecDestroy(&VvbarABS);
//    VecDestroy(&UvbarVhbar_gammaXUvbarVhtilde);
//    VecDestroy(&Vvbar2_gammaXVvbarXVvtilde);
//
//    VecDestroy(&Us); VecDestroy(&Vs);
//
//    // Destroying vectors
//    VecDestroy(&uWestBoundaryValue); VecDestroy(&uNorthBoundaryValue); VecDestroy(&uEastBoundaryValue); VecDestroy(&uSouthBoundaryValue);
//    VecDestroy(&vWestBoundaryValue); VecDestroy(&vNorthBoundaryValue); VecDestroy(&vEastBoundaryValue); VecDestroy(&vSouthBoundaryValue);
//
//    VecDestroy(&uSouthBoundaryValueScaled); VecDestroy(&uNorthBoundaryValueScaled);
//    VecDestroy(&uEastBoundaryValueScaled); VecDestroy(&uWestBoundaryValueScaled);
//    VecDestroy(&UssRHS); VecDestroy(&UssRHS);
//    VecDestroy(&vSouthBoundaryValueScaled); VecDestroy(&vNorthBoundaryValueScaled);
//    VecDestroy(&vEastBoundaryValueScaled); VecDestroy(&vWestBoundaryValueScaled);
//    VecDestroy(&VssRHS); VecDestroy(&VssRHS);
//
//    MatDestroy(&Lu); MatDestroy(&Lv);
//    VecDestroy(&UssRHSinterior); VecDestroy(&VssRHSinterior);
//
//    KSPDestroy(&viscositySolverU); KSPDestroy(&viscositySolverV);
//
//    VecDestroy(&Uss); VecDestroy(&Vss);
//
//    MatDestroy(&dUssdXoperator); MatDestroy(&dVssdYoperator);
//
//    VecDestroy(&dUssdX); VecDestroy(&dVssdY);
//    VecDestroy(&Prhs);

//}

void poseidonGetUIndex(PetscInt nUx, PetscInt nUy, PetscInt *uBoundaryIndex, PetscInt *uInteriorIndex,
                       PetscInt *uWestBoundaryIndex, PetscInt *uWestGhostBoundaryIndex,
                       PetscInt *uNorthBoundaryIndex, PetscInt *uNorthGhostBoundaryIndex,
                       PetscInt *uEastBoundaryIndex, PetscInt *uEastGhostBoundaryIndex,
                       PetscInt *uSouthBoundaryIndex, PetscInt *uSouthGhostBoundaryIndex) {
    int boundaryIndexTracker = 0;
    int interiorIndexTracker = 0;
    int westBoundaryIndexTracker = 0;
    int northBoundaryIndexTracker = 0;
    int eastBoundaryIndexTracker = 0;
    int southBoundaryIndexTracker = 0;
    // u-velocity index
    for (int i = 0; i < nUx * nUy; i++) {
        if (i < nUx) {
            uBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
            if ((i != 0) && (i != nUx - 1)) {
                uSouthBoundaryIndex[southBoundaryIndexTracker] = i;
                uSouthGhostBoundaryIndex[southBoundaryIndexTracker] = i + nUx;
                southBoundaryIndexTracker++;
            }
        }
        else if (i%nUx == 0) {
            uBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
            if ((i != 0) && (i != nUx * (nUy - 1))) {
                uWestBoundaryIndex[westBoundaryIndexTracker] = i;
                uWestGhostBoundaryIndex[westBoundaryIndexTracker] = i + 1;
                westBoundaryIndexTracker++;
            }
        }
        else if (i%nUx == nUx - 1) {
            uBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
            if ((i != nUx - 1) && (i != nUx * nUy - 1)) {
                uEastBoundaryIndex[eastBoundaryIndexTracker] = i;
                uEastGhostBoundaryIndex[eastBoundaryIndexTracker] = i - 1;
                eastBoundaryIndexTracker++;
            }
        }
        else if ((i > nUx * (nUy - 1)) && (i < nUx * nUy)) {
            uBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
            if ((nUx * (nUy - 1)) && (i != nUx * nUy - 1)) {
                uNorthBoundaryIndex[northBoundaryIndexTracker] = i;
                uNorthGhostBoundaryIndex[northBoundaryIndexTracker] = i - nUx;
                northBoundaryIndexTracker++;
            }
        }
        else {
            uInteriorIndex[interiorIndexTracker] = i;
            interiorIndexTracker++;
        }
    }
}

void poseidonGetVIndex(PetscInt nVx, PetscInt nVy, PetscInt *vBoundaryIndex, PetscInt *vInteriorIndex,
                       PetscInt *vWestBoundaryIndex, PetscInt *vWestGhostBoundaryIndex,
                       PetscInt *vNorthBoundaryIndex, PetscInt *vNorthGhostBoundaryIndex,
                       PetscInt *vEastBoundaryIndex, PetscInt *vEastGhostBoundaryIndex,
                       PetscInt *vSouthBoundaryIndex, PetscInt *vSouthGhostBoundaryIndex) {
    int boundaryIndexTracker = 0;
    int interiorIndexTracker = 0;
    int westBoundaryIndexTracker = 0;
    int northBoundaryIndexTracker = 0;
    int eastBoundaryIndexTracker = 0;
    int southBoundaryIndexTracker = 0;
    for (int i = 0; i < nVx * nVy; i++) {
        if (i < nVx) {
            vBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
            if ((i != 0) && (i != nVx - 1)) {
                vSouthBoundaryIndex[southBoundaryIndexTracker] = i;
                vSouthGhostBoundaryIndex[southBoundaryIndexTracker] = i + nVx;
                southBoundaryIndexTracker++;
            }
        }
        else if (i%nVx == 0) {
            vBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
            if ((i != 0) && (i != nVx * (nVy - 1))) {
                vWestBoundaryIndex[westBoundaryIndexTracker] = i;
                vWestGhostBoundaryIndex[westBoundaryIndexTracker] = i + 1;
                westBoundaryIndexTracker++;
            }
        }
        else if (i%nVx == nVx - 1) {
            vBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
            if ((nVx - 1) && (nVx * nVy - 1)) {
                vEastBoundaryIndex[eastBoundaryIndexTracker] = i;
                vEastGhostBoundaryIndex[eastBoundaryIndexTracker] = i - 1;
                eastBoundaryIndexTracker++;
            }
        }
        else if ((i > nVx * (nVy - 1)) && (i < nVx * nVy)) {
            vBoundaryIndex[boundaryIndexTracker] = i;
            boundaryIndexTracker++;
            if ((i != nVx * (nVy - 1)) && (i != nVx * nVy - 1)) {
                vNorthBoundaryIndex[northBoundaryIndexTracker] = i;
                vNorthGhostBoundaryIndex[northBoundaryIndexTracker] = i - nVx;
                northBoundaryIndexTracker++;
            }
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

void poseidonLaplacianOperator(Mat &A, PetscInt nx, PetscInt ny, PetscReal dx, PetscReal dy) {
    PetscInt col[5], i;
    PetscScalar value[5];
    dx = pow(dx, 2.0);
    dy = pow(dy, 2.0);
    /*
     * Assemble matrix
     */
    value[0] = 1.0; value[1] = 1; value[2] = -4; value[3] = 1; value[4] = 1;
    for (i=0; i<nx*ny; i++) {
        if ((i % nx == 0) && (i < nx - 1)) { // Element at south-west corner
            col[0] = i;
            col[1] = i + 1;
            col[2] = i + nx;
            value[0] = -2.0 * (dx + dy) / (dx * dy);
            value[1] = 1.0 / dx;
            value[2] = 1.0 / dy;
            MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES);
        }
        else if ((i < nx) && (i != nx - 1)) { // Elements on south row
            col[0] = i - 1;
            col[1] = i;
            col[2] = i + 1;
            col[3] = i + nx;
            value[0] = 1;
            value[1] = -2.0 * (dx + dy) / (dx * dy);
            value[2] = 1.0 / dx;
            value[3] = 1.0 / dy;
            MatSetValues(A, 1, &i, 4, col, value, INSERT_VALUES);
        }
        else if (i == nx - 1) { // Element at south-east corner
            col[0] = i - 1;
            col[1] = i;
            col[2] = i + nx;
            value[0] = 1.0 / dx;
            value[1] = -2.0 * (dx + dy) / (dx * dy);
            value[2] = 1.0 / dy;
            MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES);
        }
        else if ((i % nx == 0) && (i >= nx * (ny - 1))) { // Element at north-west corner
            col[0] = i - nx;
            col[1] = i;
            col[2] = i + 1;
            value[0] = 1.0 / dy;
            value[1] = -2.0 * (dx + dy) / (dx * dy);
            value[2] = 1.0 / dx;
            MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES);
        }
        else if ((i > nx * (ny - 1)) && (i < nx * ny - 1)) { // Elements on north boundary
            col[0] = i - nx;
            col[1] = i - 1;
            col[2] = i;
            col[3] = i + 1;
            value[0] = 1.0 / dy;
            value[1] = 1.0 / dx;
            value[2] = -2.0 * (dx + dy) / (dx * dy);
            value[3] = 1.0 / dx;
            MatSetValues(A, 1, &i, 4, col, value, INSERT_VALUES);
        }
        else if (i == nx * ny - 1) { // Elements on North east corner
            col[0] = i - nx;
            col[1] = i - 1;
            col[2] = i;
            value[0] = 1.0 / dy;
            value[1] = 1.0 / dx;
            value[2] = -2.0 * (dx + dy) / (dx * dy);
            MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES);
        }
        else if ((i % nx == 0) && (i != 0) && (i != nx * (ny - 1))) { // West wall
            col[0] = i - nx;
            col[1] = i;
            col[2] = i + 1;
            col[3] = i + nx;
            value[0] = 1.0 / dy;
            value[1] = -2.0 * (dx + dy) / (dx * dy);
            value[2] = 1.0 / dx;
            value[3] = 1.0 / dy;
            MatSetValues(A, 1, &i, 4, col, value, INSERT_VALUES);
        }
        else if ((i % nx == nx - 1) && (i != 4) && (i != nx * ny - 1)) { // East wall
            col[0] = i - nx;
            col[1] = i - 1;
            col[2] = i;
            col[3] = i + nx;
            value[0] = 1.0 / dy;
            value[1] = 1.0 / dx;
            value[2] = -2.0 * (dx + dy) / (dx * dy);
            value[3] = 1.0 / dy;
            MatSetValues(A, 1, &i, 4, col, value, INSERT_VALUES);
        }
        else {
            col[0] = i - nx;
            col[1] = i - 1;
            col[2] = i;
            col[3] = i + 1;
            col[4] = i + nx;
            value[0] = 1.0 / dy;
            value[1] = 1.0 / dx;
            value[2] = -2.0 * (dx + dy) / (dx * dy);
            value[3] = 1.0 / dx;
            value[4] = 1.0 / dy;
            MatSetValues(A, 1, &i, 5, col, value, INSERT_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void poseidondUdX(PetscInt nUx, PetscInt nUy, PetscReal dx, PetscReal dy, Mat &dUdX) {
    PetscInt operatorPosition[2];
    PetscScalar operatorValue[2];
    PetscInt matRow, matColumn;
    PetscInt rowIndex = 0;
    MatGetSize(dUdX, &matRow, &matColumn);
    operatorValue[0] = -1.0 / dx;
    operatorValue[1] = 1.0 / dx;
    for (int i = 0; i < nUx * nUy; i++) {
        if (fmod(i, nUx) == nUx - 1) {
            continue;
        }
        if (i < nUx) {
            continue;
        }
        if (i > nUx * (nUy - 1) - 1) {
            continue;
        }
        operatorPosition[0] = i;
        operatorPosition[1] = i + 1;
//        cout << rowIndex << "\t" << operatorPosition[0] << "\t" << operatorPosition[1] << endl;
        MatSetValues(dUdX, 1, &rowIndex, 2, operatorPosition, operatorValue, INSERT_VALUES);
        rowIndex++;
    }
    MatAssemblyBegin(dUdX, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(dUdX, MAT_FINAL_ASSEMBLY);
}

void poseidondVdY(PetscInt nVx, PetscInt nVy, PetscReal dx, PetscReal dy, Mat &dVdX) {
    PetscInt operatorPosition[2];
    PetscScalar operatorValue[2];
    PetscInt matRow, matColumn;
    PetscInt rowIndex = 0;
    MatGetSize(dVdX, &matRow, &matColumn);
    operatorValue[0] = -1.0 / dy;
    operatorValue[1] = 1.0 / dy;
    for (int i = 0; i < nVx * nVy; i++) {
        if (fmod(i, nVx) == 0) {
            continue;
        }
        if (fmod(i, nVx) == nVx - 1) {
            continue;
        }
        if (i > nVx * (nVy - 1) - 1) {
            continue;
        }
        operatorPosition[0] = i;
        operatorPosition[1] = i + nVx;
//        cout << rowIndex << "\t" << operatorPosition[0] << "\t" << operatorPosition[1] << endl;
        MatSetValues(dVdX, 1, &rowIndex, 2, operatorPosition, operatorValue, INSERT_VALUES);
        rowIndex++;
    }
    MatAssemblyBegin(dVdX, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(dVdX, MAT_FINAL_ASSEMBLY);
}

void poseidonLaplacianNeumann(PetscInt nPx, PetscInt nPy, PetscReal dx, PetscReal dy, Mat &L) {
    PetscInt operatorPosition[5];
    PetscScalar operatorValue[5];
    PetscInt matRow, matColumn;
    PetscInt rowIndex = 0;
    dx = pow(dx, 2.0);
    dy = pow(dy, 2.0);
    MatGetSize(L, &matRow, &matColumn);
    for (int i = 0; i < nPx * nPy; i++) {
        if ((i < nPx) || (i > nPx * (nPy - 1) - 1) || (fmod(i, nPx) == nPx - 1) || (fmod(i, nPx) == 0)) {
            continue;
        }
        if (i == nPx + 1) {
            operatorPosition[0] = i;
            operatorPosition[1] = i + 1;
            operatorPosition[2] = i + matRow;
            operatorValue[0] = -1.0 * (dx + dy) / (dx * dy);
            operatorValue[1] = 1.0 / dx;
            operatorValue[2] = 1.0 / dy;
            MatSetValues(L, 1, &rowIndex, 3, operatorPosition, operatorValue, INSERT_VALUES);
        }
        else if ((i < nPx * 2) && (i != nPx * 2 - 2)) {
            operatorPosition[0] = i - 1;
            operatorPosition[1] = i;
            operatorPosition[2] = i + 1;
            operatorPosition[3] = i + matRow;
            operatorValue[0] = 1.0 / dx;
            operatorValue[1] = -3.0 / 2.0 * (dx + dy) / (dx * dy);
            operatorValue[2] = 1.0 / dx;
            operatorValue[3] = 1.0 / dy;
            MatSetValues(L, 1, &rowIndex, 4, operatorPosition, operatorValue, INSERT_VALUES);
        }
        else if (i == nPx * 2 - 2) {
            operatorPosition[0] = i - 1;
            operatorPosition[1] = i;
            operatorPosition[2] = i + matRow;
            operatorValue[0] = 1.0 / dx;
            operatorValue[1] = -2.0 / 2.0 * (dx + dy) / (dx * dy);
            operatorValue[2] = 1.0 / dy;
            MatSetValues(L, 1, &rowIndex, 3, operatorPosition, operatorValue, INSERT_VALUES);
        }
        else if ((fmod(i, nPx) == 1) && (i != nPx * (nPy - 2) + 1)) {
            operatorPosition[0] = i - matRow;
            operatorPosition[1] = i;
            operatorPosition[2] = i + 1;
            operatorPosition[3] = i + matRow;
            operatorValue[0] = 1.0 / dy;
            operatorValue[1] = -3.0 / 2.0 * (dx + dy) / (dx * dy);
            operatorValue[2] = 1.0 / dx;
            operatorValue[3] = 1.0 / dy;
            MatSetValues(L, 1, &rowIndex, 4, operatorPosition, operatorValue, INSERT_VALUES);
        }
        else if ((fmod(i, nPx) == nPx - 2) && (i != nPx * (nPy - 1) - 2)) {
            operatorPosition[0] = i - matRow;
            operatorPosition[1] = i - 1;
            operatorPosition[2] = i;
            operatorPosition[3] = i + matRow;
            operatorValue[0] = 1.0 / dy;
            operatorValue[1] = 1.0 / dx;
            operatorValue[2] = -3.0 / 2.0 * (dx + dy) / (dx * dy);
            operatorValue[3] = 1.0 / dy;
            MatSetValues(L, 1, &rowIndex, 4, operatorPosition, operatorValue, INSERT_VALUES);
        }
        else if (i == nPx * (nPy - 2) + 1) {
            operatorPosition[0] = i - matRow;
            operatorPosition[1] = i;
            operatorPosition[2] = i + 1;
            operatorValue[0] = 1.0 / dy;
            operatorValue[1] = -2.0 / 2.0 * (dx + dy) / (dx * dy);
            operatorValue[2] = 1.0 / dx;
            MatSetValues(L, 1, &rowIndex, 3, operatorPosition, operatorValue, INSERT_VALUES);
        }
        else if ((i > nPx * (nPy - 2)) && (i != nPx * (nPy - 1) - 2)) {
            operatorPosition[0] = i - nPx;
            operatorPosition[1] = i - 1;
            operatorPosition[2] = i;
            operatorPosition[3] = i + 1;
            operatorValue[0] = 1.0 / dy;
            operatorValue[1] = 1.0 / dx;
            operatorValue[2] = -3.0 / 2.0 * (dx + dy) / (dx * dy);
            operatorValue[3] = 1.0 / dx;
            MatSetValues(L, 1, &rowIndex, 4, operatorPosition, operatorValue, INSERT_VALUES);
        }
        else if (i == nPx * (nPy - 1) - 2) {
            operatorPosition[0] = i - matRow;
            operatorPosition[1] = i - 1;
            operatorPosition[2] = i;
            operatorValue[0] = 1.0 / dy;
            operatorValue[1] = 1.0 / dx;
            operatorValue[2] = -2.0 / 2.0 * (dx + dy) / (dx * dy);
            MatSetValues(L, 1, &rowIndex, 3, operatorPosition, operatorValue, INSERT_VALUES);
        }
        else {
            operatorPosition[0] = i - matRow;
            operatorPosition[1] = i - 1;
            operatorPosition[2] = i;
            operatorPosition[3] = i + 1;
            operatorPosition[4] = i + matRow;
            operatorValue[0] = 1.0 / dy;
            operatorValue[1] = 1.0 / dx;
            operatorValue[2] = -4.0 / 2.0 * (dx + dy) / (dx * dy);
            operatorValue[3] = 1.0 / dy;
            operatorValue[4] = 1.0 / dx;
            MatSetValues(L, 1, &rowIndex, 5, operatorPosition, operatorValue, INSERT_VALUES);
        }
        rowIndex++;
        cout << rowIndex << endl;
    }
    MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY);
}
