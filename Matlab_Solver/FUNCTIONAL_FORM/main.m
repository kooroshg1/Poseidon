clc;
clear all;
close all;
format short g;
addpath('functions/');
% ----------------------------------------------------------------------- %
%% DEFINE PHYSICAL PROPERTIES AND DOMAIN DIMENSION
Re = 1e2;               % Reynolds number
dt = 1e-2;              % time step
tf = 15.0;               % final time
xStart = -0.5;          % Domain begining coordinate (x)
xEnd = 0.5;             % Domain end coordinate (x)
yStart = -0.5;          % Domain begining coordinate (y)
yEnd = 0.5;             % Domain end coordinate (y)
nx = 100;               % number of x-gridpoints
ny = 100;               % number of y-gridpoints
convCriteria = 1e-10;    % Convergence criteria

information = [Re, dt, tf, xStart, xEnd, yStart, yEnd, nx, ny, convCriteria];
dlmwrite('information.txt', information);
% ----------------------------------------------------------------------- %

%% GENERATE MESH
[x, y, X, Y] = generateMesh();
[xu, yu, Xu, Yu] = generateMesh('U');
[xv, yv, Xv, Yv] = generateMesh('V');
[xp, yp, Xp, Yp] = generateMesh('P');
% ----------------------------------------------------------------------- %

%% DEFINE BOUNDARY CONDITIONS
% Ubc = [uw, un, ue, us] | Vbc = [vw, vn, ve, vs]
Ubc = [1.0; 0.0; 0.0; 0.0];
Vbc = [0.0; 0.0; 0.0; 0.0];

boundaryCondition = [Ubc, Vbc];
dlmwrite('boundaryCondition.txt', boundaryCondition);
U = zeros(nx - 1, ny) + eps; V = zeros(nx, ny - 1) + eps;
[Ue, Ve, Ubc, Vbc, uW, uE, vS, vN] = assignBoundaryCondition(x, y, X, Y, U, V);
% ----------------------------------------------------------------------- %

%% DEFINE LAGRANGIAN POINTS
generateCircle(0.0, 0.0, 0.1, 50);
% ----------------------------------------------------------------------- %

%% DEFINE LAGRANGIAN POINTS VELOCITY
pointCloud = dlmread('pointCloud.txt');
uVelocity = 0 * ones(size(pointCloud, 1), 1);
vVelocity = 0 * ones(size(pointCloud, 1), 1);
lagrangePointsVelocity = [uVelocity, vVelocity];
dlmwrite('lagrangePointsVelocity.txt', lagrangePointsVelocity);
% ----------------------------------------------------------------------- %

%% IMMERSED BOUNDARY PARAMETERS
alpha = -10000;
beta = -100;
% ----------------------------------------------------------------------- %

%% INITIALIZE OPERATORS
[peru, Ru, Rut, perv, Rv, Rvt, perp, Rp, Rpt] = initOperators();
% ----------------------------------------------------------------------- %

%% SOLVE
[U, V, P] = NSsolve(x, y, X, Y, alpha, beta, peru, Ru, Rut, perv, Rv, Rvt, perp, Rp, Rpt);
dlmwrite('U.txt', U);
dlmwrite('V.txt', V);
dlmwrite('P.txt', P);
% ----------------------------------------------------------------------- %

%% PLOTING
figure,
contourf(Xu, Yu, U, 50, 'linestyle', 'none')

%% DELETE FILES
% deleteFiles()