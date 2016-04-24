clc;
clear all;
close all;
format short g;
% ----------------------------------------------------------------------- %
addpath('functions/');
if exist('output', 'dir')
    rewriteData = input('Rewrite data?(y/n)', 's');
    if strcmp(rewriteData, 'y')
        rmdir('output', 's')
        mkdir('output')
    else
        error('Output folder exists and cannot be removed...')
    end
else
    mkdir('output')
end
if exist('figures', 'dir')
    rewriteData = input('Rewrite figures?(y/n)', 's');
    if strcmp(rewriteData, 'y')
        rmdir('figures', 's')
        mkdir('figures')
    else
        error('figures folder exists and cannot be removed...')
    end
else
    mkdir('figures')
end
% ----------------------------------------------------------------------- %
%% DEFINE PHYSICAL PROPERTIES AND DOMAIN DIMENSION
Re = 1e3;                   % Reynolds number
dt = 5e-3;                  % time step
tf = 60;                  % final time
xStart = -0.5;              % Domain begining coordinate (x)
xEnd = 2.5;                 % Domain end coordinate (x)
yStart = -0.5;              % Domain begining coordinate (y)
yEnd = 0.5;                 % Domain end coordinate (y)
nx = 300;                   % number of x-gridpoints
ny = 100;                   % number of y-gridpoints
convCriteria = 0;           % Convergence criteria

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
generateCircle(0.0, 0.01, 0.1, 50);
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
initialize = true;
if ~initialize
    Uinit = dlmread('U.txt');
    Vinit = dlmread('V.txt');
    [U, V, P, Fxhist, Fyhist, Xt] = NSsolve(x, y, X, Y, alpha, beta, peru, Ru, Rut, perv, Rv, Rvt, perp, Rp, Rpt, initialize, Uinit, Vinit);
else
    [U, V, P, Fxhist, Fyhist, Xt] = NSsolve(x, y, X, Y, alpha, beta, peru, Ru, Rut, perv, Rv, Rvt, perp, Rp, Rpt, initialize);
end
dlmwrite('output/U.txt', U);
dlmwrite('output/V.txt', V);
dlmwrite('output/P.txt', P);
dlmwrite('output/Fxhist.txt', Fxhist);
dlmwrite('output/Fyhist.txt', Fyhist);
dlmwrite('output/Xthist.txt', Xt);
% ----------------------------------------------------------------------- %

%% PLOTING
figure,
contourf(Xu, Yu, U, 50, 'linestyle', 'none')
axis('equal')
%% DELETE FILES
% deleteFiles()