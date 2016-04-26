clc;
clear all;
close all;
format short g;
% ----------------------------------------------------------------------- %
addpath('functions/');
if exist('output', 'dir')
%     rewriteData = input('Rewrite data?(y/n)', 's');
%     if strcmp(rewriteData, 'y')
        rmdir('output', 's')
        mkdir('output')
%     else
%         error('Output folder exists and cannot be removed...')
%     end
else
    mkdir('output')
end
if exist('figures', 'dir')
%     rewriteData = input('Rewrite figures?(y/n)', 's');
%     if strcmp(rewriteData, 'y')
        rmdir('figures', 's')
        mkdir('figures')
%     else
%         error('figures folder exists and cannot be removed...')
%     end
else
    mkdir('figures')
end
% ----------------------------------------------------------------------- %
%% DEFINE PHYSICAL PROPERTIES AND DOMAIN DIMENSION
Re = 1e3;                   % Reynolds number
dt = 2e-2;                  % CFL Number
tf = 20;                  % final time
xStart = 0;              % Domain begining coordinate (x)
xEnd = 22;                 % Domain end coordinate (x)
yStart = 0;              % Domain begining coordinate (y)
yEnd = 4.1;                 % Domain end coordinate (y)
nx = 440;                   % number of x-gridpoints
ny = 82;                   % number of y-gridpoints
convCriteria = 0;           % Convergence criteria
Nl = 50;                    % Number of lagrangian points on solid boundary

information = [Re, dt, tf, xStart, xEnd, yStart, yEnd, nx, ny, convCriteria, Nl];
solverInformationFile = fopen('information.txt','w');
fprintf(solverInformationFile, '%-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n', 'Re', 'dt', 'tf', 'xStart', 'xEnd', 'yStart', 'yEnd', 'nx', 'ny', 'convCriteria', 'Nl');
fprintf(solverInformationFile, '%-12.4f %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f %-12i %-12i %-12.4f %-12i\n', Re, dt, tf, xStart, xEnd, yStart, yEnd, nx, ny, convCriteria, Nl);
fclose(solverInformationFile);
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
generateCircle(1.5, 2.5, 0.5, Nl);
% generateEmpty();
% ----------------------------------------------------------------------- %

%% DEFINE LAGRANGIAN POINTS VELOCITY
try
    pointCloud = dlmread('pointCloud.txt');
    uVelocity = 0 * ones(size(pointCloud, 1), 1);
    vVelocity = 0 * ones(size(pointCloud, 1), 1);
    lagrangePointsVelocity = [uVelocity, vVelocity];
    dlmwrite('lagrangePointsVelocity.txt', lagrangePointsVelocity);
catch
    disp('no solid domain defined...');
end
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
    [U, V, P] = NSsolve(x, y, X, Y, alpha, beta, peru, Ru, Rut, perv, Rv, Rvt, perp, Rp, Rpt, initialize, Uinit, Vinit);
else
    [U, V, P] = NSsolve(x, y, X, Y, alpha, beta, peru, Ru, Rut, perv, Rv, Rvt, perp, Rp, Rpt, initialize);
end
dlmwrite('output/U.txt', U);
dlmwrite('output/V.txt', V);
dlmwrite('output/P.txt', P);
% ----------------------------------------------------------------------- %

figure,
contourf(Xu, Yu, U, 50, 'linestyle', 'none'),
hold on
fill(pointCloud(:, 1), pointCloud(:, 2), 'w'),
axis('equal')

%% DELETE FILES
% deleteFiles()
