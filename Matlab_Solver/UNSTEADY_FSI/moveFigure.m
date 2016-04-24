clc;
clear all;
close all;
format short g;
% -------------------------------------------------- %
%% READ LAGRANGIAN POINTS
pointCloud = dlmread('pointCloud.txt');
xs = pointCloud(:, 1)';
ys = pointCloud(:, 2)';
ns = length(xs);
% ----------------------------------------------------------------------- %

%% GENERATE MESH
[xu, yu, Xu, Yu] = generateMesh('U');
[xv, yv, Xv, Yv] = generateMesh('V');
[xp, yp, Xp, Yp] = generateMesh('P');
% ----------------------------------------------------------------------- %

%% READ DATA
U = dlmread('output/U.txt');
V = dlmread('output/V.txt');
P = dlmread('output/P.txt');
Fxhist = dlmread('output/Fxhist.txt');
Fyhist = dlmread('output/Fyhist.txt');
Xthist = dlmread('output/Xthist.txt');
% ----------------------------------------------------------------------- %

figure,
contourf(Xu, Yu, U)
hold on
fill(xs, ys, 'w')
axis('equal')