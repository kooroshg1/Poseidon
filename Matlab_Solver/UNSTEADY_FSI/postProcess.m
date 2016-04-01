clc;
clear all;
close all;
format short g;
% ----------------------------------------------------------------------- %
addpath('functions/');

%% READ MESH INFO
meshInfo = dlmread('information.txt');
Re = meshInfo(1);
dt = meshInfo(2);
tf = meshInfo(3);
xStart = meshInfo(4);
xEnd = meshInfo(5);
yStart = meshInfo(6);
yEnd = meshInfo(7);
nx = meshInfo(8);
ny = meshInfo(9);
convCriteria = meshInfo(10);
hx = (xEnd - xStart) / nx;
hy = (yEnd - yStart) / ny;
nt = ceil(tf/dt);
t = linspace(0, tf, nt);
% ----------------------------------------------------------------------- %

%% DEFINE DELTA FUNCTION (MATRIX) TO MAP RESULTS BETWEEN LAGRANGIAN AND EULERIAN DOMAINS
[Dux, Duy, Dvx, Dvy, Dpx, Dpy] = mappingFunction();
% ----------------------------------------------------------------------- %

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
U = dlmread('U.txt');
V = dlmread('V.txt');
P = dlmread('P.txt');
Fxhist = dlmread('Fxhist.txt');
Fyhist = dlmread('Fyhist.txt');
% ----------------------------------------------------------------------- %

Ub = diag(Dux * U * Duy') * hy * hx;
Vb = diag(Dvx * V * Dvy') * hy * hx;
Pb = diag(Dpx * P * Dpy') * hy * hx;

% figure,
% plot(Pb)
figure,
subplot(1,2,1)
plot(t, Fxhist)
ylim([0.07, 0.08])
xlim([10.0, tf])
subplot(1,2,2)
plot(t, Fyhist)