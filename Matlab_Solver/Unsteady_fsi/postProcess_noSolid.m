clc;
clear all;
close all;
format short g;
% ----------------------------------------------------------------------- %
addpath('functions/');

%% READ MESH INFO
meshInfo = dlmread('information.txt','',1,0);
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

%% GENERATE MESH
[xu, yu, Xu, Yu] = generateMesh('U');
[xv, yv, Xv, Yv] = generateMesh('V');
[xp, yp, Xp, Yp] = generateMesh('P');
% ----------------------------------------------------------------------- %

%% READ DATA
U = dlmread('output/U.txt');
V = dlmread('output/V.txt');
P = dlmread('output/P.txt');
% ----------------------------------------------------------------------- %

figure,
contourf(Xu, Yu, U, 40, 'linestyle', 'none')
axis('equal')

figure,
plot(Yu(50, :), U(50, :))