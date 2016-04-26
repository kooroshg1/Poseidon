clc;
clear all;
close all;
format short g;
% ----------------------------------------------------------------------- %
addpath('functions/');

RE = 10000;
informationFileName = ['Poseidon_results/RE', num2str(RE), '/information.txt'];
%% READ MESH INFO
meshInfo = dlmread(informationFileName, '', 1, 0);
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
[xu, yu, Xu, Yu] = generateMesh('U', informationFileName);
[xv, yv, Xv, Yv] = generateMesh('V', informationFileName);
[xp, yp, Xp, Yp] = generateMesh('P', informationFileName);
% ----------------------------------------------------------------------- %

%% READ DATA
dataFileName = ['Poseidon_results/RE', num2str(RE)];
U = dlmread([dataFileName '/U.txt']);
V = dlmread([dataFileName '/V.txt']);
P = dlmread([dataFileName '/P.txt']);
% ----------------------------------------------------------------------- %

%% READ GHIA EL AL. RESULTS
ghiaFileName = ['Ghia_Results/RE', num2str(RE), '_u.txt'];
U_ghia = dlmread(ghiaFileName, ',', 1, 0);
ghiaFileName = ['Ghia_Results/RE', num2str(RE), '_v.txt'];
V_ghia = dlmread(ghiaFileName, ',', 1, 0);
% ----------------------------------------------------------------------- %

%% PLOTTING
figure,
plot(U(nx/2, :), Yu(nx/2, :) + 0.5, 'k', ...
     U_ghia(:, 1), U_ghia(:, 2), 'ro')
title(['Re = ', num2str(RE)])
xlabel('u-velocity')
ylabel('y')
legend('Poseidon', 'Ghia et al.')

figure,
plot(Xv(:, ny/2) + 0.5, V(:, ny/2), 'k', ...
     V_ghia(:, 1), V_ghia(:, 2), 'ro')
title(['Re = ', num2str(RE)])
xlabel('x')
ylabel('v-velocity')
legend('Poseidon', 'Ghia et al.')
% ----------------------------------------------------------------------- %