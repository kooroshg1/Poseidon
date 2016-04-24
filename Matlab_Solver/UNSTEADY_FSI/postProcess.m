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

%% MOVE FIGURE
% fileList = dir('figures/');
% try
%     rmdir('filteredFigures', 's')
%     mkdir('filteredFigures')
% catch
% end
% fileIndex = 1;
% for iF = 3:length(fileList)
%     fileName = fileList(iF).name;
%     fileName = str2double(fileName(1:end-4));
%     if rem(fileName, 50) == 1
%         fileName = [num2str(fileIndex, '%05d'), '.png'];
%         fileNameSource = ['figures/', fileList(iF).name];
%         fileNameDestination = ['filteredFigures/', fileName];
%         fileIndex = fileIndex + 1;
%         copyfile(fileNameSource, fileNameDestination);
%     end
% end
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
U = dlmread('output/U.txt');
V = dlmread('output/V.txt');
P = dlmread('output/P.txt');
Fxhist = dlmread('output/Fxhist.txt');
Fyhist = dlmread('output/Fyhist.txt');
Xthist = dlmread('output/Xthist.txt');
% ----------------------------------------------------------------------- %

Ub = diag(Dux * U * Duy') * hy * hx;
Vb = diag(Dvx * V * Dvy') * hy * hx;
Pb = diag(Dpx * P * Dpy') * hy * hx;

% figure,
% subplot(1,2,1)
% plot(t, Fxhist)
% subplot(1,2,2)
% plot(t, Fyhist)

figure,
plot(t, Xthist(2, :))