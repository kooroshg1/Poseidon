% clear all;
close all;
format short g;
clc;
% ============================== %
nx = 50;
ny = 30;
 
% For U
nUx = nx + 1; nUy = ny + 2;
% For V
% nVx = nx + 2; nVy = ny + 1;
% For P
% nPx = nx; nPy = ny;
% for Us
% nUx = nx - 1; nUy = ny;
% for Vs
% nx = nx; ny = ny - 1;

cd Debug
Ufile; U = reshape(U, nUx, nUy); %U = (U(2:end, 2:end-1) + U(1:end-1, 2:end-1)) / 2;
% Vfile; V = reshape(V, nVx, nVy); V = (V(2:end-1, 2:end) + V(2:end-1, 1:end-1)) / 2;
% Pfile; P = reshape(P, nPx, nPy);
cd ..

lx = 1.0;
ly = 1.0;
x = linspace(0, lx, nUx); y = linspace(0, ly, nUy);
% x = linspace(0, lx, nPx); y = linspace(0, ly, nPy);
[Y, X] = meshgrid(y, x);

figure,
contourf(X, Y, U, 50, 'linestyle', 'none')
caxis([0, 2])
colorbar
% contourf(X, Y, P, 50, 'linestyle', 'none')