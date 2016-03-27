clear all;
close all;
format short g;
clc;
% ============================== %
nx = 50;
ny = 30;
 
% For U
nUx = nx + 1; nUy = ny + 2;
% For V
nVx = nx + 2; nVy = ny + 1;
% For P
nPx = nx; nPy = ny;

cd Re100
Ufile; U = reshape(U, nUx, nUy); U100 = (U(2:end, 2:end-1) + U(1:end-1, 2:end-1)) / 2;
Vfile; V = reshape(V, nVx, nVy); V100 = (V(2:end-1, 2:end) + V(2:end-1, 1:end-1)) / 2;
Pfile; P100 = reshape(P, nPx, nPy);
cd ..

cd Re1000
Ufile; U = reshape(U, nUx, nUy); U1000 = (U(2:end, 2:end-1) + U(1:end-1, 2:end-1)) / 2;
Vfile; V = reshape(V, nVx, nVy); V1000 = (V(2:end-1, 2:end) + V(2:end-1, 1:end-1)) / 2;
Pfile; P1000 = reshape(P, nPx, nPy);
cd ..

cd Re10000
Ufile; U = reshape(U, nUx, nUy); U10000 = (U(2:end, 2:end-1) + U(1:end-1, 2:end-1)) / 2;
Vfile; V = reshape(V, nVx, nVy); V10000 = (V(2:end-1, 2:end) + V(2:end-1, 1:end-1)) / 2;
Pfile; P10000 = reshape(P, nPx, nPy);
cd ..

lx = 1.0;
ly = 1.0;
x = linspace(0, lx, nx);
y = linspace(0, ly, ny);
[Y, X] = meshgrid(y, x);

U100paper = dlmread('Re100_y.txt');

figure,
plot(U100(25, :), y, 'k', ...
     U100paper(:, 1), U100paper(:, 2), 'ro')

% figure,
% contourf(X, Y, U10000, 50, 'linestyle', 'none')