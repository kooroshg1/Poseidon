clear all;
% close all;
format short g;
clc;
% ============================== %
nx = 100;
ny = 100;
 
% For U
nUx = nx + 1; nUy = ny + 2;
% For V
nVx = nx + 2; nVy = ny + 1;
% For P
nPx = nx; nPy = ny;
% for Us
% nx = nx - 1; ny = ny;
% for Vs
% nx = nx; ny = ny - 1;

cd Debug/t50
Ufile; U = reshape(U, nUx, nUy); U = (U(2:end, 2:end-1) + U(1:end-1, 2:end-1)) / 2;
Vfile; V = reshape(V, nVx, nVy); V = (V(2:end-1, 2:end) + V(2:end-1, 1:end-1)) / 2;
Pfile; P = reshape(P, nPx, nPy);
cd ..

lx = 1.0;
ly = 1.0;
x = linspace(0, lx, nx);
y = linspace(0, ly, ny);
[Y, X] = meshgrid(y, x);

figure(1),
% contourf(X, Y, U, 50, 'linestyle', 'none')
hold on
plot(U(50, :), y, 'b')