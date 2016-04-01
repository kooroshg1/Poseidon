clc;
clear all;
close all;
format short g;
% ----------------------------------------------------------------------- %
m = 0.1;
c = 0.0;
k = 0.01;
f = dlmread('Fyhist.txt');
% omega = 1.;
% f = @(t) sin(omega * t);
% f = @(t) 0;

dt = 5e-3;
nt = length(f); 
Tf = (nt - 0) * dt;

X = zeros(2, nt);
% INITIAL CONDITIONS
X(1) = 0; X(2) = 0;

% STATE MATRIX
A = [0   , 1; ...
     -k/m, -c/m];

for it = 2:nt
    X(:, it) = dt * (A * X(:, it - 1) + [0; f(it - 1)]) + X(:, it - 1);
end

figure,
subplot(2,1,1)
plot(linspace(0, Tf, nt), X(1, :))
subplot(2,1,2)
plot(linspace(0, Tf, nt), X(2, :))