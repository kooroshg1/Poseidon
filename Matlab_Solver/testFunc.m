clear all;
close all;
format short g;
clc;
% ======================================================================= %
x = linspace(0, 1, 100);
y = linspace(0, 1, 100);
[Y, X] = meshgrid(y, x);

% delta1 = deltaFunction(0.5, 0.5, X, Y, 0.01);
% delta2 = deltaFunction(0.25, 0.25, X, Y, 0.01);
% delta = delta1 + delta2;

theta = linspace(0, 2*pi, 10);
xs = 0.5 + 0.1 * cos(theta);
ys = 0.5 + 0.1 * sin(theta);
delta = deltaFunction(xs, ys, X, Y, 0.01);

trapz(y, trapz(x, delta))

figure,
surf(X, Y, delta)