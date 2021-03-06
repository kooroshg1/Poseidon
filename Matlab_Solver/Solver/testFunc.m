clear all;
close all;
format short g;
clc;
% ======================================================================= %
n = 100;
x = linspace(0, 1, n); dx = x(2) - x(1);
y = linspace(0, 1, n); dy = y(2) - y(1);
[Y, X] = meshgrid(y, x);

% delta1 = deltaFunction(0.5, 0.5, X, Y, 0.01);
% delta2 = deltaFunction(0.25, 0.25, X, Y, 0.01);
% delta = delta1 + delta2;

eta = dx / atanh(sqrt(1 - 0.9));
ns = 10;
theta = linspace(0, 2*pi, ns);
xs = 0.5 + 0.1 * cos(theta);
ys = 0.5 + 0.1 * sin(theta);

Dx = zeros(ns, n);
Dy = zeros(ns, n);
for i=1:ns
    [deltaX, deltaY] = deltaFunction(xs(i), ys(i), x, y, eta);
    Dx(i, :) = deltaX;
    Dy(i, :) = deltaY;
end

% (deltaX * X * deltaX') * dx^2
% (deltaY * Y * deltaY') * dy^2
xCalc = diag((Dx * X * Dx')) * dx^2;
yCalc = diag((Dy * Y * Dy')) * dy^2;
% trapz(x, deltaX)

figure,
plot(xCalc, yCalc)
axis('equal')

% figure,
% plot(x, deltaX)
% trapz(y, trapz(x, delta))
% 
% figure,
% surf(X, Y, delta)