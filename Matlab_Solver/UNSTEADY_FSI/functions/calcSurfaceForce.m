function [Fx, Fy] = calcSurfaceForce(P, Dpx, Dpy, xs, ys)

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
% ----------------------------------------------------------------------- %

%% DEFINE DISTANCE BETWEEN TWO NODES IN THE SURFACE OF IMMERSED BOUNDARY
ds = sqrt((xs(2) - xs(1))^2 + (ys(2) - ys(1))^2);
% ----------------------------------------------------------------------- %

%% CALCULATE THE PRESSURE ON THE SURFACE OF THE IMMERSED BOUNDARY
Pb = diag(Dpx * P * Dpy') * hx * hy;
% ----------------------------------------------------------------------- %

%% CALCULATE NORMAL VECTOR TO THE SURFACE OF THE IMMERSED BOUNDARY
tangentVector = [(xs(3:end) - xs(1:end - 2))', (ys(3:end) - ys(1:end - 2))'];
tangentVector = normr(tangentVector);
rMat = [cosd(-90), -sind(-90); sind(-90), cosd(-90)];
normalVector = zeros(size(tangentVector));
for i = 1:size(normalVector, 1)
    normalVector(i, :) = (rMat * tangentVector(i, :)')';
end
% ----------------------------------------------------------------------- %

%% CALCULATE FORCES
Fx = -Pb(2:end-1) .* normalVector(:, 1) * ds; Fx = sum(Fx);
Fy = -Pb(2:end-1) .* normalVector(:, 2) * ds; Fy = sum(Fy);
% ----------------------------------------------------------------------- %