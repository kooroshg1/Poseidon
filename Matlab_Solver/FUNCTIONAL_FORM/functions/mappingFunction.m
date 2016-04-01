function [Dux, Duy, Dvx, Dvy, Dpx, Dpy] = mappingFunction()

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

%% READ LAGRANGIAN POINTS
pointCloud = dlmread('pointCloud.txt');
xs = pointCloud(:, 1)';
ys = pointCloud(:, 2)';
ns = length(xs);
% ----------------------------------------------------------------------- %

%% GENERATE MAPPING FUNCTIONS
[xu, yu, Xu, Yu] = generateMesh('U');
[xv, yv, Xv, Yv] = generateMesh('V');
[xp, yp, Xp, Yp] = generateMesh('P');

eta = hx / atanh(sqrt(1 - 0.25));
Dux = sparse(ns, nx - 1); Duy = sparse(ns, ny);
Dvx = sparse(ns, nx); Dvy = sparse(ns, ny - 1);
Dpx = sparse(ns, nx); Dpy = sparse(ns, ny);
for i=1:ns
    [deltaX, deltaY] = deltaFunction(xs(i), ys(i), xu, yu, eta);
    Dux(i, :) = deltaX; Duy(i, :) = deltaY;
    [deltaX, deltaY] = deltaFunction(xs(i), ys(i), xv, yv, eta);
    Dvx(i, :) = deltaX; Dvy(i, :) = deltaY;
    [deltaX, deltaY] = deltaFunction(xs(i), ys(i), xp, yp, eta);
    Dpx(i, :) = deltaX; Dpy(i, :) = deltaY;
end
% ----------------------------------------------------------------------- %