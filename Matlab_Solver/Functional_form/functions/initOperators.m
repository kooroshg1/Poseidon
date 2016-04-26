function [peru, Ru, Rut, perv, Rv, Rvt, perp, Rp, Rpt] = initOperators()

%% READ MESH INFO
meshInfo = dlmread('information.txt','',1,0);
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

%% INITIALIZE DIFFERENTIAL OPERATORS
Lp = kron(speye(ny), K1(nx, hx, 1)) + kron(K1(ny, hy, 1), speye(nx));
Lp(1, 1) = 3 / 2 * Lp(1, 1);
perp = symamd(Lp); Rp = chol(Lp(perp, perp)); Rpt = Rp';

Lu = speye((nx-1) * ny) + dt / Re * (kron(speye(ny), K1(nx - 1, hx, 2)) + ...
     kron(K1(ny, hy, 3), speye(nx - 1)));
peru = symamd(Lu); Ru = chol(Lu(peru, peru)); Rut = Ru';

Lv = speye(nx * (ny - 1)) + dt / Re * (kron(speye(ny - 1), K1(nx, hx, 3)) + ...
     kron(K1(ny - 1, hy, 2), speye(nx)));
perv = symamd(Lv); Rv = chol(Lv(perv, perv)); Rvt = Rv';

Lq = kron(speye(ny - 1), K1(nx - 1, hx, 2)) + kron(K1(ny - 1, hy, 2), speye(nx - 1));
perq = symamd(Lq); Rq = chol(Lq(perq, perq)); Rqt = Rq';
% ----------------------------------------------------------------------- %