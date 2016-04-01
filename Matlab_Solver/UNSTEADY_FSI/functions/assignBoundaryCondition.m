function [Ue, Ve, Ubc, Vbc, uW, uE, vS, vN] = assignBoundaryCondition(x, y, X, Y, U, V)

%% READ BOUNDARY CONDITIONS
boundaryConditions = dlmread('boundaryCondition.txt');
uw = boundaryConditions(1, 1); vw = boundaryConditions(1, 2);
un = boundaryConditions(2, 1); vn = boundaryConditions(2, 2);
ue = boundaryConditions(3, 1); ve = boundaryConditions(3, 2);
us = boundaryConditions(4, 1); vs = boundaryConditions(4, 2);
% ----------------------------------------------------------------------- %

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
% ----------------------------------------------------------------------- %

%% CALCULATE hx AND hy
hx = x(2) - x(1); hy = y(2) - y(1);
% Lid-driven cavity boundary condition
uN = x * 0 + un;       vN = avg(x) * 0 + vn;
uS = x * 0 + us;       vS = avg(x) * 0 + vs;
uW = avg(y) * 0 + uw;  vW = y * 0 + vw;
uE = avg(y) * 0 + ue;  vE = y * 0 + ve;

% Free-slip boundary condition on top and bottom walls
uN(2:end-1) = U(:,end);
uS(2:end-1) = U(:,1);

% Zero gradient for velocity on east wall
uE = (mean(uW) ./ mean(U(end,:))) .* U(end,:);
vE(2:end-1) = (mean(vW) ./ mean(V(end,:))) .* V(end,:);

% Ubc = dt/Re*([2*uS(2:end-1)' zeros(nx-1,ny-2) 2*uN(2:end-1)']/hx^2+...
%       [uW;zeros(nx-3,ny);uE]/hy^2);
% Vbc = dt/Re*([vS' zeros(nx,ny-3) vN']/hx^2+...
%       [2*vW(2:end-1);zeros(nx-2,ny-1);2*vE(2:end-1)]/hy^2);
  
Ubc = dt / Re * ...
      ([2 * uS(2:end-1)' zeros(nx-1,ny-2) 2 * uN(2:end-1)'] / hy^2 + ...
      [uW; zeros(nx-3,ny); uE] / hx^2);
Vbc = dt / Re * ...
      ([vS' zeros(nx,ny-3) vN'] / hy^2 + ...
      [2 * vW(2:end-1); zeros(nx-2,ny-1); 2 * vE(2:end-1)] / hx^2);
  
Ue = [uW;U;uE]; Ue = [2*uS'-Ue(:,1) Ue 2*uN'-Ue(:,end)];
Ve = [vS' V vN']; Ve = [2*vW-Ve(1,:);Ve;2*vE-Ve(end,:)];