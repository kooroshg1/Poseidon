% clc;
% clear all;
% close all;
% format short g;
% % ======================================================================= %
function [U, V, P] = NSsolve(x, y, X, Y, alpha, beta, peru, Ru, Rut, perv, Rv, Rvt, perp, Rp, Rpt)

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

%% INITIALIZE VARIABLES
initialize = true;
if initialize
    U = zeros(nx - 1, ny) + eps; V = zeros(nx, ny - 1) + eps;
end
% ----------------------------------------------------------------------- %

%% ASSIGN BOUNDARY CONDITIONS
[Ue, Ve, Ubc, Vbc, uW, uE, vS, vN] = assignBoundaryCondition(x, y, X, Y, U, V);
% ----------------------------------------------------------------------- %

%% READ LAGRANGIAN POINTS
pointCloud = dlmread('pointCloud.txt');
xs = pointCloud(:, 1)';
ys = pointCloud(:, 2)';
ns = length(xs);
% ----------------------------------------------------------------------- %

%% DEFINE DELTA FUNCTION (MATRIX) TO MAP RESULTS BETWEEN LAGRANGIAN AND EULERIAN DOMAINS
[Dux, Duy, Dvx, Dvy, Dpx, Dpy] = mappingFunction();
% ----------------------------------------------------------------------- %

%% VELOCITIES AT THE LAGRANGIAN POINTS
lagrangePointsVelocity = dlmread('lagrangePointsVelocity.txt');
uVelocity = lagrangePointsVelocity(:, 1);
vVelocity = lagrangePointsVelocity(:, 2);
% ----------------------------------------------------------------------- %

%% BEGIN SOLUTION
fxt = 0;
fyt = 0;
Fx = 0;
Fy = 0;
fprintf(', time loop\n--20%%--40%%--60%%--80%%-100%%\n')
tic
for k = 1:nt
    %% SAVE VARIABLES FOR CONVERGENCE STUDY
    Uold = U;
    Vold = V; 
    % ------------------------------------------------------------------- %
    
    %% APPLY BOUNDARY CONDITIONS
    [Ue, Ve, Ubc, Vbc, uW, uE, vS, vN] = assignBoundaryCondition(x, y, X, Y, U, V);
    % ------------------------------------------------------------------- %
    
    gamma = min(1.2*dt*max(max(max(abs(U)))/hx,max(max(abs(V)))/hy),1);

    %% INTERPOLATE U AND V AT THE LAGRANGIAN POINTS
    UX = diag(Dux * U * Duy') * hx * hy;
    VX = diag(Dvx * V * Dvy') * hx * hy;
    % ------------------------------------------------------------------- %
    
    %% CALCULATE THE FORCE TERMS
    fxt = fxt + (UX - uVelocity) * dt; fx = alpha * fxt + beta * (UX - uVelocity); Fx = Dux' * diag(fx) * Duy * hx * hy;
    fyt = fyt + (VX - vVelocity) * dt; fy = alpha * fyt + beta * (VX - vVelocity); Fy = Dvx' * diag(fy) * Dvy * hx * hy;
    % ------------------------------------------------------------------- %

    %% TREAT NONLINEAR TERMS
    Ua = avg(Ue')'; Ud = diff(Ue')'/2;
    Va = avg(Ve);   Vd = diff(Ve)/2;
    UVx = diff(Ua.*Va-gamma*abs(Ua).*Vd)/hx;
    UVy = diff((Ua.*Va-gamma*Ud.*abs(Va))')'/hy;

    Ua = avg(Ue(:,2:end-1));   Ud = diff(Ue(:,2:end-1))/2;
    Va = avg(Ve(2:end-1,:)')'; Vd = diff(Ve(2:end-1,:)')'/2;
    U2x = diff(Ua.^2-gamma*abs(Ua).*Ud)/hx;
    V2y = diff((Va.^2-gamma*abs(Va).*Vd)')'/hy;

    U = U-dt*(UVy(2:end-1,:)+U2x) + dt * Fx;
    V = V-dt*(UVx(:,2:end-1)+V2y) + dt * Fy;
    % ------------------------------------------------------------------- %

    %% IMPLICIT VISCOSITY
    rhs = reshape(U + Ubc,[],1);
    u(peru) = Ru\(Rut\rhs(peru));
    U = reshape(u,nx-1,ny);

    rhs = reshape(V + Vbc,[],1);
    v(perv) = Rv\(Rvt\rhs(perv));
    V = reshape(v,nx,ny-1);
    % ------------------------------------------------------------------- %

    %% PRESSURE CORRECTION
    rhs = reshape(diff([uW;U;uE])/hx+diff([vS' V vN']')'/hy,[],1) / dt;
    p(perp) = -Rp\(Rpt\rhs(perp));
    P = reshape(p,nx,ny);


    U = U-diff(P)/hx * dt;
    V = V-diff(P')'/hy * dt;
    % ------------------------------------------------------------------- %
    
    %% CHECK FOR CONVERGENCE
    Uerr = abs(U - Uold); Uerr = max(max(Uerr));
    Verr = abs(V - Vold); Verr = max(max(Verr));
    if (Uerr < convCriteria) && (Verr < convCriteria)
        fprintf('\n');
        fprintf('CONVERGENCE SATISFIED')
        break;
    end
%     figure(1)
%     subplot(2,1,1)
%     plot(k, Uerr, 'ko')
%     hold('on')
%     subplot(2,1,2)
%     plot(k, Verr, 'ko')
%     hold('on')
    % ------------------------------------------------------------------- %
    
    %% TIMER
    if floor(25*k/nt)>floor(25*(k-1)/nt), fprintf('.'), end
    % ------------------------------------------------------------------- %
    
    %% CHECK FOR NAN
    if max(max(isnan(U))) || max(max(isnan(V)))
        disp('NAN');
        break
    end
    % ------------------------------------------------------------------- %
    
    %% PLOT
%     figure(1),
%     contourf(Xu, Yu, U, 50, 'linestyle', 'none')
%     axis('equal')
%     % ------------------------------------------------------------------- %
    
end
fprintf('\n')
toc