% clc;
% clear all;
% close all;
% format short g;
% % ======================================================================= %
function [U, V, P, Fxhist, Fyhist, Xt] = NSsolve(x, y, X, Y, alpha, beta, peru, Ru, Rut, perv, Rv, Rvt, perp, Rp, Rpt, initialize, Uinit, Vinit)
if nargin == 16
    fprintf('initializeing with zeros ...\n');
else
    fprintf('initializeing with your values ...\n');
end

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
if initialize
    U = zeros(nx - 1, ny) + eps; V = zeros(nx, ny - 1) + eps;
else
    U = Uinit; V = Vinit;
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

%% SAVE INITIAL LOCATION AND VELOCITY OF LAGRANGIAN POINTS
pointCloudInitLoc = pointCloud;
lagrangePointsInitVelocity = lagrangePointsVelocity;
% ----------------------------------------------------------------------- %

%% GENERATE MESH
[xu, yu, Xu, Yu] = generateMesh('U');
[xv, yv, Xv, Yv] = generateMesh('V');
[xp, yp, Xp, Yp] = generateMesh('P');
% ----------------------------------------------------------------------- %

%% STRUCTURAL MODEL
k = 1.0; m = 1.0; c = 0.0;
A = [0   , 1; ...
	 -k/m, -c/m];
Xt = zeros(2, nt);

%% BEGIN SOLUTION
if initialize
    fxt = 0;       fyt = 0;
else
    fxt = dlmread('fxt.txt');       fyt = dlmread('fyt.txt');
end
Fxhist = zeros(nt, 1); Fyhist = zeros(nt, 1);
fprintf('Solving ...\n--20%%--40%%--60%%--80%%-100%%\n')
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
    % ------------------------------------------------------------------- %
    
    %% CALCULATE SURFACE FORCES
    [Fx, Fy] = calcSurfaceForce(P, Dpx, Dpy, xs, ys);
    Fxhist(k) = Fx; Fyhist(k) = Fy;
    
    %% SOLVE DYNAMIC EQUATION OF MOTION
    if k > 0
        %% BACKWARD EULER TIME INTEGRATION
%         Xt(:, k) = dt * (A * Xt(:, k - 1) + [0; Fxhist(k)] / m) + Xt(:, k - 1);
        
        %% ADAMS-BASHFORTH TIME INTEGRATION
        if k == 1
            Xt(:, k + 1) = (A * Xt(:, k) + [0; Fxhist(k)] / m) * dt + Xt(:, k);
        else
            Xt(:, k + 1) = Xt(:, k) + ...
                                 dt * (3/2 * A * Xt(:, k) + 3/2 * [0; Fxhist(k)] / m + ...
                                         -1/2 * A * Xt(:, k) - 1/2 * [0;Fxhist(k - 1)] / m);
        end
        
        pointCloud(:, 2) = pointCloudInitLoc(:, 2) + Xt(1, k + 1);
        lagrangePointsVelocity(:, 2) = lagrangePointsInitVelocity(:, 2) + Xt(2, k + 1);
        
        dlmwrite('pointCloud.txt', pointCloud);
        dlmwrite('lagrangePointsVelocity.txt', lagrangePointsVelocity);
        [Dux, Duy, Dvx, Dvy, Dpx, Dpy] = mappingFunction();
        
        xs = pointCloud(:, 1)';
        ys = pointCloud(:, 2)';
        
        uVelocity = lagrangePointsVelocity(:, 1);
        vVelocity = lagrangePointsVelocity(:, 2);
    end
    
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
    % VELOCITY CONTOUR
    figure(1),
    fileName = ['figures/', num2str(k, '%010d'), '.png'];
    contourf(Xu, Yu, U, 50, 'linestyle', 'none')
    hold('on')
    area(xs, ys)
    hold('off')
    axis('equal')
    title([num2str(k) '/' num2str(nt)])
    colorbar
    saveas(gcf, fileName)

    % Fx AND Fy ON THE BOUNDARY
    figure(2),
    subplot(1, 2, 1)
    plot(k, Fx, 'ko')
    xlabel('# of iterations')
    ylabel('Fx')
    title([num2str(k) '/' num2str(nt)])
    hold on
    subplot(1, 2, 2)
    plot(k, Fy, 'ko')
    xlabel('# of iterations')
    ylabel('Fy')
    title([num2str(k) '/' num2str(nt)])
    hold on
    
    % LAGRANGE POINTS VELOCITY
    figure(3),
    subplot(1, 2, 1)
    plot(k, mean(lagrangePointsVelocity(:, 2)), 'ko')
    xlabel('Lagrangian points')
    ylabel('Velocity')
    ylim([-1, 1])
    title([num2str(k) '/' num2str(nt)])
    hold on
    subplot(1, 2, 2)
    plot(k, mean(pointCloud(:, 2)), 'ko')
    xlabel('Lagrangian points')
    ylabel('Center')
    ylim([-1, 1])
    title([num2str(k) '/' num2str(nt)])
    hold on
    % ------------------------------------------------------------------- %
    
end
dlmwrite('fxt.txt', fxt);
dlmwrite('fyt.txt', fyt);
fprintf('\n')
toc