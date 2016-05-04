% clc;
% clear all;
% close all;
% format short g;
% % ======================================================================= %
function [U, V, P] = NSsolve(x, y, X, Y, alpha, beta, peru, Ru, Rut, perv, Rv, Rvt, perp, Rp, Rpt, initialize, Uinit, Vinit)
if nargin == 16
    fprintf('initializeing with zeros ...\n');
else
    fprintf('initializeing with your values ...\n');
end

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
try 
    pointCloud = dlmread('pointCloud.txt');
    xs = pointCloud(:, 1)';
    ys = pointCloud(:, 2)';
    ns = length(xs);
    solidDomainExists = true;
catch
    solidDomainExists = false;
end
% ----------------------------------------------------------------------- %

%% DEFINE DELTA FUNCTION (MATRIX) TO MAP RESULTS BETWEEN LAGRANGIAN AND EULERIAN DOMAINS
if solidDomainExists
    [Dux, Duy, Dvx, Dvy, Dpx, Dpy] = mappingFunction();
end
% ----------------------------------------------------------------------- %

%% VELOCITIES AT THE LAGRANGIAN POINTS
if solidDomainExists
    lagrangePointsVelocity = dlmread('lagrangePointsVelocity.txt');
    uVelocity = lagrangePointsVelocity(:, 1);
    vVelocity = lagrangePointsVelocity(:, 2);
end
% ----------------------------------------------------------------------- %

%% SAVE INITIAL LOCATION AND VELOCITY OF LAGRANGIAN POINTS
if solidDomainExists
    pointCloudInitLoc = pointCloud;
    lagrangePointsInitVelocity = lagrangePointsVelocity;
end
% ----------------------------------------------------------------------- %

%% GENERATE MESH
[xu, yu, Xu, Yu] = generateMesh('U');
[xv, yv, Xv, Yv] = generateMesh('V');
[xp, yp, Xp, Yp] = generateMesh('P');
% ----------------------------------------------------------------------- %

%% STRUCTURAL MODEL
k = 1.0; m = 0.01; c = 0.0;
A = [0   , 1; ...
	 -k/m, -c/m];
Xt = zeros(2, nt + 1);
% ----------------------------------------------------------------------- %

%% OPEN TEXT FILE FOR WRITING SOLUTION HISTORY
if solidDomainExists
    forceDispHistFile = fopen('output/force_displacement_history.txt','w');
    fprintf(forceDispHistFile,'%-12s %-12s %-12s %-12s %-12s %-12s %-12s\n', 'Iter', 'F_x', 'F_y', '\bar{X}_c', '\bar{Y}_c', 'Y(t)', '\dot{Y}(t)');
end
% ----------------------------------------------------------------------- %

%% OPEN TEXT FILE FOR WRITING CONVERGENCE HISTORY CONVERGENCE HISTROY
convergenceHistFile = fopen('output/convergence_hist.txt','w');
fprintf(convergenceHistFile, '%-12s %-12s %-12s %-12s\n', 'Iter', 'U_{err}', 'V_{err}', 'CFL');
% ----------------------------------------------------------------------- %

%% INITIALIZE FORCE HISTORY
Fxhist = zeros(nt, 1); Fyhist = zeros(nt, 1);
% ----------------------------------------------------------------------- %

%% BEGIN SOLUTION
if initialize
    fxt = 0;       fyt = 0;
else
    fxt = dlmread('fxt.txt');       fyt = dlmread('fyt.txt');
end
fprintf('Solving ...\n--20%%--40%%--60%%--80%%-100%%\n')
tic
k = 0;
t = 0;
while t < tf
    %% SET TIME STEP
%     dt = CFL * min(Re/2 / (1/hx^2 + 1/hy^2), ...
%                    min(hx/max(max(U)), hy/max(max(V))));
	% ------------------------------------------------------------------- %
    
	%% CALCULATE CURRENT TIME
    k = k + 1;
    t = k * dt;
    % ------------------------------------------------------------------- %
    
    %% SAVE VARIABLES FOR CONVERGENCE STUDY
    Uold = U;
    Vold = V; 
    % ------------------------------------------------------------------- %
    
    %% APPLY BOUNDARY CONDITIONS
    [Ue, Ve, Ubc, Vbc, uW, uE, vS, vN] = assignBoundaryCondition(x, y, X, Y, U, V);
    % ------------------------------------------------------------------- %
    
    %% CALCULATE GAMMA FOR UPWIND FINITE DIFFERENCING
    gamma = min(1.2*dt*max(max(max(abs(U)))/hx,max(max(abs(V)))/hy),1);
    % ------------------------------------------------------------------- %

    %% INTERPOLATE U AND V AT THE LAGRANGIAN POINTS
    if solidDomainExists
        UX = diag(Dux * U * Duy') * hx * hy;
        VX = diag(Dvx * V * Dvy') * hx * hy;
    end
    % ------------------------------------------------------------------- %
    
    %% CALCULATE THE FORCE TERMS
    if solidDomainExists
        fxt = fxt + (UX - uVelocity) * dt; fx = alpha * fxt + beta * (UX - uVelocity); Fx = Dux' * diag(fx) * Duy * hx * hy;
        fyt = fyt + (VX - vVelocity) * dt; fy = alpha * fyt + beta * (VX - vVelocity); Fy = Dvx' * diag(fy) * Dvy * hx * hy;
    end
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
    
    if solidDomainExists
        U = U-dt*(UVy(2:end-1,:)+U2x) + dt * Fx;
        V = V-dt*(UVx(:,2:end-1)+V2y) + dt * Fy;
    else
        U = U-dt*(UVy(2:end-1,:)+U2x);
        V = V-dt*(UVx(:,2:end-1)+V2y);
    end
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
    if solidDomainExists
        [Fx, Fy] = calcSurfaceForce(P, Dpx, Dpy, xs, ys);
        Fxhist(k) = Fx; Fyhist(k) = Fy;
    end
    
    %% SOLVE DYNAMIC EQUATION OF MOTION
%     if solidDomainExists
%         %% BACKWARD EULER TIME INTEGRATION
% %         Xt(:, k) = dt * (A * Xt(:, k - 1) + [0; Fyhist(k)] / m) + Xt(:, k - 1);
%         
%         %% ADAMS-BASHFORTH TIME INTEGRATION
%         if k == 1
%             Xt(:, k + 1) = (A * Xt(:, k) + [0; Fyhist(k)] / m) * dt + Xt(:, k);
%         else
%             Xt(:, k + 1) = Xt(:, k) + ...
%                                  dt * (3/2 * A * Xt(:, k) + 3/2 * [0; Fyhist(k)] / m + ...
%                                          -1/2 * A * Xt(:, k) - 1/2 * [0;Fyhist(k - 1)] / m);
%         end
%         
%         pointCloud(:, 2) = pointCloudInitLoc(:, 2) + Xt(1, k + 1);
%         lagrangePointsVelocity(:, 2) = lagrangePointsInitVelocity(:, 2) + Xt(2, k + 1);
%         
%         dlmwrite('pointCloud.txt', pointCloud);
%         dlmwrite('lagrangePointsVelocity.txt', lagrangePointsVelocity);
%         [Dux, Duy, Dvx, Dvy, Dpx, Dpy] = mappingFunction();
%         
%         xs = pointCloud(:, 1)';
%         ys = pointCloud(:, 2)';
%         
%         uVelocity = lagrangePointsVelocity(:, 1);
%         vVelocity = lagrangePointsVelocity(:, 2);
%     end
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
    
    %% WRITE DATA TO TEXT FILE
    if solidDomainExists
        fprintf(forceDispHistFile,'%-12i %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f\n', k, Fx, Fy, mean(pointCloud(:, 1)), mean(pointCloud(:, 2)), Xt(1, k), Xt(2, k));
    end
    % ------------------------------------------------------------------- %
    
    %% CALCULATE CFL NUMBER
    CFL = max(max(max(U)), max(max(V))) * dt / max(hx, hy);
    
    %% WRITE CONVERGENCE HISTORY RESULTS
    fprintf(convergenceHistFile, '%-12i %-12.4f %-12.9f %-12.4f\n', k, Uerr, Verr, CFL);
    % ------------------------------------------------------------------- %
    
    %% PLOT
    if solidDomainExists
        % VELOCITY CONTOUR
        figure(1),
        fileName = ['figures/', num2str(k, '%010d'), '.png'];
        contourf(Xu, Yu, U, 50, 'linestyle', 'none')
        hold('on')
        fill(xs, ys, 'w')
        hold('off')
        axis('equal')
        title([num2str(k) '/' num2str(nt)])
        colorbar
%         saveas(gcf, fileName)

        % F_x AND F_y ON THE BOUNDARY
%         figure(2),
%         subplot(2, 2, 1)
%         plot(k, Fx, 'ko')
%         xlabel('# of iterations')
%         ylabel('Fx')
%         title([num2str(k) '/' num2str(nt)])
%         hold on
%         subplot(2, 2, 2)
%         plot(k, Fy, 'ko')
%         xlabel('# of iterations')
%         ylabel('Fy')
%         title([num2str(k) '/' num2str(nt)])
%         hold on    
%         % LAGRANGE POINTS VELOCITY
%         subplot(2, 2, 3)
%         plot(k, mean(lagrangePointsVelocity(:, 2)), 'ko')
%         xlabel('# of iterations')
%         ylabel('Velocity')
%         title([num2str(k) '/' num2str(nt)])
%         hold on
%         subplot(2, 2, 4)
%         plot(k, mean(pointCloud(:, 2)), 'ko')
%         xlabel('# of iterations')
%         ylabel('Center')
%         title([num2str(k) '/' num2str(nt)])
%         hold on
    else
%         figure(1),
%         fileName = ['figures/', num2str(k, '%010d'), '.png'];
%         contourf(Xu, Yu, U, 50, 'linestyle', 'none')
%         hold('off')
%         axis('equal')
%         title([num2str(k) '/' num2str(nt)])
%         colorbar
%         saveas(gcf, fileName)
    end
    % ------------------------------------------------------------------- %    
end
fprintf('\n')
% ----------------------------------------------------------------------- %

%% WRITE FILES IF SOLUTION NEEDS TO BE CONTINUED
dlmwrite('fxt.txt', fxt);
dlmwrite('fyt.txt', fyt);
dlmwrite('U.txt', U);
dlmwrite('V.txt', V);
% ----------------------------------------------------------------------- %

%% CLOSE OPENED FILES
if solidDomainExists
    fclose(forceDispHistFile);
end
fclose(convergenceHistFile);
toc
