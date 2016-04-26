% ----------------------------------------------------------------------- %
close all;
clear all;
format short g;
clc;
% ----------------------------------------------------------------------- %
%% ADDS THE DIRECTORY FOR REQUIRED FUNCTIONS TO THE PATH
addpath('functions/');
% ----------------------------------------------------------------------- %

%% DEFINE PHYSICAL PROPERTIES AND DOMAIN DIMENSION
Re = 1e2;               % Reynolds number
dt = 1e-2;              % time step
tf = 5.0;               % final time
xStart = -0.5;          % Domain begining coordinate (x)
xEnd = 2.5;             % Domain end coordinate (x)
yStart = -0.5;          % Domain begining coordinate (y)
yEnd = 0.5;             % Domain end coordinate (y)
nx = 300;               % number of x-gridpoints
ny = 100;               % number of y-gridpoints
convCriteria = 1e-3;    % Convergence criteria
% ----------------------------------------------------------------------- %

%% DEFINE LAGRANGIAN POINTS
generateCircle(0.0, 0.0, 0.1, 50);
% generateSquare(-0.1, -0.1, 0.1, 0.1, 50);
% generateNozzle(0.3, 0.1, 0.4, 0.0, 0.0, 0.5, 1.5, 0.5, 100)
% generateAirfoil('naca001234'); 
% generateAirfoil('naca2408'); 

pointCloud = dlmread('pointCloud.txt');
xs = pointCloud(:, 1)';
ys = pointCloud(:, 2)';
delete('pointCloud.txt');
ns = length(xs);
% ----------------------------------------------------------------------- %

%% SPATIAL DISCRETIZATION
nt = ceil(tf/dt); dt = tf/nt;
x = linspace(xStart, xEnd, nx + 1); hx = (xEnd - xStart) / nx;
y = linspace(yStart, yEnd, ny + 1); hy = (yEnd - yStart) / ny;
[Y,X] = meshgrid(y,x);
% ----------------------------------------------------------------------- %

%% INITIALIZE VELOCITY
U = zeros(nx - 1, ny) + eps; V = zeros(nx, ny - 1) + eps;
% ----------------------------------------------------------------------- %

%% ASSIGN BOUNDARY CONDITIONS
% Lid-driven cavity boundary condition
uN = x*0+0;       vN = avg(x)*0+0;
uS = x*0+0;       vS = avg(x)*0+0;
uW = avg(y)*0+1;  vW = y*0+0;
uE = avg(y)*0+0;  vE = y*0+0;

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
% ----------------------------------------------------------------------- %

%% INITIALIZE DIFFERENTIAL OPERATORS
fprintf('initialization\n')
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

%% DEFINE X AND Y MATRICES
xu = linspace(xStart, xEnd, nx - 1);
yu = linspace(yStart, yEnd, ny);
[Yu, Xu] = meshgrid(yu, xu);

xv = linspace(xStart, xEnd, nx);
yv = linspace(yStart, yEnd, ny - 1);
[Yv, Xv] = meshgrid(yv, xv);

xp = linspace(xStart, xEnd, nx);
yp = linspace(yStart, yEnd, ny);
[Yp, Xp] = meshgrid(yp, xp);

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

% figure,
% plot(diag(Dux * Xu * Duy') * hx * hy, diag(Dux * Yu * Duy') * hx * hy)
% axis('equal')

% figure,
% surf(Xu, Yu, Du)
% waitforbuttonpress
% ----------------------------------------------------------------------- %

%% BEGIN SOLUTION
alpha = -10000;
beta = -100;
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
    % Free-slip boundary condition on top and bottom walls
    uN(2:end-1) = U(:,end);
    uS(2:end-1) = U(:,1);

    % Zero gradient for velocity on east wall
    uE = (mean(uW) ./ mean(U(end,:))) .* U(end,:);
    vE(2:end-1) = (mean(vW) ./ mean(V(end,:))) .* V(end,:);

%     Ubc = dt/Re*([2*uS(2:end-1)' zeros(nx-1,ny-2) 2*uN(2:end-1)']/hx^2+...
%           [uW;zeros(nx-3,ny);uE]/hy^2);
%     Vbc = dt/Re*([vS' zeros(nx,ny-3) vN']/hx^2+...
%           [2*vW(2:end-1);zeros(nx-2,ny-1);2*vE(2:end-1)]/hy^2);

    Ubc = dt/Re*([2*uS(2:end-1)' zeros(nx-1,ny-2) 2*uN(2:end-1)']/hy^2+...
          [uW;zeros(nx-3,ny);uE]/hx^2);
    Vbc = dt/Re*([vS' zeros(nx,ny-3) vN']/hy^2+...
          [2*vW(2:end-1);zeros(nx-2,ny-1);2*vE(2:end-1)]/hx^2);
    
    gamma = min(1.2*dt*max(max(max(abs(U)))/hx,max(max(abs(V)))/hy),1);
    Ue = [uW;U;uE]; Ue = [2*uS'-Ue(:,1) Ue 2*uN'-Ue(:,end)];
    Ve = [vS' V vN']; Ve = [2*vW-Ve(1,:);Ve;2*vE-Ve(end,:)];
    % ------------------------------------------------------------------- %
    
    %% INTERPOLATE U AND V AT THE LAGRANGIAN POINTS
    UX = diag(Dux * U * Duy') * hx * hy;
    VX = diag(Dvx * V * Dvy') * hx * hy;
    % ------------------------------------------------------------------- %
    
    %% CALCULATE THE FORCE TERMS
    fxt = fxt + (UX - 0) * dt; fx = alpha * fxt + beta * (UX - 0); Fx = Dux' * diag(fx) * Duy * hx * hy;
    fyt = fyt + (VX - 0) * dt; fy = alpha * fyt + beta * (VX - 0); Fy = Dvx' * diag(fy) * Dvy * hx * hy;
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
%% ADDING BOUNDARY CONDITIONS TO THE VELOCITIES
Ue = [uW;U;uE]; Ue = [2*uS'-Ue(:,1) Ue 2*uN'-Ue(:,end)];
Ve = [vS' V vN']; Ve = [2*vW-Ve(1,:);Ve;2*vE-Ve(end,:)];
% ======================================================================= %

%% CALCULATING PRESSURE OVER THE BOUNDARY
PX = diag(Dpx * P * Dpy') * hx * hy;
% ======================================================================= %

%% PLOTTING VELOCITY AND PRESSURE CONTOURS
xu = linspace(xStart, xEnd, nx + 1);
yu = linspace(yStart, yEnd, ny + 2);
[Yu, Xu] = meshgrid(yu, xu);
% [uin, uon] = inpolygon(Xu, Yu, xs, ys);
% Ue(uin) = nan;

xv = linspace(xStart, xEnd, nx + 2);
yv = linspace(yStart, yEnd, ny + 1);
[Yv, Xv] = meshgrid(yv, xv);
% [vin, von] = inpolygon(Xv, Yv, xs, ys);
% Ve(vin) = nan;

xp = linspace(xStart, xEnd, nx);
yp = linspace(yStart, yEnd, ny);
[Yp, Xp] = meshgrid(yp, xp);
% [pin, pon] = inpolygon(Xp, Yp, xs, ys);
% P(pin) = nan;

figure,
subplot(1,2,1)
plot(xu, Ue(:, (ny + 2) / 2))
subplot(1,2,2)
plot(UX)

figure,
plot(PX)
ylabel('pressure')

figure,
contourf(Xu, Yu, Ue, 50, 'linestyle', 'none')
axis('equal')

figure,
contourf(Xp, Yp, P, 50, 'linestyle', 'none')
axis('equal')
% ======================================================================= %

% ds = sqrt((xs(2) - xs(1))^2 + (ys(2) - ys(1))^2);
% [PX, sqrt(fx.^2 + fy.^2)]
% % [sqrt(fx.^2 + fy.^2) ./ PX]