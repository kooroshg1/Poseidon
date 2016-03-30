% ----------------------------------------------------------------------- %
close all;
clear all;
format short g;
clc;
% ----------------------------------------------------------------------- %
%% DEFINE PHYSICAL PROPERTIES AND DOMAIN DIMENSION
Re = 1e2;     % Reynolds number
dt = 1e-2;    % time step
tf = 4.0;    % final time
lx = 1;       % width of box
ly = 1;       % height of box
nx = 100;      % number of x-gridpoints
ny = 100;      % number of y-gridpoints
nsteps = 10;  % number of steps with graphic output
% ----------------------------------------------------------------------- %

%% SPATIAL DISCRETIZATION
nt = ceil(tf/dt); dt = tf/nt;
x = linspace(0,lx,nx+1); hx = lx/nx; %hx = 1;
y = linspace(0,ly,ny+1); hy = ly/ny; %hy = 1;
[Y,X] = meshgrid(y,x);
% ----------------------------------------------------------------------- %

%% INITIALIZE VELOCITY
U = zeros(nx-1,ny) + eps; V = zeros(nx,ny-1) + eps;
% ----------------------------------------------------------------------- %

%% ASSIGN BOUNDARY CONDITIONS
% Lid-driven cavity boundary condition
uN = x*0+0;    vN = avg(x)*0+0;
uS = x*0+0;      vS = avg(x)*0+0;
uW = avg(y)*0+1; vW = y*0+0;
uE = avg(y)*0+0; vE = y*0+0;

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
  
Ubc = dt/Re*([2*uS(2:end-1)' zeros(nx-1,ny-2) 2*uN(2:end-1)']/hy^2+...
      [uW;zeros(nx-3,ny);uE]/hx^2);
Vbc = dt/Re*([vS' zeros(nx,ny-3) vN']/hy^2+...
      [2*vW(2:end-1);zeros(nx-2,ny-1);2*vE(2:end-1)]/hx^2);
% ----------------------------------------------------------------------- %

%% INITIALIZE DIFFERENTIAL OPERATORS
fprintf('initialization\n')
Lp = kron(speye(ny),K1(nx,hx,1))+kron(K1(ny,hy,1),speye(nx));
Lp(1,1) = 3/2*Lp(1,1);
perp = symamd(Lp); Rp = chol(Lp(perp,perp)); Rpt = Rp';
Lu = speye((nx-1)*ny)+dt/Re*(kron(speye(ny),K1(nx-1,hx,2))+...
     kron(K1(ny,hy,3),speye(nx-1)));
peru = symamd(Lu); Ru = chol(Lu(peru,peru)); Rut = Ru';
Lv = speye(nx*(ny-1))+dt/Re*(kron(speye(ny-1),K1(nx,hx,3))+...
     kron(K1(ny-1,hy,2),speye(nx)));
perv = symamd(Lv); Rv = chol(Lv(perv,perv)); Rvt = Rv';
Lq = kron(speye(ny-1),K1(nx-1,hx,2))+kron(K1(ny-1,hy,2),speye(nx-1));
perq = symamd(Lq); Rq = chol(Lq(perq,perq)); Rqt = Rq';
% ----------------------------------------------------------------------- %

%% DEFINE LAGRANGIAN POINTS
ns = 10;
theta = linspace(0, 2*pi, ns);
xs = 0.5 + 0.1 * cos(theta);
ys = 0.5 + 0.1 * sin(theta);
% ----------------------------------------------------------------------- %

%% DEFINE X AND Y MATRICES
xu = linspace(0, lx, nx - 1);
yu = linspace(0, ly, ny);
[Yu, Xu] = meshgrid(yu, xu);

xv = linspace(0, lx, nx);
yv = linspace(0, ly, ny - 1);
[Yv, Xv] = meshgrid(yv, xv);

eta = hx / atanh(sqrt(1 - 0.9));
Dux = zeros(ns, nx - 1); Duy = zeros(ns, ny);
Dvx = zeros(ns, nx); Dvy = zeros(ns, ny - 1);
for i=1:ns
    [deltaX, deltaY] = deltaFunction(xs(i), ys(i), xu, yu, eta);
    Dux(i, :) = deltaX; Duy(i, :) = deltaY;
    [deltaX, deltaY] = deltaFunction(xs(i), ys(i), xv, yv, eta);
    Dvx(i, :) = deltaX; Dvy(i, :) = deltaY;
end

% figure,
% plot(diag(Dux * Xu * Duy') * hx * hy, diag(Dux * Yu * Duy') * hx * hy)
% axis('equal')

% figure,
% surf(Xu, Yu, Du)
% waitforbuttonpress
% ----------------------------------------------------------------------- %

%% BEGIN SOLUTION
alpha = -1000;
beta = -1;
fxt = zeros(ns, 1);
fyt = zeros(ns, 1);
fprintf(', time loop\n--20%%--40%%--60%%--80%%-100%%\n')
for k = 1:nt
    Fx = 0;
    Fy = 0;
    %% APPLY BOUNDARY CONDITIONS
    % Free-slip boundary condition on top and bottom walls
    uN(2:end-1) = U(:,end);
    uS(2:end-1) = U(:,1);

    % Zero gradient for velocity on east wall
    uE = (mean(uW) ./ mean(U(end,:))) .* U(end,:);
    vE(2:end-1) = (mean(vW) ./ mean(V(end,:))) .* V(end,:);

    Ubc = dt/Re*([2*uS(2:end-1)' zeros(nx-1,ny-2) 2*uN(2:end-1)']/hy^2+...
      [uW;zeros(nx-3,ny);uE]/hx^2);
    Vbc = dt/Re*([vS' zeros(nx,ny-3) vN']/hy^2+...
         [2*vW(2:end-1);zeros(nx-2,ny-1);2*vE(2:end-1)]/hx^2);
    
    gamma = min(1.2*dt*max(max(max(abs(U)))/hx,max(max(abs(V)))/hy),1);
    Ue = [uW;U;uE]; Ue = [2*uS'-Ue(:,1) Ue 2*uN'-Ue(:,end)];
    Ve = [vS' V vN']; Ve = [2*vW-Ve(1,:);Ve;2*vE-Ve(end,:)];
    
    %% INTERPOLATE U AND V AT THE LAGRANGIAN POINTS
    UX = diag(Dux * U * Duy') * hx * hy;
    VX = diag(Dvx * V * Dvy') * hx * hy;

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

    %% IMPLICIT VISCOSITY
    rhs = reshape(U + Ubc,[],1);
    u(peru) = Ru\(Rut\rhs(peru));
    U = reshape(u,nx-1,ny);

    rhs = reshape(V + Vbc,[],1);
    v(perv) = Rv\(Rvt\rhs(perv));
    V = reshape(v,nx,ny-1);

    %% PRESSURE CORRECTION
    rhs = reshape(diff([uW;U;uE])/hx+diff([vS' V vN']')'/hy,[],1) / dt;
    p(perp) = -Rp\(Rpt\rhs(perp));
    P = reshape(p,nx,ny);


    U = U-diff(P)/hx * dt;
    V = V-diff(P')'/hy * dt;   

    %% TIMER
    if floor(25*k/nt)>floor(25*(k-1)/nt), fprintf('.'), end
    
    %% CHECK FOR NAN
    if max(max(isnan(U))) || max(max(isnan(V)))
        break
    end
end
fprintf('\n')
Ue = [uW;U;uE]; Ue = [2*uS'-Ue(:,1) Ue 2*uN'-Ue(:,end)];
Ve = [vS' V vN']; Ve = [2*vW-Ve(1,:);Ve;2*vE-Ve(end,:)];
% ======================================================================= %
% xu = linspace(0, lx, nx + 1);
% yu = linspace(0, ly, ny + 2);
% [Yu, Xu] = meshgrid(yu, xu);
% 
% xv = linspace(0, lx, nx + 2);
% yv = linspace(0, ly, ny + 1);
% [Yv, Xv] = meshgrid(yv, xv);
% 
% xp = linspace(0, lx, nx);
% yp = linspace(0, ly, ny);
% [Yp, Xp] = meshgrid(yp, xp);
% 
% 
% figure,
% contourf(Xu, Yu, Ue, 50, 'linestyle', 'none')
% 
% % figure,
% % contourf(Xp, Yp, P, 50, 'linestyle', 'none')