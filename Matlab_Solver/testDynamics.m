clc;
clear all;
close all;
format short g;
% ----------------------------------------------------------------------- %
nt = 10000;
dt = 0.01;
k = 1.0; m = 1.0; c = 0.0;
A = [0   , 1; ...
	 -k/m, -c/m];
x0 = 0; v0 = 0;
Xt = zeros(2, nt); Xt(1,1) = x0; Xt(2,1) = v0;
f = @(t) sin(2 * t);

%% FIRST ORDER TEMPORAL
for it = 1:nt-1
    t = it * dt;
    Xt(:, it + 1) = (A * Xt(:, it) + [0; f(t)] / m) * dt + Xt(:, it);
end

Xt_BE = Xt;
Xt = zeros(2, nt); Xt(1,1) = x0; Xt(2,1) = v0;
%% SECOND ORDER TEMPORAL
for it = 1:nt-1
    t = it * dt;
    if it == 1
        Xt(:, it + 1) = (A * Xt(:, it) + [0; f(t)] / m) * dt + Xt(:, it);
    else
        Xt(:, it+ 1) = Xt(:, it) + ...
                             dt * (3/2 * A * Xt(:, it) + 3/2 * [0; f(t)] / m + ...
                                     -1/2 * A * Xt(:, it - 1) - 1/2 * [0;f(t-dt)] / m);
    end
end
Xt_AB = Xt;

T = 0:dt:((nt - 1)*dt);
figure,
subplot(1,2,1)
plot(T, Xt_BE(1, :), 'k', ...
       T, 2/3 * sin(T) - 1/3 * f(T), 'r--')
xlabel('Time')
ylabel('Displacement')
title('Backward Euler')
subplot(1,2,2)
plot(T, Xt_AB(1, :), 'k', ...
       T, 2/3 * sin(T) - 1/3 * f(T), 'r--')
xlabel('Time')
ylabel('Displacement')
title('Adams–Bashforth')
