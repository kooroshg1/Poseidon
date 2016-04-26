clc;
clear all;
close all;
format short g;
% ----------------------------------------------------------------------- %
case_number = 2;
% inf: Re, dt, tf, xStart, xEnd, yStart, yEnd, nx, ny, convCriteria
% hist: Iter, F_x, F_y, \bar{X}_c, \bar{Y}_c, Y(t), \dot{Y}(t)  
infFileName = ['results/', num2str(case_number), '/information.txt'];
inf = dlmread(infFileName, '', 1, 0);
forceFileName = ['results/', num2str(case_number), '/force_displacement_history.txt'];
force = dlmread(forceFileName, '', 1, 0);
% convHistFileName = ['results/', num2str(case_number), '/convergence_hist.txt'];
% convHist = dlmread(convHistFileName, '', 1, 0);

dt = inf(2) * 0.5;

figure,
plot(force(:, 1) * dt, force(:, 2), 'k')
title(['Re = ', num2str(inf(1))])
xlabel('time')
ylabel('F_x')
xlim([5,25])

figure,
plot(force(:, 1) * dt, force(:, 3), 'k')
title(['Re = ', num2str(inf(1))])
xlabel('time')
ylabel('F_y')