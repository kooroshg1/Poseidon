% clc
% clear all;
% close all;
% format short g;
% % ======================================================================= %
% Ain = 1.0;
% Athroat = 0.5;
% Aout = 1.5;
% xStart = 0.0;
% yStart = 0.0;
% xThroat = 0.3;
% xEnd = 1.0;
% yMax = 1.0;
% n = 10;
% dx = (xEnd - xStart) / (n - 1);
function generateNozzle(Ain, Athroat, Aout, xStart, yStart, xThroat, xEnd, yMax, n)

xTopStart = xStart;        yTopStart = yStart + Ain / 2;
xTopThroat = xThroat;      yTopThroat = yStart + Athroat / 2;
xTopEnd = xEnd;            yTopEnd = yStart + Aout / 2;

coefficients = [xTopStart^2, xTopStart, 1; ...
                xTopThroat^2, xTopThroat, 1; ...
                2*xTopThroat, 1, 0] \ ...
               [yTopStart; yTopThroat; 0];
           
aFront = coefficients(1); bFront = coefficients(2); cFront = coefficients(3);

coefficients = [xTopThroat^2, xTopThroat, 1; ...
                xTopEnd^2, xTopEnd, 1; ...
                2*xTopThroat, 1, 0] \ ...
               [yTopThroat; yTopEnd; 0];
           
aBack = coefficients(1); bBack = coefficients(2); cBack = coefficients(3);

xFrontSymmetry = linspace(xStart, xThroat, 20);
xBackSymmetry = linspace(xThroat, xEnd, 20);

% xFrontSymmetry = xStart:dx:xThroat;
% xBackSymmetry = xThroat:dx:xEnd;

yFrontTop = aFront * xFrontSymmetry.^2 + bFront * xFrontSymmetry + cFront;
yBackTop = aBack * xBackSymmetry.^2 + bBack * xBackSymmetry + cBack;

yFrontBottom = yStart - yFrontTop;
yBackBottom = yStart - yBackTop;

% ======================================================================= %
xFrontWall = xStart * ones(1, n);      yFrontWall = linspace(yTopStart, yMax, n);
xTopWall = linspace(xStart, xEnd, n);  yTopWall = yMax * ones(1, n);
xBackWall = xEnd * ones(1, n);         yBackWall = linspace(yMax, yTopEnd, n);

% xFrontWall = [xFrontWall, xFrontWall]; yFrontWall = [yFrontWall, yStart - yFrontWall];
% xTopWall = [xTopWall, xTopWall];       yTopWall = [yTopWall, yStart - yTopWall];
% xBackWall = [xBackWall, xBackWall];    yBackWall = [yBackWall, yStart - yBackWall];

% figure,
% plot(xFrontSymmetry, yFrontTop, 'ko',...
%      xBackSymmetry, yBackTop, 'ko', ...
%      xFrontSymmetry, yFrontBottom, 'ko', ...
%      xBackSymmetry, yBackBottom, 'ko', ...
%      xFrontWall, yFrontWall, 'ko', ...
%      xTopWall, yTopWall, 'ko', ...
%      xBackWall, yBackWall, 'ko')
% axis('equal')

coordinate = [xFrontWall', yFrontWall'; ...
              xTopWall', yTopWall'; ...
              xBackWall', yBackWall'; ...
              xFrontSymmetry', yFrontTop'; ...
              xBackSymmetry', yBackTop'];

coordinate = [coordinate; ...
              coordinate(:, 1), yStart - coordinate(:, 2)];
% figure,
% plot(coordinate(:, 1), coordinate(:, 2), 'ko')
% waitforbuttonpress
% axis('equal')

dlmwrite('pointCloud.txt', coordinate);