clear all;
close all;
format short g;
% clc;
% ============================== %
fid = fopen('Debug/operator.output');

tline = fgetl(fid);
tline = fgetl(fid);
nx = 5;
ny = 3;

% For U and V
% nUx = nx + 1; nVx = nx + 2;
% nUy = ny + 2; nVy = ny + 1;
% for Us and Vs
nUx = nx - 1; nVx = nx;
nUy = ny; nVy = ny - 1;

U = zeros(nUx * nUy, 1);
V = zeros(nVx * nVy, 1);
readU = false; readV = true;
% readU = true; readV = false;

rowNumber = 1;

if readU
while ~feof(fid)
    tline = fgetl(fid);
    tline = sscanf(tline, '%f');
    U(rowNumber) = tline;
    rowNumber = rowNumber + 1;
end
U = reshape(U, nUx, nUy)
end

if readV
while ~feof(fid)
    tline = fgetl(fid);
    tline = sscanf(tline, '%f');
    V(rowNumber) = tline;
    rowNumber = rowNumber + 1;
end
V = reshape(V, nVx, nVy)
end
fclose(fid);