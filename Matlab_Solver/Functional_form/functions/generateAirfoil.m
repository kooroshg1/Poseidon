function generateAirfoil(type)
myDir = pwd;
if strcmp(type(1:4), 'naca')
    coordinate = dlmread([myDir, '/functions/airfoil_library/', type, '.dat']);
    dlmwrite('pointCloud.txt', coordinate);
end