function generateFin(xs, ys, xe, ye, ns)
x = linspace(xs, xe, ns)';
y = linspace(ys, ye, ns)';
coordinate = [x, y];
dlmwrite('pointCloud.txt', coordinate);