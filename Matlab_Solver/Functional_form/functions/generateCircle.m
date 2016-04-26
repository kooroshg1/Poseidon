function generateCircle(xc, yc, r, ns)
theta = linspace(0, 2*pi, ns + 1)';
theta = theta(1:end-1);
xs = xc + r * cos(theta);
ys = yc + r * sin(theta);
coordinate = [xs, ys];
dlmwrite('pointCloud.txt', coordinate);