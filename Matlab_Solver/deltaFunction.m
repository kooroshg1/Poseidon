function delta = deltaFunction(xs, ys, X, Y, eta)
delta = 0;
for i = 1:length(xs)
    deltaX = (1 / eta) .* (-tanh((X - xs(i)) ./ eta).^2 + 1) / 2;
    deltaY = (1 / eta) .* (-tanh((Y - ys(i)) ./ eta).^2 + 1) / 2;
    delta = delta + deltaX .* deltaY;
end