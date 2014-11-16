function [ deltas2, deltas3 ] = deltaNN(Thetas1, Thetas2, Ys, a2, a3)
% Calculates delta's for a neural network

    deltas3 = a3 - Ys
    deltas2 = Thetas2' * deltas3 .* (a2 .* (1 - a2))

end

