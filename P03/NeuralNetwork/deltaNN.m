function [Deltas2, Deltas3] = deltaNN(Thetas2, Ys, a2, a3)
% Calculates delta's for a neural network
    
    % Calculate deltas
    Deltas3 = a3 - Ys';
    
    Deltas2 = Thetas2' * Deltas3 .* (a2 .* (1 - a2));
    Deltas2 = Deltas2(2:end,:);
    
end