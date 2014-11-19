function [Deltas2, Deltas3] = deltaNN(Thetas2, Ys, a2, a3)
% Calculates delta's for a neural network

    % Drop the first column vector since this is for the bias node which
    % doesn't have any edges going back to the previous layer
    a2 = a2(:,2:end);
    
    % Calculate delta's
    Deltas3 = a3 - Ys;
    Deltas2 = Thetas2' * Deltas3 .* (a2 .* (1 - a2));

    
end