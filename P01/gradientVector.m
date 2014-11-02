function [ g ] = gradientVector( theta_0, theta_1, X, Y )
% Calculate gradient (vector multiplication based)

    m = length(X);
    
    % Sum of partial derivatives to get gradient
    g(1) = sum(theta_0 + theta_1 * X - Y);
    g(2) = sum(X .* (theta_0 + theta_1 * X - Y));

    % Divide by length to get average
    g = g / m;
    
end

