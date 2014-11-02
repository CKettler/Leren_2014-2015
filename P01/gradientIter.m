function [ g ] = gradientIter( theta_0, theta_1, X, Y )
% Calculate gradient (iteration based)

    m = length(X);

    % Sum of partial derivatives to get gradient
    for i = 1:m
        g(1) = g(1) + (theta_0 + theta_1 * X(i) - Y(i));
        g(2) = g(2) + (X(i) * (theta_0 + theta_1 * X(i) - Y(i)));
    end % end for i
    
    % Divide by length to get average
    g = g / m;

end

