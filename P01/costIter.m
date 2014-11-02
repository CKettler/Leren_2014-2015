function [ cost ] = costIter( theta_0, theta_1, X, Y )
% calculates cost for given parameters and variables

    m = length(X);

    % Calc cost based on J(theta_0, theta_1)
    cost = 0;
    for i = 1:m
        cost = cost + (theta_0 + theta_1 * X(i) - Y(i))^2;
    end % end for i
    
    cost = cost / (2 * m);
    
end
