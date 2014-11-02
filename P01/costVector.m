function [ cost ] = costVector( theta_0, theta_1, X, Y )
% calculates cost for given parameters and variables

    m = length(X);

    % Calc cost based on J(theta_0, theta_1)
    cost = (1 / (2 * m)) * sum((theta_0 + theta_1 * X - Y).^2);

end

