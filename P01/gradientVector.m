function [ g ] = gradientVector( theta_0, theta_1, x, y )
% Calculate gradient (vector)

    m = length(x);

    g(0) =   sum(theta_0 + theta_1 * x - y);
    g(1) = sum(x .* (theta_0 + theta_1 * x - y));

end

