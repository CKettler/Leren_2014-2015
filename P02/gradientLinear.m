function [ g ] = gradientLinear( Thetas, Xs, Ys )
% Calculates gradient for multivariate linear regression

    m = length(Xs);
    n = size(Xs);
    n = n(2);

    assert(length(Xs) == length(Ys), 'Size of data must be equal');
    assert(length(Thetas) == n, 'Size of thetas must be same of Xs');
    
    h = Thetas * Xs' - Ys';
    g = 1/m * (Xs' * h');
end

