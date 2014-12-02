function [ g ] = gradientLogistic( Thetas, Xs, Ys )
    % Calculates gradient for multivariate logistic regression

    m = length(Xs);
    n = size(Xs);
    n = n(2);

    assert(length(Thetas) == n, 'Nr. of parameters and variables must be equal');
    assert(length(Xs) == length(Ys), 'Size of data must be equal');
    
    h = hypLogistic(Thetas, Xs);
    g = 1/m * ((h - Ys)' * Xs)';
    
end

