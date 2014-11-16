function [ cost ] = costLogRegularized( Thetas, Xs, Ys, lambda )
    % Calculates the cost of logistic regression given Thetas, features and
    % Y. 
    Hs = hypLogistic(Thetas, Xs);

    m = length(Ys);
    cost = (-1/m) * sum((Ys .* log(Hs) + (1 - Ys) .* log(1 - Hs))) + ...
        (lambda/(2 * m)) * sum(Thetas(2:end).^2);
    
end

