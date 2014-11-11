function [ costs ] = costLogistic( Thetas, Xs, Ys )
    % Calculates the cost of logistic regression given Thetas, features and
    % Y. 
    Hs = hypLogistic(Thetas, Xs);

    m = length(Ys);
    costs = -1/m*sum((Ys .* log(Hs) + (1 - Ys) .* log(1 - Hs)));
    
end

