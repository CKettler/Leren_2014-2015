function [ Thetas, Ynew ] = compareClasses(Thetas, alpha, Xs, Ys, iter, y, lambda)
    % Compares one class versus the rest (OvA algorithm)
    % Replace wanted Y with 1, rest with 0
    Ynew = Ys;
    Ynew ( Ys ~= y ) = 0;
    Ynew ( Ynew == y ) = 1;
    cost = [1, iter];
        
    % Update thetas and keep track of cost for plot
    for i = 1:iter
       Thetas = updateLogRegularized(Thetas, alpha, Xs, Ynew, lambda); 
       cost(i) = costLogRegularized(Thetas, Xs, Ynew, lambda);
    end
    
    plot(1:iter, cost);
    
end