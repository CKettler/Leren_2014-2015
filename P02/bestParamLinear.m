function [ NewThetas ] = bestParamLinear( Thetas, alpha, iterations, Xs, Ys )
    % returns best thetas for linear regression. Stops after 'iterations'.

    NewThetas = Thetas;
    for i=1:iterations
        NewThetas = updateParamLinear(NewThetas, alpha, Xs, Ys);
    end

end

