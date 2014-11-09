function [ NewThetas ] = bestParamLinear( Thetas, alpha, iterations, Xs, Ys )

    NewThetas = Thetas;
    for i=1:iterations
        NewThetas = updateParamLinear(NewThetas, alpha, Xs, Ys);
    end

end

