function [ Thetas2 ] = updateParamLinear( Thetas, alpha, Xs, Ys )

    Thetas2 = (Thetas' - alpha * gradientLinear(Thetas, Xs, Ys))';

end

