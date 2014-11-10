function [ Thetas2 ] = updateParamLinear( Thetas, alpha, Xs, Ys )
    % updates thetas
    Thetas2 = (Thetas' - alpha * gradientLinear(Thetas, Xs, Ys))';

end

