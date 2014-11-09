function [ Thetas2 ] = updateParamLogistic( Thetas, alpha, Xs, Ys )

    Thetas2 = (Thetas' - alpha * gradientLogistic(Thetas, Xs, Ys))';

end

