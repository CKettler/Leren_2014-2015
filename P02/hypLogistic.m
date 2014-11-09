function [ h ] = hypLogistic( Thetas, Xs )

    e = exp(1);
    h = 1./(1 + e.^(-Thetas * Xs'));
    h = h';
end

