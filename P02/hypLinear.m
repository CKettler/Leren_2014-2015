function [ h ] = hypLinear( Thetas, Xs )
    h = Xs * Thetas';
    h = h';
end

