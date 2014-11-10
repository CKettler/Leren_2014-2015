function [ costs ] = costLinear( Hs, Ys )
    % Calculates the cost (J) of linear regression given the hypothesises
    % and the target data set Y.
    m = length(Ys);
    costs = 1/(2*m)*sum((Hs - Ys).^2);
end

