function [ costs ] = costLinear( Hs, Ys )
    % Hs = hypothesises
    % Ys = Ys
    m = length(Ys);
    costs = 1/(2*m)*sum((Hs - Ys).^2);
end

