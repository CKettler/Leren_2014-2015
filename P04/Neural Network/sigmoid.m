function [ s ] = sigmoid( Zs )
    % calculates sigmoid on matrix of values
    
    s = 1./(1 + exp(-Zs));

end

