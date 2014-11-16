function [ h ] = hypNN( Thetas, Xs )
    % Calculates hypothesis implemented for Q2
    
    h = 1./(1 + exp(-Thetas * Xs));
    
end

