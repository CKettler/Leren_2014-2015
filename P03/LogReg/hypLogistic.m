function [ h ] = hypLogistic( Thetas, Xs )
    % Calculates hypothesis for logistic regression given thetas and
    % features.
    h = 1./(1 + exp(-Thetas * Xs'))';
    
end

