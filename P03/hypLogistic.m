function [ h ] = hypLogistic( Thetas, Xs )
    % Calculates hypothesis for logistic regression given thetas and
    % features.
    e = exp(1);
    h = 1./(1 + e.^(-Thetas * Xs'));
    h = h';
end

