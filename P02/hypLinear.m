function [ h ] = hypLinear( Thetas, Xs )
    % Calculates hypothesis for linear regression given thetas and
    % features.
    h = Xs * Thetas';
    h = h';
end

