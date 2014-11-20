function [cost] = costNN(Xs, Thetas1, Thetas2, Ys, lambda)
% Cost function for neural network

    m = length(Ys);
    
    % Lambda not yet implemented
    lambda = (lambda/(2 * m)) * (sum(sum(Thetas1(2:end,:)).^2)) + ...
              sum(Thetas2(2:end,:).^2);
    
    % Calculate hypotheses
    [a2, a3] = forwardProp(Xs, Thetas1, Thetas2);
    
    % Calculate cost
    cost = (-1/m) * sum(Ys .* log(a3)' + (1 - Ys) .* log(1 - a3)');% + lambda;
    
end

