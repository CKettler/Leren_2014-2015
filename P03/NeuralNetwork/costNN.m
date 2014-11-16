function [ cost ] = costNN(Xs, Thetas1, Thetas2, Ys, lambda)
% Cost function for neural network

    m = length(Ys);
    lambda = (lambda/(2 * m)) * (sum(sum(Thetas1(2:end,:)).^2)) + ...
              sum(Thetas2(2:end,:).^2);
    
    hyps = forwardProp(Xs, Thetas1, Thetas2);
    
    cost = (1/m) * sum(Ys .* log(hyps) + (1 - Ys).* log(1 - hyps)) + lambda;
    
end

