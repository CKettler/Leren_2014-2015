function [ Thetas ] = updateLogRegularized(Thetas, alpha, Xs, Ys, lambda)
    % updates thetas
    
    m = length(Xs);
    % Calculate gradients
    gradients = alpha * gradientLogistic(Thetas, Xs, Ys);
    % Theta 0 shouldn't be regularized
    t0 = Thetas(1) - gradients(1);
    % Other thetas are regularized
    T_rest = (Thetas(2:end)' - gradients(2:end) - (lambda/(2 * m)) * Thetas(2:end)')';
    
    Thetas = [t0, T_rest];

end

