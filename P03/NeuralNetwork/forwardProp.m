function [a2, a3] = forwardProp(Xs, Thetas1, Thetas2)
% Applies forward propagation

    % Calculate activiation values
    a2 = hypNN(Thetas1, Xs);
    
    % Add bias node in hidden layer
    a2 = [ones(1, size(a2, 2)); a2];
    
    % Calculate output activation value
    a3 = hypNN(Thetas2, a2);
    
end

