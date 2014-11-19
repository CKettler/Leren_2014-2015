function [a2, a3] = forwardProp(Xs, Thetas1, Thetas2)
% Applies forward propagation

    % Initalise activation values for hidden layer
    a2 = [0 0 0];
    
    % Calculate activiation values
    a2 = hypNN(Xs, Thetas1);
    
    % Add bias node in hidden layer
    a2 = [ones(length(a2), 1), a2];
    
    % Calculate output activation value
    a3 = hypNN(a2, Thetas2);
    
end

