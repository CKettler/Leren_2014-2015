function [ theta_0, theta_1 ] = updateParam( theta_0, theta_1, ...
                                             alpha, X, Y )
    % Update theta0 and theta1
    
    % Get gradient
    g = gradientVector(theta_0, theta_1, X, Y);
    
    % Update parameters
    theta_0 = theta_0 - alpha * g(1);
    theta_1 = theta_1 - alpha * g(2);

end

