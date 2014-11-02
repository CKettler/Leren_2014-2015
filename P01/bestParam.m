function [ theta_0, theta_1 ] = bestParam( theta_0, theta_1, ...
                                           alpha, X, Y, max_iter )
% updates theta_0 and theta_1 to find the best values (by minimizing mean
% squared error)

    for i = 1:max_iter
        [theta_0, theta_1] = updateParam(theta_0, theta_1, alpha, X, Y);
    end % end for i

end

