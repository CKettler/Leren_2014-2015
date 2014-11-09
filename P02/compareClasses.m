function [ Thetas, Ynew ] = compareClasses( Thetas, alpha, Xs, Ys, iter, y )
% Compares one class versus the rest (OvA algorithm)

    % Replace wanted Y with 1, rest with 0
    Ynew = Ys;
    Ynew ( Ys ~= y ) = 0;
    Ynew ( Ynew == y ) = 1;

    for i = 1:iter
       Thetas = updateParamLogistic( Thetas, alpha, Xs, Ynew ); 
    end

end