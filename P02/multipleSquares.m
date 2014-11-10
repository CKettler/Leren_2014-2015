function [ Params, Xs, Ys ] = multipleSquares(Thetas, alpha, iterations)
    % read data
    data = csvread('housesRegr.csv', 1, 0);
    Xs = data(:,2:(end-1));
    Xs = [ones(length(Xs),1 ), Xs];
    Ys = data(:,end); % Extract all Y values

    % generate squares
    Xs2 = Xs.^2;
    
    % attach them to Xs
    Xs = [Xs Xs2];
    
    % normalize data
    Xs_size = size(Xs);
    for i=1:Xs_size(2)
        Xs(:,i) = normalizeColumn(Xs(:,i));
    end
    % also normalize price
    Ys = normalizeColumn(Ys);

    Params = bestParamLinear(Thetas, alpha, iterations, Xs, Ys);
end