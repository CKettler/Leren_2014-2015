function [ Params, Xs, Ys ] = multiple(Thetas, alpha, iterations)
    %% Q1.1: LOAD DATA (ESSENTIAL FOR RUNNING OTHER SECTIONS)
    % Houses data

    data = csvread('housesRegr.csv', 1, 0);
    Xs = data(:,2:(end-1));
    Xs = [ones(length(Xs),1 ), Xs];
    Ys = data(:,end); % Extract all Y values

    % normalize data
    Xs_size = size(Xs);
    for i=1:Xs_size(2)
        Xs(:,i) = normalizeColumn(Xs(:,i));
    end
    % also normalize price
    Ys = normalizeColumn(Ys);

    Params = bestParamLinear(Thetas, alpha, iterations, Xs, Ys);
end