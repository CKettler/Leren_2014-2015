function [ cov_matrices, means, classes ] = train_gauss(xs, ys)
    % enumerates dataset, extract classes and calculates the mean
    % and covariance matrix for each class

    classes = unique(ys)';

    % cell where we store our covariance matrices. 1 for each class
    cov_matrices = cell(size(classes, 1), 1);
    
    % vector where we store mean value for each vector
    means = cell(size(classes, 1));
    
    % calculate covariance matrix for each class
    for c=classes
        % get xs in question
        xs_f = xs(ys == c, :);
        
        % get mean
        means{c} = mean(xs_f);
        
        % calculate cov
        cov_matrices{c} = cov(xs_f);
    end
end

