function [ accuracies ] = k_means_acc( xs, ys, centroids, print )
% K_MEANS_ACC calculates the accuracy for the given centroids and data.
%
% @INPUT ARGUMENTS:
% - xs = a M by N matrix with M data examples with each N features
% - ys = a M by 1 vector containing per row the class for each data example
% - centroids = a cell array with each cell containing a k by N matrix with
%               each row representing a centroid for the kth cluster
% - print = boolean wether to print or not
%
% @RETURNS:
% - accuracies = a 1 by k vector containing the average accuracy for the
%                given k

    classes = unique(ys);
    accuracies = zeros(1, length(centroids));
    
    for m = 1:length(centroids)
        k = size(centroids{1, m}, 1);
        % Load clusters and get centroid(s) corresponding to k
        clusters = cell(1, k);
        c = centroids{1, m};
        for i = 1:size(xs, 1)
            % Calculate euclidean distance
            x = xs(i,:);
            dist = sqrt(sum(bsxfun(@minus, c, x).^2, 2));
            min_dist = min(dist);
            % Find index and add to that cluster
            index = find(dist == min_dist);
            % If more than one (on cluster boundary) min take random index
            if length(index) > 1 
                index = datasample(index, 1);
            end
            % Now add corresponding class to cluster
            clusters{1, index} = [clusters{1, index}; ys(i)];
        end % end for i = data example
        % Once we have all the clusters calculate main class and wrong
        % points
        if print
            fprintf('Accuracy per cluster for k = %i:\n',k);
        end
        for j = 1:size(clusters, 2)
            [counts, ~] = hist(clusters{1, j}, classes);
            class = find(counts == max(counts));
            if length(class) > 1 % If classes evenly spread in cluster
                class = datasample(class, 1);
            end % End if evenly spread
            acc = max(counts) * 100 / sum(counts);
            accuracies(m) = accuracies(m) + acc;
            if print
                fprintf('* Class = %i\t\tAccuracy = %.2f \n',class, acc);
            end
        end % end for j = cluster
        accuracies(m) = accuracies(m) / k;
    end % end for k = centroids
end

