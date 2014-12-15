function [ centroids, best_cost ] = k_means( xs, ks, iterations, ...
                                                inits, print )
% K_MEANS is a iteration based implementation of the k-means clustering
% algorithm. It is a unsupervised learning algorithm that calssifies the
% data into k clusters based on the euclidean distance. 
%
% @INPUT ARGUMENTS:
% - xs = data matrix to be clustered
% - ks = a vector with the k's
% - iterations = number of iterations per k and initialization
% - inits = number of times the algorithm runs for a specific k
%
% @OUTPUT ARGUMENTS:
% - accuracy = accuracy of the predictions
% - predicted = how many per class were falsely predicted (Q1.b)
%
% REMARK: We chose not to save the clusters corresponding to the 'best'
% centroids since this eats up considerable memory and time (this function
% would run at least three times as slow ).

    % Initialize cell array for centroids, best_cost and dataset size
    centroids = cell(1, length(ks));
    best_cost = inf(1, length(ks));
    [samples, features] = size(xs);
    
    for m = 1:length(ks)
        k = ks(m);
        if print
            display(k);
        end % end if print
        for l = 1:inits % Initialize centroids inits time and save best
            % Take k random initial centroids
            c = datasample(xs, k);
            for i = 1:iterations
                % Reset cost, clusters and clusterszies
                cost = 0;
                clusters = zeros(k, features);
                clustersize = zeros(k, 1);
                for j = 1:samples
                    % Calculate euclidean distance
                    x = xs(j,:);
                    dist = sqrt(sum(bsxfun(@minus, c, x).^2, 2));
                    % Find min distance, add to cost
                    min_dist = min(dist);
                    cost = cost + min_dist;
                    % Find index and add to that cluster
                    index = find(dist == min_dist);
                    % If more than one minimum, take random index
                    if length(index) > 1 
                        index = datasample(index, 1);
                    end
                    % Add to total in corresponding cluster
                    clusters(index,:) = clusters(index,:) + xs(j,:);
                    clustersize(index) = clustersize(index) + 1;
                end % end for j example

                % Update centroids
                clustersize = repmat(clustersize, 1, size(c, 2));
                c = clusters ./ clustersize;
            end % end for i = iterations
            
            % Check if cost is better than previous
            if cost < best_cost(m)
                best_cost(k) = cost;
                centroids{1, m} = c;
            end % end if cost better
            
        end % end for k = initializations
    end % end for k = k clusters

end

