function [ C_predicted ] = kNN( Xs, Cs, X, k, weight_distance, featurews )
    % k nearest neighbour function.
    % get distance to all points
    Dists = zeros(size(Xs, 1), 1);
    for i = 1:size(Xs,1)
        Dists(i) = euclidean_distance(Xs(i,:), X, featurews);
    end
                            
    % sort by distance, get indices
    [Dists, indices] = sort(Dists);
    % sort Cs by indices
    Cs = Cs(indices,:);
    
    % get first k elements
    Cs_k = Cs((1:k),:);
    
    if (strcmp(weight_distance, 'distance'))
        % get classes that will vote
        uniq = unique(Cs_k);
        % build vote matrix
        votes = [uniq, zeros(length(uniq), 1)];
        
        % get first k elements of Dists
        Dists = Dists((1:k),:);
        for i = 1:length(Dists)           
            % TO_CHECK: If this is a good measure for voting power
            % http://www.cs.colorado.edu/~grudic/teaching/CSCI4202/NearestNeighbor.pdf
            vote_power = 1/Dists(i)^2;
            
            % get index for votes matrix so that we can update it
            [~, index] = ismember(Cs_k(i), votes);
            votes(index, 2) = votes(index, 2) + vote_power;
            
        end
        % get most predictive element
        [~, index] = max(votes(:,2));
        C_predicted = votes(index);
    else
        % most occuring element is prediction
        
        % TO-CHECK: what if 2 or more elements have same occurence
        C_predicted = mode(Cs_k);
    end
end

function [ dist ] = euclidean_distance(p1, p2, featurews)
    % calculates euclidean distance between two points p1 and p2
    assert(length(p1) == length(p2), 'p1 and p2 must be of same length');
    temp = (p2 - p1).^2;
    if featurews
        
        temp = temp .* featurews';
        
    end
    dist = sqrt(sum(temp));
end