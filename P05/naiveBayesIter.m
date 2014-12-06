function [accuracy] = naiveBayesIter(Xs_tst, Ys_tst, ...
                        Xs_tr_sigma, Xs_tr_mu, pY, classes)
% NAIVEBAYESITER is a iteration based implementation of the Naive Bayes
% Classifier algorithm.
%
% @INPUT ARGUMENTS:
% - Xs_tst = Matrix with test set features and examples.
% - Ys_tst = Vector with classes per example.
% - Xs_tr_mu = matrix with trained mean per variable per class.
% - Xs_tr_sigma = matrix with trained std dev per variable per class.
% - classes = vector with the seperate classes.
% - pY = vector with trained base chances for each class.
%
% @OUTPUT ARGUMENTS:
% - accuracy = accuracy of the predictions
% - predicted = how many per class were falsely predicted (Q1.b)


    % Correct predictions
    correct = 0;
    
    % Loop through training examples
    for i = 1:size(Xs_tst, 1)
        % Initialise array with chances per class
        Ps = ones(1, 3);
        % Loop through classes
        for c = 1:length(classes)
            % Loop through features per example
            for j = 1:size(Xs_tst, 2)
                % Chance is previous total chance times chance obtained
                % through normal distribution of new feature with mean and
                % std dev of that feature.
                Ps(c) = Ps(c) * ((1 / (Xs_tr_sigma(c, j) * sqrt(2 * pi))) * ...
                        exp((-(Xs_tst(i, j) - Xs_tr_mu(c, j))^2) / ...
                            (2 * Xs_tr_sigma(c, j)^2)));
            end % End for j (features)
        end % end for c (classes)
        
        % Chance times base chance on class
        Ps(c) = Ps(c) * pY(c);
        
        % Get prediction
        pred = max(Ps);
        if pred == Ps(1)
            pred = 1;
        elseif pred == Ps(2)
            pred = 2;
        else 
            pred = 3;
        end % End prediction if else
        
        % Plus one if predicted correct
        correct = correct + (pred == Ys_tst(i));
    end % end for i (test example)

    % Calculate accuracy
    accuracy = correct / length(Ys_tst) * 100;

end

