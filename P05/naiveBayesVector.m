function [accuracy, predicted] = naiveBayesVector(Xs_tst, Ys_tst, ...
                        Xs_tr_sigma, Xs_tr_mu, pY, classes)
% NAIVEBAYESVECTOR is a vector based implementation of the Naive Bayes
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


    % Get number of test examples to initialize arrays
    examples = size(Xs_tst, 1);
    % All our chances
    Ps = zeros(size(Xs_tst, 1), length(classes));
    
    % Loop through classes
    for c = 1:length(classes)
        % Replicate matrix so we can do matrix operations
        mu = repmat(Xs_tr_mu(c,:), examples, 1);
        sigma = repmat(Xs_tr_sigma(c,:), examples, 1);
        % Obtain probability density per class
        Ps(:,c) = prod(normpdf(Xs_tst, mu, sigma), 2) .* pY(c);
    end % end for c (classes)
    
    % Check which classes are predicted
    pred = max(Ps, [], 2);
    pred = repmat(pred, 1, size(Ps, 2));
    pred = pred == Ps;
    
    % Seperate them by classes
    pones = pred(1:80,:);
%      find(pones(:,1) == 0) % For Q1.b (see which were predicted wrong)
    pones = length(pones(pones(:,1) == 1));
    ptwos = pred(81:160,:);
    ptwos = length(ptwos(ptwos(:,2) == 1));
    pthrees = pred(161:240,:);
    pthrees = length(pthrees(pthrees(:,3) == 1));
    
    % Calculate accuracy
    predicted = [pones, ptwos, pthrees];
    accuracy = (sum(predicted)) / examples * 100;

end

