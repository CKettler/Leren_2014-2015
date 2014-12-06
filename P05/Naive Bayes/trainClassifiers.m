function [classes, pY, Xs_tr_mu, Xs_tr_sigma ] = trainClassifiers(digits1231)
% TRAINCLASSIFIER trains the classifiers for the Naive Bayes Classifier
% algorithm.
%
% @INPUT ARGUMENTS:
% - digits1231 = matrices with data about handwritten digits, this is the
%                training set.
%
% @OUTPUT ARGUMENTS:
% - classes = vector with the seperate classes.
% - pY = vector with base chances for each class.
% - Xs_tr_mu = matrix with mean per variable per class.
% - Xs_tr_sigma = matrix with std dev per variable per class.

    
    % Read training data
    [Xs_tr, Ys_trn] = read_dataset(digits1231);
    
    % Number of classes and their chances
    classes = unique(Ys_trn);
    pY = zeros(length(classes), 1);
    pY(1) = length(Ys_trn(Ys_trn == 1)) / length(Ys_trn);
    pY(2) = length(Ys_trn(Ys_trn == 2)) / length(Ys_trn);
    pY(3) = length(Ys_trn(Ys_trn == 3)) / length(Ys_trn);

    % Get all features per class
    Xs_tr1 = Xs_tr(Ys_trn == 1,:);
    Xs_tr2 = Xs_tr(Ys_trn == 2,:);
    Xs_tr3 = Xs_tr(Ys_trn == 3,:);

    % Calculate mean per class
    Xs_tr_mu = zeros(length(classes), size(Xs_tr, 2));
    Xs_tr_mu(1,:) = mean(Xs_tr1);
    Xs_tr_mu(2,:) = mean(Xs_tr2);
    Xs_tr_mu(3,:) = mean(Xs_tr3);
    Xs_tr_mu(Xs_tr_mu == 0) = 10e-100; % Set 0's to a very small value 
    % so we can't divide by 0 for certain calculations

    % Calculate standard deviation per class
    Xs_tr_sigma = zeros(length(classes), size(Xs_tr, 2));
    Xs_tr_sigma(1,:) = std(Xs_tr1);
    Xs_tr_sigma(2,:) = std(Xs_tr2);
    Xs_tr_sigma(3,:) = std(Xs_tr3);
    Xs_tr_sigma(Xs_tr_sigma < 0.1) = 0.1;

end

