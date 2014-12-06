%%% Leren Practicum # 5
%%% Thomas Meijers (10647023)
%%% December 2014

%% Q1.a - Naive Bayes Classifier

% Load data
load('digit_dataset.mat');

% Read test data
[Xs_tst, Ys_tst] = read_dataset(digits1232);

% Train classifiers
[classes, pY, Xs_tr_mu, Xs_tr_sigma] = trainClassifiers(digits1231);


tic;
acc_iter = naiveBayesIter(Xs_tst, Ys_tst, Xs_tr_sigma, Xs_tr_mu, pY, classes);
time_iter = toc;
tic;
[acc_vector, predicted] = naiveBayesVector(Xs_tst, Ys_tst, Xs_tr_sigma, Xs_tr_mu, pY, classes);
time_vector = toc;

fprintf('\nQ1.a - Naive Bayes Classifier\n');
fprintf('Iteration based naive Bayes: \tAcc = %f\t Time = %f\n', acc_iter, time_iter);
fprintf('Vector based naive Bayes: \t\tAcc = %f\t Time = %f\n\n', acc_vector, time_vector);

% Q1.b - Analyzing wrong predictions
% REMARK FOR GRADING: To check evaluations remove semicolons to unsurpress 
% the output used for analyzing these predictions.

fprintf('Q1.b - Analyzing wrong predictions\n');
fprintf('Ones predicted correctly: \t\t%i/80\n', predicted(1));
fprintf('Twos predicted correctly: \t\t%i/80\n', predicted(2));
fprintf('Threes predicted correctly: \t%i/80\n', predicted(3));

% The two wrong predictions for 1 are the 60th and 63rd training test
% example (uncomment rule 40 in naiveBayesVector.m to show these indices).
% Let's test what their deviation is per feature:

% Test example #60
dev11 = (Xs_tr_mu(1,:) - Xs_tst(60,:)) ./ Xs_tr_sigma(1,:);
dev12 = (Xs_tr_mu(2,:) - Xs_tst(60,:)) ./ Xs_tr_sigma(2,:);
dev13 = (Xs_tr_mu(3,:) - Xs_tst(60,:)) ./ Xs_tr_sigma(3,:);
% Test example #63
dev21 = (Xs_tr_mu(1,:) - Xs_tst(63,:)) ./ Xs_tr_sigma(1,:);
dev22 = (Xs_tr_mu(2,:) - Xs_tst(63,:)) ./ Xs_tr_sigma(2,:);
dev23 = (Xs_tr_mu(3,:) - Xs_tst(63,:)) ./ Xs_tr_sigma(3,:);
% Filter out the really low devations where mean is at or near 0.
del_indices1 = find(Xs_tr_mu(1,:) < 10e-10);
del_indices2 = find(Xs_tr_mu(2,:) < 10e-10);
del_indices3 = find(Xs_tr_mu(3,:) < 10e-10);
% Indices where mu = 0 for all mu's, delete these from devations
del_indices = intersect(del_indices1, del_indices2);
del_indices = intersect(del_indices, del_indices3);
dev11(del_indices) = [];
dev12(del_indices) = [];
dev13(del_indices) = [];
dev21(del_indices) = [];
dev22(del_indices) = [];
dev23(del_indices) = [];

% Lets compare these devations with the other classifiers
bigger12 = find(dev11 > dev12);
bigger13 = find(dev11 > dev13);
bigger22 = find(dev21 > dev22);
bigger23 = find(dev21 > dev23);
% Find commong higher deviations in examples
bigger1 = intersect(bigger12, bigger22);
bigger2 = intersect(bigger13, bigger23);
bigger = intersect(bigger1, bigger2);
% See what removing these does:
Xs_tst_tmp = Xs_tst;
Xs_tr_sigma_tmp = Xs_tr_sigma;
Xs_tr_mu_tmp = Xs_tr_mu;
Xs_tst_tmp(:,bigger) = [];
Xs_tr_sigma_tmp(:,bigger) = [];
Xs_tr_mu_tmp(:,bigger) = [];
[acc_vector, predicted] = naiveBayesVector(Xs_tst_tmp, Ys_tst, ...
    Xs_tr_sigma_tmp, Xs_tr_mu_tmp, pY, classes);
false_predictions = 80 - predicted(1);
% Unfortunately we now predict 4 wrong. The two old ones plus example 53
% and 55. Let's try another method.

% Let's also find indices from features that have a high devation from 
% class 1:
highdev1 = find(abs(dev11) > 1);
highdev2 = find(abs(dev21) > 1);
% Then find common indices:
highdev = intersect(highdev1, highdev2);
% Let's try removing these features and classify the set:
Xs_tst_tmp = Xs_tst;
Xs_tr_sigma_tmp = Xs_tr_sigma;
Xs_tr_mu_tmp = Xs_tr_mu;
Xs_tst_tmp(:,highdev) = [];
Xs_tr_sigma_tmp(:,highdev) = [];
Xs_tr_mu_tmp(:,highdev) = [];
[acc_vector, predicted] = naiveBayesVector(Xs_tst_tmp, Ys_tst, ...
    Xs_tr_sigma_tmp, Xs_tr_mu_tmp, pY, classes);
false_predictions = 80 - predicted(1);
% We now predict one wrong, 63, so we have fixed the wrong prediction of
% feature 60. I find this enough work for this question, probably also for
% grading it.

%% Q2.a

%% Q2.b

