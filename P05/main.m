%%% Leren Practicum # 5
%%% Thomas Meijers (10647023)
%%% 6 December 2014

%% Q1.a - Naive Bayes Classifier
% REMARK: Q1.b is included in this section, see below

% Load data
load('digit_dataset.mat');

% Read test data
[Xs_tst, Ys_tst] = read_dataset(digits1232);

% Train classifiers
[classes, pY, Xs_tr_mu, Xs_tr_sigma] = trainClassifiers(digits1231);


% Predict, get accuracy and time
tic;
acc_iter = naiveBayesIter(Xs_tst, Ys_tst, Xs_tr_sigma, Xs_tr_mu, pY, classes);
time_iter = toc;
tic;
[acc_vector, predicted] = naiveBayesVector(Xs_tst, Ys_tst, Xs_tr_sigma, Xs_tr_mu, pY, classes);
time_vector = toc;
    
fprintf('\nQ1.a - Naive Bayes Classifier\n');
fprintf('Iteration based naive Bayes: \tAcc = %f\t\tTime = %f\n', acc_iter, time_iter);
fprintf('Vector based naive Bayes: \t\tAcc = %f\t\tTime = %f\n\n', acc_vector, time_vector);

% Q1.b - Analyzing wrong predictions
% REMARK FOR GRADING: To check evaluations remove semicolons to unsurpress 
% the output used for analyzing these predictions.

fprintf('Q1.b - Analyzing wrong predictions\n');
fprintf('Ones predicted correctly: \t\t%i/80\n', predicted(1));
fprintf('Let''s analyze these two wrong predicted examples, see code and comments.\n')

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

clear; 

%% Q2.a - Statistic test comparing Naive Bayes with K Nearest Neighbours

% NAIVE BAYS ACCURACY (code from Q1.a)
% Load data
load('digit_dataset.mat');
% Read test data
[Xs_tst, Ys_tst] = read_dataset(digits1232);
% Train classifiers
[classes, pY, Xs_tr_mu, Xs_tr_sigma] = trainClassifiers(digits1231);
% Predict, get accuracy
[acc_nbayes, ~] = naiveBayesVector(Xs_tst, Ys_tst, Xs_tr_sigma, Xs_tr_mu, pY, classes);

% K NEAREST NEIGHBOUR ACCURACY (code from P04)
% Read data
[Xs_train, Cs_train] = read_dataset(digits1231);
[Xs_test, Cs_test] = read_dataset(digits1232);
% sqrt(n) seems to be a good measure for k
k = ceil(sqrt(length(Xs_test)));
results = zeros(length(Xs_test), 1);
for i = 1:length(Xs_test)
    % no weights
    pred = kNN(Xs_train, Cs_train, Xs_test(i,:), k, 'standard', false);
    C_gold = Cs_test(i);
    results(i) = (pred == C_gold);
end
acc_knn = (length(results(results==1)) / length(results)) * 100;

% Mean is zero if no difference
mu = 0;
% Actual difference is
diff = acc_knn - acc_nbayes;
% Get std dev
p = (acc_nbayes + acc_knn) / 200;
sigma = sqrt((2 * p * (1 - p)) / 100);
% Now calculate z score and score needed to reject the hypothesis that
% there is no difference between the algorithms (accuracy wise)
z = diff / sigma;
z_reject = sqrt(2) * erfcinv(0.025*2);

fprintf('\nQ2.a - Accuracy Naive Bayes vs. K Nearest Neighbours\n');
fprintf('Z-score = %f\t\tZ-threshold to reject = %f\n', z, z_reject);

% This show's us the deviation for this difference in accuracy is 212 times
% the standard devation, we reject the hypothesis that the accuracy of both
% methods is equal with a z-score above 1.96 or below -1.96, thus we can
% assume that the accuracy of these methods is (very) different.

clear;

%% Q2.b - Swapping test and training sets

% I predict different results for multiple reasons, the first is we have
% noise in either one of the sets, else our accuracy would have been 100%
% at the previous question. This means that we can't have a perfect
% classifier and there is a descripancy between these sets. Secondly, using
% this information, one set (digits123-1) consists of more examples, namely
% 300 versus the 240 in the other set (digits123-2) which matters since we
% have noise in our data.

% Load data
load('digit_dataset.mat');

% NON SWAPPED (same as Q1)
% Read test data
[Xs_tst, Ys_tst] = read_dataset(digits1232);
% Train classifiers
[classes, pY, Xs_tr_mu, Xs_tr_sigma] = trainClassifiers(digits1231);
% Predict and get accuracy
[acc_notswapped, ~] = naiveBayesVector(Xs_tst, Ys_tst, Xs_tr_sigma, Xs_tr_mu, pY, classes);

% SWAPPED
% Read test data (now digits123-1 as test set)
[Xs_tst, Ys_tst] = read_dataset(digits1231);
% Train classifiers (now train on digits123-2)
[classes, pY, Xs_tr_mu, Xs_tr_sigma] = trainClassifiers(digits1232);
% Predict and get accuracy
[acc_swapped, ~] = naiveBayesVector(Xs_tst, Ys_tst, Xs_tr_sigma, Xs_tr_mu, pY, classes);

fprintf('\nQ2.b - Swapping data sets\n');
fprintf('Difference in accuracy: \tNot swapped = %f\t\tSwapped = %f\n\n', acc_notswapped, acc_swapped);

% And indeed, the result is (very) different. This is due to the reason
% named above, namely the difference in size.

clear; 