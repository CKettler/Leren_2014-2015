% Leren Assignment # 4: Gaussian Probability and k-means clustering
%
% Markus Pfundstein, 10452397
% Thomas Meijers, 10647023

% REMARK: PLEASE ADD ALL SUBFOLDERS TO YOUR PATH

%% Excursion: Just some playing

load('digit_dataset.mat');

[xs_train, ys_train] = read_dataset(digits1231);
[xs_test, ys_test] = read_dataset(digits1232);

[cov_matrices, means, classes] = train_gauss(xs_train, ys_train);

goods = zeros(size(xs_test, 1), 1);
for i = 1:size(xs_test, 1)
    probs = run_gauss(xs_test(i,:), cov_matrices, means, classes);  
    probs = cell2mat(probs);
    [~, predicted_y] = max(probs);
    goods(i) = predicted_y == ys_test(i);
    %fprintf('predicted_y: %d, ys_test: %d\n', predicted_y, ys_test(i));
end

accuracy = nnz(goods) / size(goods, 1);

fprintf('accuracy %1.02f\n', accuracy * 100);

clear;

%% Q1: Gaussian Probability Density

load('digit_dataset.mat');

[xs_train, ys_train] = read_dataset(digits1231);
[cov_matrices, means, classes] = train_gauss(xs_train, ys_train);

% get classes for first and second example
class_example1 = ys_train(1);
class_example2 = ys_train(2);

% get gaussian probabilities for first and second example in xs_train 
prob_example1 = run_gauss(xs_train(1,:), cov_matrices, means, classes);
prob_example2 = run_gauss(xs_train(2,:), cov_matrices, means, classes);

% this assignment doesn't really make sense but ok. lets print stuff
fprintf('probability example 1 for class %d: %d\n', class_example1, prob_example1{class_example1});
fprintf('probability example 2 for class %d: %d\n', class_example2, prob_example2{class_example2});


%% Q2: Finding anomalies

load('digit_dataset.mat');

[xs_train, ys_train] = read_dataset(digits1231);

idx1 = find(ys_train==1, 1);
idx2 = find(ys_train==2, 1);
idx3 = find(ys_train==3, 1);

ys_trainf = ys_train;
ys_trainf(idx1) = 2;
ys_trainf(idx1 + 1) = 2;
ys_trainf(idx2) = 3;
ys_trainf(idx2 + 1) = 3;
ys_trainf(idx3) = 1;
ys_trainf(idx3 + 1) = 1;

% train as if there are no fault labeled instances
[cov_matrices, means, classes] = train_gauss(xs_train, ys_trainf);

epsilon = 2e-29;
fprintf('run anomaly detection with epsilon: %d\n', epsilon);
% do anomaly detection on each class
anomalies_found_total = 0;
correct_found = 0;
for class=classes
    fprintf('test for anomalies in class: %d\n', class);
    
    % get indices from faulty target set. 
    indices = ys_trainf == class;
    
    % extract values from REAL target set...
    ys_f = ys_train(indices);
    
    % extract xs from faulty target set
    xs_f = xs_train(indices, :);
    % check for each the probability
    for i=1:size(xs_f, 1)
        p = run_anomaly(xs_f(i,:), cov_matrices{class}, means{class});
        % if p is very low, then we should have an anomaly.
        if p < epsilon
            % if current class is different then class should be in REAL
            % target set, its one of our 'artificial' anomalies
            if ys_f(i) ~= class
                fprintf('found correct anomaly of class %d\n', ys_f(i));
                fprintf('probability: %d\n', p);
                correct_found = correct_found + 1;
            end
   
            anomalies_found_total = anomalies_found_total + 1;
        end
    end
end

fprintf('correct anomalies found: %d. Total anomalies found: %d\n', correct_found, anomalies_found_total);

% As we can see, all anomalies have been detected but also 6 anomalies that
% we didn't introduce. 

clear

%% Q3: K-Means Clustering
% Implementation of the k means clustering algorithm, tested on the data
% digits123.

clear all;
close all;

% Load and read data
load('digit_dataset.mat');
[xs_train, ys_train] = read_dataset(digits1231);
[xs_test, ys_test] = read_dataset(digits1232);
xs = [xs_train; xs_test];
ys = [ys_train; ys_test];
clearvars -except xs ys % Clear all raw data, keep data structures

% Set parameters
k = [1:5 7 10]; % Even though any other k than 3 is ridiculous, our program
                  % can run with vectors of arbitrary length and values
iter = 10; % Nr. of iterations per k
inits = 10; % Nr. of new initializations per k
print = true; % Print k and accuracy

% Run algorithm and determine accuracy for obtained centroids
[centroids, best_cost] = k_means(xs,  k, iter, inits, print);
accuracies = k_means_acc(xs, ys, centroids, print);

% Find elbow point (based on relative increase in accuracy)
% REMARK: Most of the time the point will be at k = 3. However, when one
% uses a large range of k (say a 1:20 vector) it will (ofcourse) fluctuate
eff_incr = zeros(1, length(accuracies) - 1);
for i = 1:length(accuracies) - 1
    eff_incr(i) = (accuracies(i + 1) - accuracies(i)) / (k(i + 1) - k(i));
end % end for i = relative accuracy
rel_incr = zeros(1, length(eff_incr) - 1);
for i = 1:length(eff_incr) - 1
    rel_incr(i) = eff_incr(i) / eff_incr(i + 1);
end % end for i
% Now find index so we can find "optimal" k
optimal_k = k(find(rel_incr == max(rel_incr)) + 1);

% Now plot the data
figure('name','Question 3: k-means clustering');
hold on;
% Plot all points
plot(k, accuracies, '-bo', 'MarkerEdgeColor', 'b', ...
        'MarkerFaceColor', 'b', 'MarkerSize', 5);
% Plot elbow point in different style
plot(optimal_k, accuracies(find(k == optimal_k)), 'o', ...
    'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
hold off;
% Other cosmetic options
title('Average accuracy for k clusters on Digits123');
legend('Data point', 'Elbow point', 'Location','southeast');
ylabel('% Average accuracy');
xlabel('# clusters');
set(gca,'YLim',[0 100]);
set(gca,'XLim',[min(k) max(k)]);
set(gca,'XTick',[min(k):max(k)]);
set(gca,'XTickLabel',[k(1):k(end)]);

clear all;