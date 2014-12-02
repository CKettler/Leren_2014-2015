% Leren Assignment 4
%
% Markus Pfundstein, 10452397
% Thomas Meijers, 10647023

%% pre-work for 1.3 
% find feature with highest predictive value
% weights will get put into a dataset 'digit_weights.mat' at the 
% end of this section. The real assignments start one section further

clear

load('digit_dataset.mat');

% read training and test sets
[Xs_train, Cs_train] = read_dataset(digits1231);
[Xs_test, Cs_test] = read_dataset(digits1232);

k = ceil(sqrt(length(Xs_test)));

feature_count = size(Xs_train, 2);

feature_weights = zeros(feature_count, 1);

for f = 1:feature_count
    % take i-th feature
    Xs_trainCopy = Xs_train(:,f);
    Xs_testCopy = Xs_test(:,f);

    results_no_weight = zeros(length(Xs_testCopy), 1);
    for i = 1:length(Xs_test)
        C_pred_no_weight = kNN(Xs_trainCopy, Cs_train, Xs_testCopy(i,:), k, 'standard', false);
        C_gold = Cs_test(i);
        results_no_weight(i) = (C_pred_no_weight == C_gold);
    end
    
    % calculate accuracy for each time trained with a certain feature
    % removed
    feature_weights(f) = (length(results_no_weight(results_no_weight==1)) ... 
                     / length(results_no_weight));
end

feature_weights

save('Data\digit_weights.mat', 'feature_weights');

clear;

%% 1.1 Implementation of k nearest neighbour and test run on digit set
%  & 1.2 Distance based voting
%  & 1.3 Feature weighted voting

% loads digits1231, digits1232
load('digit_dataset.mat');

% loads feature weights
load('digit_weights.mat');

% read training and test sets
[Xs_train, Cs_train] = read_dataset(digits1231);
[Xs_test, Cs_test] = read_dataset(digits1232);

% http://saravananthirumuruganathan.wordpress.com/2010/05/17/a-detailed-introduction-to-k-nearest-neighbor-knn-algorithm/
% sqrt(n) seems to be a good measure for k
k = ceil(sqrt(length(Xs_test)));

results_no_weight = zeros(length(Xs_test), 1);
results_weighted = results_no_weight;
results_featurew = results_no_weight;
for i = 1:length(Xs_test)
    % no weights
    C_pred_no_weight = kNN(Xs_train, Cs_train, Xs_test(i,:), k, 'standard', false);
    % weight by distance
    C_pred_weighted = kNN(Xs_train, Cs_train, Xs_test(i,:), k, 'distance', false);
    % weight by weighted distances
    C_pred_featurew = kNN(Xs_train, Cs_train, Xs_test(i,:), k, 'distance', feature_weights);
    C_gold = Cs_test(i);
    results_no_weight(i) = (C_pred_no_weight == C_gold);
    results_weighted(i) = (C_pred_weighted == C_gold);
    results_featurew(i) = (C_pred_featurew == C_gold);
end

accuracy_no_weight = (length(results_no_weight(results_no_weight==1)) ... 
                     / length(results_no_weight)) * 100;
accuracy_weighted = (length(results_weighted(results_weighted==1)) ... 
                     / length(results_weighted)) * 100;
accuracy_featurew = (length(results_featurew(results_featurew==1)) ... 
                     / length(results_featurew)) * 100;

fprintf('scores for kNN with k = %d\n', k);
fprintf('accuracy regular kNN: %f\n', accuracy_no_weight);
fprintf('accuracy distance weighted kNN: %f\n', accuracy_weighted);
fprintf('accuracy feature weighted kNN: %f\n', accuracy_featurew);

clear;

%% 2.1

% We suggest that classification algorithms would be most suitable for a
% problem like this. This includes Logistic Regression with a One-Vs-All
% classification, A Neural Network with a sigmoid activation function and
% K-Nearest Neighbour. 
% Linear Regression could also work but then we would have to map a
% continous output space to a discrete value space. This is not preferable.

%% 2.2

% See opdracht 1.1, we implemented accuracy there.

%% 2.3

% REMARK: The steps followed for this assignment will be described and
% discussed within this matlab script. This way we can execute code and one
% can clearly see where which comments are appropriate. 

% The algorithms we can choose for this specific problem (digits) are the
% following:
%   - Linear Regression
%   - Logistic Regression
%   - Neural Network
%   - K-Nearest Neighbour
%   - Decision Tree (Only theoretically since it's not implemented). 

% But before we blindly start testing the algorithms, let's analyze the 
% problem first: 

% We have a classification problem with three classes at hand, which
% already rules out Linear Regression as we have many more algorithms which
% handle classification problems way better.

% Besides it being a classification problem we have a dataset with 64
% features. If a few of these features would have significant weighting we
% could easily use a decision tree. Unfortunately assignment 1.3 (see
% above) has taught us that all the features are quite close to one another
% with a mean weight around 0.4, and a range of less than 0.3. This tells
% us something we already know about the features. As they correspond with
% grey values of written digits most of the features are weighted quite
% equally. Ofcourse the features at the edges don't tell us as much but
% this still leaves us too many features to build a concise decision tree.

% So the algorithms we have left are:
%   - Logistic Regression
%   - Neural Network
%   - K-Nearest Neighbour

% Let's run a test for these three algorithms, we will use accuracy as our
% cost/error function as we can then efficiently compare the results of
% these three different algorithms.

% Load data
load('digit_dataset.mat');

% % % % % % % % % % LOGISTIC REGRESSION % % % % % % % % % %

% read training and test sets
[Xs_train, Cs_train] = read_dataset(digits1231);
[Xs_test, Cs_test] = read_dataset(digits1232);

Xs_train = [ones(size(Xs_train, 1), 1), Xs_train];
Xs_test = [ones(size(Xs_test, 1), 1), Xs_test];

alpha = 0.1;
lambda = 0; % No regularization as it's not implemented for neural networks
iterations = 100;
thetas = ones(1, size(Xs_train, 2)) / 2;

% Train thetas
thetasLR1 = compareClasses(thetas, alpha, Xs_train, Cs_train, iterations, 1);
thetasLR2 = compareClasses(thetas, alpha, Xs_train, Cs_train, iterations, 2);
thetasLR3 = compareClasses(thetas, alpha, Xs_train, Cs_train, iterations, 3);

% Get hypotheses
h1 = hypLogistic(thetasLR1, Xs_train);
h2 = hypLogistic(thetasLR2, Xs_train);
h3 = hypLogistic(thetasLR3, Xs_train);
    
% Check correct guesses on the test set
correct = 0;
for i = 1:length(Cs_test)
    
    % Get maximum hypothesis
    h = max([h1(i), h2(i), h3(i)]);
    pred = NaN();
    
    if (h == h1(i)) 
        pred = 1;
    elseif (h == h2(i))
        pred = 2;
    elseif (h == h3(i))
        pred = 3;
    end
    
    if (pred == Cs_test(i))
        correct = correct + 1;
    end
end

accuracy = (correct / size(Xs_test, 1)) * 100;
fprintf('1. Accuracy Logistic Regression: %f\n', accuracy);


% % % % % % % % % % NEURAL NETWORK % % % % % % % % % %

% Read data
[Xs_train, Cs_train] = read_dataset(digits1231);
[Xs_test, Cs_test] = read_dataset(digits1232);
classes = length(unique(Cs_train));

training_examples = size(Xs_train, 1);
features = size(Xs_train, 2);
iterations = 100;
alpha = 0.05;

% Get number of nodes and edges
input_nodes = features + 1; % Plus bias node
hidden_nodes = input_nodes; % Already includes bias
output_nodes = 1; % Will use one vs. all classification

edges_first = input_nodes * (hidden_nodes - 1); % No input to hidden bias
edges_second = hidden_nodes * output_nodes;
edges = edges_first + edges_second;

% generate random theta for each connection
epsilon = 0.5;
Thetas1 = (rand(input_nodes, hidden_nodes - 1) * (2 * epsilon) - epsilon);
Thetas2 = (rand(hidden_nodes, output_nodes) * (2 * epsilon) - epsilon);

T1C = zeros(size(Thetas1, 1), size(Thetas1, 2), classes);
T2C = zeros(size(Thetas2, 1), size(Thetas2, 2), classes);

% accumulators from video
Grad1 = zeros(size(Thetas1));
Grad2 = zeros(size(Thetas2));

for class = 1:classes
    % Multiclass, will try one versus all on neural networks
    T1C(:,:,class) = Thetas1;
    T2C(:,:,class) = Thetas2;
    
    for it = 1:iterations
        %fprintf('--> iterate %d/%d\n', it, iterations);
        % loop through training examples.
        % we do first Forward Propagation with input set k
        % then we calculate deltas for each node and then we calculate our gradient
        % finally, after the inner for loop, we update our thetas
        for k = 1:training_examples
            % forward propagation on data set i
            [Output, As2] = forwardPropagation2Layer( ...
                T1C(:,:,class), T2C(:,:,class), ...
                Xs_train(k,:));

            %fprintf('output was: %f, should be %f\n', Output(1), Ys(k));

            % calculate deltas (errors) for all layers
            [DeltasL, Deltas2] = calculateDeltas2Layer(Output, ...
                Cs_train(k), As2, T2C(:,:,class), class);
            % drop delta for bias in hidden layer
            Deltas2 = Deltas2((2:end),:);

            % add bias unit to training set Xs. we do this already in
            % forwardPropagation2Layer as well but its not destructive. 
            % Hence we need to do it again
            Zs = [1 Xs_train(k,:)];

            % loop trough our gradient matrix and update values
            % I tried to figure out a way to do this with matrices but couldnt
            for i=1:6
                for j=1:1
                    Grad2(i,j) = Grad2(i,j) + As2(i) * DeltasL(j);
                end
            end
            for i=1:4
                for j=1:5
                    Grad1(i,j) = Grad1(i,j) + Zs(i) * Deltas2(j);
                end
            end   
        end
        % update thetas
        T1C(:,:,class) = T1C(:,:,class) - ...
            alpha/training_examples * Grad1;
        T2C(:,:,class) = T2C(:,:,class) - ...
            alpha/training_examples * Grad2;
    end
end
    
% Check correct guesses on the test set
correct = 0;
for i = 1:length(Cs_test)
    
    output1 = forwardPropagation2Layer(T1C(:,:,1), T2C(:,:,1), Xs_test(i,:));
    output2 = forwardPropagation2Layer(T1C(:,:,2), T2C(:,:,2), Xs_test(i,:));
    output3 = forwardPropagation2Layer(T1C(:,:,3), T2C(:,:,3), Xs_test(i,:));
    
    % Get maximum hypothesis
    h = max([output1, output2, output3]);
    pred = NaN();
    
    if (h == output1) 
        pred = 1;
    elseif (h == output2)
        pred = 2;
    elseif (h == output3)
        pred = 3;
    end
    
    if (pred == Cs_test(i))
        correct = correct + 1;
    end
end

accuracy = (correct / length(Cs_test)) * 100;
fprintf('2. Accuracy Neural Network: %f\n', accuracy);


% % % % % % % % % % K Nearest Neighbour % % % % % % % % % %

% Read data
[Xs_train, Cs_train] = read_dataset(digits1231);
[Xs_test, Cs_test] = read_dataset(digits1232);

% http://saravananthirumuruganathan.wordpress.com/2010/05/17/a-detailed-introduction-to-k-nearest-neighbor-knn-algorithm/
% sqrt(n) seems to be a good measure for k
k = ceil(sqrt(length(Xs_test)));

results = zeros(length(Xs_test), 1);

for i = 1:length(Xs_test)
    % no weights
    pred = kNN(Xs_train, Cs_train, Xs_test(i,:), k, 'standard', false);
    
    C_gold = Cs_test(i);
    results(i) = (pred == C_gold);
end

accuracy = (length(results(results==1)) / length(results)) * 100;
fprintf('3. Accuracy K Nearest Neighbour: %f\n', accuracy);

% First analysis, steps taken = 3:
% As we can see, KNN clearly outperforms the other algorithms on this first
% run. Logistic regression has an accuracy of 75%, neural network
% only got ~62% on this run while K Nearest achieves a accuracy of 97%!
% Even though this is only the first run, we can clearly conclude that K
% Nearest is the best algorithm for this dataset as it is computationally
% very inexpensive and the other algorithm take a lot of effort to tweak.
% The reason for this must be that the handwritten number are very clearly
% clustered in their feature space.

clear;