%%% Leren Practicum # 3
%%% Thomas Meijers (10647023)
%%% November 2014

%% Q1.a

% Regularized Logistic Regression

% Data:
Xs = [1 1 1 1; 10 7 5 2; 4 3 4 3; 4 3 2 1]';
Ys = [0 0 1 1]';
% Parameters and variables
Thetas = [0.5, 0.5, 0.5, 0.5];
iter = 1000;
alpha = 0.1;
lambda = 1;
y = 1;

% Update parameters, print cost (plot done within compareClasses)
[Thetas_new, Ys_new] = compareClasses(Thetas, alpha, Xs, Ys, iter, y, lambda);
cost = costLogRegularized(Thetas_new, Xs, Ys_new, lambda)

clear

%% Q1.b

Xs1 = [1 1 3; 1 2 6; 1 3 8; 1 4 9; 1 6 9; 1 7 8; 1 8 6; 1 9 3; ...
       1 1 1; 1 2 2; 1 3 3; 1 4 4; 1 6 4; 1 7 3; 1 8 2; 1 9 1];
Ys = [1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0]';

Thetas = [0.5 0.5 0.5];
iter = 1000;
alpha = 0.1;
lambda = 1;
y = 1;

[Thetas_new, Ys_new] = compareClasses(Thetas, alpha, Xs1, Ys, iter, y, lambda);
cost = costLogRegularized(Thetas_new, Xs1, Ys_new, lambda)

% Not done yet

%% Q2.a

% Data (Q1.a):
Xs = [1 1 1 1; 10 7 5 2; 4 3 4 3; 4 3 2 1];
Ys = [0 0 1 1]';

% Nodes per layer
input1 = size(Xs, 2); % Already including Bias
hidden2 = input1;
output3 = 1;

% Edges per layer
edges_1to2 = input1 * (hidden2 - 1);
edges_2to3 = hidden2 * output3;

% Init random thetas per node per layer
epsilon = 0.5;
Thetas1 = rand(hidden2 - 1, input1) * (2 * epsilon) - epsilon;
Thetas2 = rand(output3, hidden2) * (2 * epsilon) - epsilon;

% Forward propagation
[a2, a3] = forwardProp(Xs, Thetas1, Thetas2);

% Print output vs Y values
output_vs_y = [a3', Ys]

clear

%% Q2.b & c
% Combines b and c since the full network comprises error backpropagation

% Data (Q1.a):
Xs = [1 1 1 1; 10 7 5 2; 4 3 4 3; 4 3 2 1];
Ys = [0 0 1 1]';

% Nodes per layer
input1 = size(Xs, 2); % Already including Bias
hidden2 = input1;
output3 = 1;

% Edges per layer
edges_1to2 = input1 * (hidden2 - 1);
edges_2to3 = hidden2 * output3;

% Init random thetas per node per layer
epsilon = 1;
Thetas1 = (rand(input1, hidden2 - 1) * (2 * epsilon) - epsilon)';
Thetas2 = (rand(hidden2, output3) * (2 * epsilon) - epsilon)';

% Thetas to 0.5 for testing:
% Thetas1 = ones(size(Thetas1)) / 2;
% Thetas2 = ones(size(Thetas2)) / 2;

% Variables
alpha = 0.05;
iter = 10000;
lambda = 0; % Algorithm doesnt work with lambda yet.
cost = zeros(1,iter);

for i = 1: iter
    % Get activation values and calculate gradients
    [a2, a3] = forwardProp(Xs, Thetas1, Thetas2);
    [Deltas2, Deltas3] = deltaNN(Thetas2, Ys, a2, a3);
    
    % Update Thetas2
    Thetas2 = Thetas2 - alpha * Deltas3;
    
    % Get total Delta, replicate matrix for matrix wise subtraction
    Deltas2 = sum(Deltas2, 2);
    Deltas2 = Deltas2 / size(Deltas2, 2);
    Deltas2 = repmat(Deltas2, 1,size(Thetas1, 2));
    
    % Update Thetas1
    Thetas1 = Thetas1 - alpha * Deltas2;
    
    cost(i) = costNN(Xs, Thetas1, Thetas2, Ys, lambda);
end

% Get predictions with trained parameters and print
[a2, a3] = forwardProp(Xs, Thetas1, Thetas2);
output_vs_y = [a3', Ys]

plot(cost);
clear    