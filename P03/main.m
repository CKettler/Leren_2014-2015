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
iter = 100
alpha = 0.1;
lambda = 1;
y = 1;

% Update parameters, print cost (plot done within compareClasses)
[Thetas_new, Ys_new] = compareClasses(Thetas, alpha, Xs, Ys, iter, y, lambda);
cost = costLogRegularized(Thetas_new, Xs, Ys_new, lambda)

clear

%% Q1.b



%% Q2.a

% Data (Q1.a):
Xs = [1 1 1 1; 10 7 5 2; 4 3 4 3; 4 3 2 1]';
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
Thetas1 = rand(input1, hidden2 - 1) * (2 * epsilon) - epsilon;
Thetas2 = rand(hidden2, output3) * (2 * epsilon) - epsilon;

% Forward propagation
[a2, a3] = forwardProp(Xs, Thetas1, Thetas2);

% Print output vs Y values
output_vs_y = [a3, Ys]

clear

%% Q2.b

% Will do c, as it's 1 vs 4 points and a complete network incorporates
% error backpropagation

%% Q2.c

% Data (Q1.a):
Xs = [1 1 1 1; 10 7 5 2; 4 3 4 3; 4 3 2 1]';
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
Thetas1 = rand(input1, hidden2 - 1) * (2 * epsilon) - epsilon;
Thetas2 = rand(hidden2, output3) * (2 * epsilon) - epsilon;

% Variables
alpha = 0.1;
iter = 100;
lambda = 1;
cost = zeros(1, iter);

for i = 1: iter
    [a2, a3] = forwardProp(Xs, Thetas1, Thetas2);
    deltas = deltaNN(Thetas1, Thetas2, Ys, a2, a3);
end



