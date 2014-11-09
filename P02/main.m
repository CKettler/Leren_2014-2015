%%% Leren Practicum #2
%%% Markus Pfundstein (10452397)
%%% Thomas Meijers (10647023)
%%% November 2014

%% Q1 && Q2

% Question 1

Iterations = 10;
% we can use a high alpha as we normalize the data
Alpha = 20;

Thetas1 = [0.5, 0.5, 0.5, 0.5];

[Best_Thetas, Xs, Ys] = multiple(Thetas1, Alpha, Iterations);

% calculate costs. both from start thetas and from best params that
% are returned from multiple.
Hypos1 = hypLinear(Thetas1, Xs);
Hypos2 = hypLinear(Best_Thetas, Xs);

cost1 = costLinear(Hypos1', Ys)
cost2 = costLinear(Hypos2', Ys)

% Question 2

% 8 thetas now. We have 4 features, we add 4 squares, hence 8 in total
ThetasSquares1 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

% runs multiple but also uses squares of features
[Best_Thetas_Squares, Xs_Squares, Ys_Squares] = multipleSquares(ThetasSquares1, Alpha, Iterations);

% calculate costs. both from start thetas and from best params that
% are returned from multiple.
HyposSquares1 = hypLinear(ThetasSquares1, Xs_Squares);
HyposSquares2 = hypLinear(Best_Thetas_Squares, Xs_Squares);

costSquares1 = costLinear(HyposSquares1', Ys_Squares)
costSquares2 = costLinear(HyposSquares2', Ys_Squares)

% Observation:

% The initial cost with all thetas = 0.5, is lower on the non square
% version. This is because in the square version, the values are spread out.
% What is interesting is, that if we run only 1 iteration, we can see that
% the square version converges slower to a lower cost than the regular
% version.
% Nevertheless, both reach the same minimum cost. Both reach 0.0013 after
% only 6 iterations. After 10 iterations both have a cost of 0.0012.
% Another interesting iteration is, that if we run the algoritm with 3000
% iterations, the square version will converge to a cost of 0.0010, while
% the regular version will converge to 0.0011

%% Question 1.5

% find best alpha and iteration rate

data = csvread('housesRegr.csv', 1, 0);
Xs = data(:,2:(end-1));
Xs = [ones(length(Xs),1 ), Xs];
Ys = data(:,end); % Extract all Y values

% normalize data
Xs_size = size(Xs);
for i=1:Xs_size(2)
    Xs(:,i) = normalizeColumn(Xs(:,i));
end
% also normalize price
Ys = normalizeColumn(Ys);

%Params = bestParamLinear(Thetas, alpha, iterations, Xs, Ys);

% Initalize parameters
max_iter = [10, 100, 1000];
alpha = linspace(5, 30, 20);
best_alpha = 0;
best_cost = inf;
best_iter = 0;

for i = 1:length(max_iter)
    for j = 1:length(alpha)
        Thetas = [0.5, 0.5, 0.5, 0.5];

        NewThetas = bestParamLinear(Thetas, alpha(j), max_iter(i), Xs, Ys);
        
        h = hypLinear(NewThetas, Xs);
        

        % Use Ys or newY (OvA class vector)?!
        cost = costLinear(h', Ys);

        if (cost < best_cost)
            best_cost = cost;
            best_alpha = alpha(j);
            best_iter = max_iter(i);
        end % end if 

    end % end for j
end % end for i


best_cost
best_alpha
best_iter

clear

%% Q3.2: GRADIENT

% See: gradientLogistic.m

%% Q3.3: UPDATE PARAMETERS

% See: updateParamLogistic.m & bestParamLogistic.m

%% Q3.4: COST

% See: costLogistic.m

%% Q3.5: COMPARING CLASSES (PAIR WISE)

% See: compareClasses.m
% Handwritten number data

%

logreg

clear

%% Q3.6: FINDING ALPHA AND NR OF ITERATIONS
% NOTE: Instead of making a function called logreg the main computation is
% done in this section. This due to resembling the assignment structure
% more and because it feels more intuitive.

% Handwritten number data

data = csvread('digits123.csv');
Xs = data(:,1:(end-1));
Xs = [ones(length(Xs),1 ), Xs];
Ys = data(:,end); % Extract all Y values
n = size(Xs);
n = n(2);
Yclasses = unique(Ys);

% Initalize parameters
max_iter = [10, 100, 300];
alpha = linspace(10, .1, 20) * 10e-03;
best_alpha = zeros(1, length(Yclasses));
best_cost = Inf(1, length(Yclasses));
best_iter = zeros(1, length(Yclasses));

for k = 1:length(Yclasses)
    for i = 1:length(max_iter)
        for j = 1:length(alpha)
            Thetas = 0.001 * ones(1, n);
            
            [Thetas, Ynew] = compareClasses(Thetas, alpha(j), Xs, ...
                Ys, max_iter(i), Yclasses(k));
            
            % Use Ys or newY (OvA class vector)?!
            cost = costLogistic(Thetas, Xs, Ynew);
            
            if (cost < best_cost(k))
                best_cost(k) = cost;
                best_alpha(k) = alpha(j);
                best_iter(k) = max_iter(i);
            end % end if 

        end % end for j
    end % end for i
end % end for k

best_cost
best_alpha
best_iter

clear
