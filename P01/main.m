%%% Leren Practicum #1
%%% Thomas Meijers
%%% 10647023

%% Q1: LOAD DATA MATRICES (ESSENTIAL FOR RUNNING OTHER SECTIONS)

% Read provided data
data_houses = csvread('housesRegr.csv',1,0);
data_assignment = [5,6;5,6;3,10];

% Store seperate data in vectors
house_bedrooms = data_houses(:,2);
house_bathrooms = data_houses(:,3);
house_size = data_houses(:,4);
house_price = data_houses(:,5);


%% Q2: PLOT DATA

% Plot each variable seperately against cost
figure('name', 'Question 2 - plotting data');

subplot(3, 3, 4);
plot(house_bedrooms, house_price, '.');
xlabel('bedrooms');
ylabel('price');

subplot(3, 3, 5);
plot(house_bathrooms, house_price, '.');
xlabel('bathrooms');
ylabel('price');

subplot(3, 3, 6);
plot(house_size, house_price, '.');
xlabel('size (sq.ft.)');
ylabel('price');

%% Q3: GRADIENT 

% See the following files:
%   - gradientVector.m
%   - gradientIter.m

%% Q4: UPDATE RULE

% See the following file:
%   - updateParam.m

%% Q5: COST

% See the following files:
%   - costVector.m
%   - costIter.m

%% Q6: PLOTTING REGRESSION

figure('name','Q6 - Plotting regression');

% PLOT BEDROOMS VS PRICE

% Set initial parameters
theta_0 = 0;
theta_1 = 1;
alpha = 0.1;
max_iter = 100;

[theta_0, theta_1] = bestParam(theta_0, theta_1, alpha, house_bedrooms, ...
                               house_price, max_iter);
prediction = theta_0 + theta_1 * house_bedrooms;
subplot(3, 3, 4);
hold all;
theta_0
theta_1
plot(house_bedrooms, house_price, '.');
plot(house_bedrooms, prediction, '-', 'linewidth', 2);
xlabel('bedrooms');
ylabel('price');
hold off;

% PLOT BATHROOMS VS PRICE

% Set initial parameters
theta_0 = 0;
theta_1 = 1;
alpha = 0.1;
max_iter = 100;

[theta_0, theta_1] = bestParam(theta_0, theta_1, alpha, house_bathrooms, ...
                               house_price, max_iter);
prediction = theta_0 + theta_1 * house_bathrooms;
subplot(3, 3, 5);
hold all;
theta_0
theta_1
plot(house_bathrooms, house_price, '.');
plot(house_bathrooms, prediction, '-', 'linewidth', 2);
xlabel('bathrooms');
ylabel('price');
hold off;

% PLOT SIZE VS PRICE

% Set initial parameters
theta_0 = 0;
theta_1 = 1;
alpha = 0.0000001;
max_iter = 100;

[theta_0, theta_1] = bestParam(theta_0, theta_1, alpha, house_size, ...
                               house_price, max_iter);
prediction = theta_0 + theta_1 * house_size;
subplot(3, 3, 6);
hold all;
theta_0
theta_1
plot(house_size, house_price, '.');
plot(house_size, prediction, '-', 'linewidth', 2);
xlabel('size');
ylabel('price');
hold off;

%% Q7: BEST PRICE PREDICTING VARIABLE

% Find optimal parameters per variable

% BEDROOMS
% Set initial parameters
theta_0 = 0;
theta_1 = 1;
alpha = 0.1;
max_iter = 100;

[theta_0, theta_1] = bestParam(theta_0, theta_1, alpha, house_bedrooms, ...
                               house_price, max_iter);
cost_bedrooms = costVector(theta_0, theta_1, house_bedrooms, house_price);
                           
% BATHROOMS
% Set initial parameters
theta_0 = 0;
theta_1 = 1;
alpha = 0.1;
max_iter = 100;

[theta_0, theta_1] = bestParam(theta_0, theta_1, alpha, house_bathrooms, ...
                               house_price, max_iter);
cost_bathrooms = costVector(theta_0, theta_1, house_bathrooms, house_price);
                           
% SIZE
% Set initial parameters
theta_0 = 0;
theta_1 = 1;
alpha = 0.0000001;
max_iter = 100;

[theta_0, theta_1] = bestParam(theta_0, theta_1, alpha, house_size, ...
                               house_price, max_iter);
cost_size = costVector(theta_0, theta_1, house_size, house_price);

fprintf('\nCost bedrooms: %d\nCost bathrooms: %d\nCost size: %d\n', ...
        cost_bedrooms, cost_bathrooms, cost_size);
fprintf('\nThe lowest of the above three is the best predictor, i.e. size\n');

%% Q8: FINDING BEST ALPHA AND MAX_ITER

% PICK ONE DATASET
%data = house_bedrooms;
%data = house_bathrooms;
data = house_size;

% Initalize parameters
max_iter = [10, 100, 1000, 10000];
best_cost = inf;
alpha = linspace(100, 1, 100) * 10e-3;

if (data == house_size) 
    % House size requires a very small learning rate
    alpha = linspace(10000, 1, 100) * 10e-10;
end % end if

for i = 1:length(max_iter)
    for j = 1:length(alpha)
        theta_0 = 0;
        theta_1 = 1;
        [theta_0, theta_1] = bestParam(theta_0, theta_1, alpha(j), ...
                                       data, house_price, max_iter(i));
        cost = costVector(theta_0, theta_1, data, house_price);
        
        if (cost < best_cost)
            best_cost = cost;
            best_alpha = alpha(j);
            best_iter = max_iter(i);
        end % end if 
        
    end % end for j
end % end for i

fprintf('\nFor given dataset, best alpha: %d, best iter: %d\n', ...
        best_alpha, best_iter);

% ANSWERS:
% Test bedrooms vs price:
% "For given dataset, best alpha: 1.700000e-01, best iter: 10000"
% Test bathrooms vs price:
% "For given dataset, best alpha: 2.300000e-01, best iter: 1000"
% Test size vs price:
% "For given dataset, best alpha: 5.060000e-07, best iter: 10000"

% As can be seen, a high iteration will give better results. Of course,
% if the optimal parameters have been found, increasing iteration will only
% increase the required time (Bathroom vs. price).
% For alpha, this method gives a good approximation for the optimal
% learning rate. Optimal alpha was already found for bed- and bathrooms
% with 10 iterations, for size this was at 100 iterations. 


