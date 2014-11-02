%%% Leren Practicum #1
%%% Thomas Meijers
%%% 10647023

function lin_reg_univariate() 

    % Q1: LOAD DATA MATRICES
    data_houses = csvread('housesRegr.csv',1,0);
    data_assignment = [5,6;5,6;3,10];

    house_bedrooms = data_houses(:,2);
    house_bathrooms = data_houses(:,3);
    house_size = data_houses(:,4);
    house_price = data_houses(:,5);


    % Q2: PLOT DATA
    figure('name', 'Q2: X = Size vs. Y = Prize');
    plot(house_size, house_price, '.');

    % Q3: GRADIENT 
    g = gradientVector(0, 1, house_size, house_price);

end % End main function
