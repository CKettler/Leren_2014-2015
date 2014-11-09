data = csvread('digits123.csv');
Xs = data(:,1:(end-1));
Xs = [ones(length(Xs),1 ), Xs];
Ys = data(:,end); % Extract all Y values
n = size(Xs);
features_count = n(2);

% Start thetas
Thetas = ones(1, features_count) / 1000;

% how much numbers do we want to predict
% -> this also indicates how much will be randomly
% removed from the dataset
predict_count = 400;

% generate random indicies into dataset
indices = randperm(n(1));

indices = indices([1:predict_count]);

% remove random rows from feature set and answer set
Xs_without = Xs;
Xs_without(indices,:) = [];
Ys_without = Ys;
Ys_without(indices,:) = [];

% train our algoritm
[NewThetas1] = compareClasses(Thetas, 0.03, Xs_without, Ys_without, 1000, 1);
[NewThetas2] = compareClasses(Thetas, 0.03, Xs_without, Ys_without, 1000, 2);
[NewThetas3] = compareClasses(Thetas, 0.03, Xs_without, Ys_without, 1000, 3);

model = [NewThetas1; NewThetas2; NewThetas3];

% get the rows that we have extracted earlier on. We will predict them now
numbers = Ys(indices);
features = Xs(indices,:);

% counter for correct answers
correct=0;
for i=1:predict_count
    % target number
    number = numbers(i);
    
    % get feature for target number
    xs = features(i,:);
    
    % run model
    h1 = hypLogistic(model(1,:), xs);
    h2 = hypLogistic(model(2,:), xs);
    h3 = hypLogistic(model(3,:), xs);
    
    m = max([h1 h2 h3]);
    
    if (m == h1) 
        pred = 1;
    end
    if (m == h2)
        pred = 2;
    end
    if (m == h3)
        pred = 3;
    end
    
    if (pred == number)
        correct = correct + 1;
    end
    
    fprintf('predicted: %d, should be: %d\n', pred, number);
end

fprintf('accuracy: %f\n', (correct / predict_count * 100));