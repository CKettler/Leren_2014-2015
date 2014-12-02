%%% Leren Practicum # 5
%%% Thomas Meijers (10647023)
%%% December 2014

%% Q1.a - Naive Bayes Classifier

% Load and read data
load('digit_dataset.mat');
[Xs_trn, Ys_trn] = read_dataset(digits1231);
[Xs_tst, Ys_tst] = read_dataset(digits1232);

Xs_tr1 = Xs_trn(Ys_trn == 1,:);
Xs_tr2 = Xs_trn(Ys_trn == 2,:);
Xs_tr3 = Xs_trn(Ys_trn == 3,:);

Xs_tr1_mu = mean(Xs_tr1);
Xs_tr2_mu = mean(Xs_tr2);
Xs_tr3_mu = mean(Xs_tr3);

%% Q1.b - Analyzing wrong predictions

%% Q2.a

%% Q2.b

