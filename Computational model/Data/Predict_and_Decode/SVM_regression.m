clear;

load('data.mat');

CES_D = data.CES_D;
params = data.result_SR_bias.param;

test_ind = randsample(1:63, 6);
train_ind = setdiff(1:63, test_ind);

train_params = params(train_ind, :);
train_CES_D = CES_D(train_ind, :);
test_params = params(test_ind, :);
test_CES_D = CES_D(test_ind, :);

params = hyperparameters('fitrsvm', train_params, train_CES_D);
params = params(1:3);
params(1).Range = [1e-2, 2000];
params(2).Range = [1e-2, 100];

Mdl = fitrsvm(train_params, train_CES_D,...
    'Standardize', true,...
    'KernelFunction','rbf',...
    'OptimizeHyperparameters', params,...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName', 'expected-improvement-plus',...
    "UseParallel", true, ...
    'MaxObjectiveEvaluations', 500));

result = predict(Mdl, test_params);