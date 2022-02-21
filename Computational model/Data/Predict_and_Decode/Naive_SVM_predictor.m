%%
clear;
close all;

%%
warning('off','all')

%%
load("../data");
params = data.result_SR_bias.param;
ces_d = data.CES_D;

%%
N_PERM = 1000;
N_SUB = size(params, 1);
N_TEST_SUB = 6;
N_PARAM = size(params, 2);

result = nan(N_SUB, N_PERM);

f = waitbar(0, 'Starting...');
pause(1);

for iter = 1:N_PERM
    task_string = ['Training and Testing... (', num2str(iter), '/', num2str(N_PERM), ')'];
    waitbar((iter-1)/N_PERM, f, task_string);
    
    test_ind = randsample(1:N_SUB, N_TEST_SUB);
    train_ind = setdiff(1:N_SUB, test_ind);

    X_train = params(train_ind, :);
    X_test = params(test_ind, :);
    y_train = ces_d(train_ind, :);
    y_test = ces_d(test_ind, :);

    mdl = fitrsvm(X_train, y_train,...
        'Standardize', 1,...
        'kernelfunction', 'rbf',...
         'OptimizeHyperparameters', 'auto',...
         'HyperparameterOptimizationOptions',...
         struct('Verbose',0,'Useparallel',true,'MaxObjectiveEvaluations', 60, 'ShowPlots',false));

    result(test_ind, iter) = predict(mdl, X_test);
end

waitbar(1, f, 'Finished!');
pause(1);
close(f);

%%
save("Naive_SVM_result", "result");
 