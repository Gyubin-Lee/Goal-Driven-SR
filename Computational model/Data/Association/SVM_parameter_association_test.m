%%
clear;
close all;

%%
warning('off','all')

%%
load("../data.mat")
params = data.result_SR_bias.param;
ces_d = data.CES_D;

%%
c = 0;
N_SUB = size(params ,1);
N_PARAM = size(params, 2);
N_PAIR = N_SUB*(N_SUB-1);

X = zeros(N_PAIR, 2*N_PARAM);
y = zeros(N_PAIR,1);
Z = zeros(N_PAIR,2);

for i = 1:N_SUB
    for j = 1:N_SUB
        if (i~=j)
            c = c+1;
            X(c,:) = [params(i, :) params(j, :)];
            y(c,:) = ces_d(i) > ces_d(j);
            Z(c,:) = [i,j];
        end
    end
end

%%
p_train = sqrt(0.8);
N_train = round(N_SUB*p_train);
N_perm = 2000;
aoc_X = nan(N_PAIR,N_perm);
acc = nan(1, N_perm);
aoc_X_swap = nan(N_PAIR,N_perm, N_PARAM);
acc_swap = nan(N_PARAM, N_perm);
d_array = zeros(5, N_perm);

f = waitbar(0, 'Starting...');
pause(1);
tmp = 0;

for i = 1:N_perm
    task_string = ['Testing... (', num2str(i), '/', num2str(N_perm), ')'];
    waitbar((i-1)/N_perm, f, task_string);

    idxtrain = randperm(N_SUB,N_train);
    idxtest = setdiff(1:N_SUB,idxtrain);
    trainmask = (ismember(Z(:,1),idxtrain))&(ismember(Z(:,2),idxtrain));
    testmask = ~trainmask;

    X_train = X(trainmask,:);
    y_train = y(trainmask,1);
    X_test = X(testmask,:);
    
    params = hyperparameters('fitcsvm', X_train, y_train);
    params = params(1:2);
    params(1).Range = [1e-2, 5];
    params(2).Range = [1e-2, 10];

    mdl = fitcsvm(X_train, y_train,...
        'Standardize', 1,...
        'kernelfunction', 'rbf',...
         'OptimizeHyperparameters', params,...
         'HyperparameterOptimizationOptions',...
         struct('Verbose',0,'Useparallel',true,'MaxObjectiveEvaluations', 40, 'ShowPlots',false));
     
    y_test = predict(mdl,gpuArray(X_test));
    is_correct = (y_test == y(testmask,1));
    aoc_X(testmask,i) = is_correct;
    acc(i) = sum(is_correct)/length(is_correct);
     
    [X_test_swap1, X_test_swap2, X_test_swap3, X_test_swap4, X_test_swap5] = deal(X_test);
    X_test_swap1(:, [1, 6]) = X_test(:, [6, 1]);
    X_test_swap2(:, [2, 7]) = X_test(:, [7, 2]);
    X_test_swap3(:, [3, 8]) = X_test(:, [8, 3]);
    X_test_swap4(:, [4, 9]) = X_test(:, [9, 4]);
    X_test_swap5(:, [5, 10]) = X_test(:, [10, 5]);

    y_test_swap1 = predict(mdl,gpuArray(X_test_swap1));
    y_test_swap2 = predict(mdl,gpuArray(X_test_swap2));
    y_test_swap3 = predict(mdl,gpuArray(X_test_swap3));
    y_test_swap4 = predict(mdl,gpuArray(X_test_swap4));
    y_test_swap5 = predict(mdl,gpuArray(X_test_swap5));
    
    is_correct_swap1 = (y_test_swap1 == y(testmask,1));
    is_correct_swap2 = (y_test_swap2 == y(testmask,1));
    is_correct_swap3 = (y_test_swap3 == y(testmask,1));
    is_correct_swap4 = (y_test_swap4 == y(testmask,1));
    is_correct_swap5 = (y_test_swap5 == y(testmask,1));
    
    aoc_X_swap(testmask,i, 1) = is_correct_swap1;
    aoc_X_swap(testmask,i, 2) = is_correct_swap2;
    aoc_X_swap(testmask,i, 3) = is_correct_swap3;
    aoc_X_swap(testmask,i, 4) = is_correct_swap4;
    aoc_X_swap(testmask,i, 5) = is_correct_swap5;
    
    acc_swap(1, i) = sum(is_correct_swap1)/length(is_correct_swap1);
    acc_swap(2, i) = sum(is_correct_swap2)/length(is_correct_swap2);
    acc_swap(3, i) = sum(is_correct_swap3)/length(is_correct_swap3);
    acc_swap(4, i) = sum(is_correct_swap4)/length(is_correct_swap4);
    acc_swap(5, i) = sum(is_correct_swap5)/length(is_correct_swap5);
    
    for k = 1:5
        p_v = acc(i);
        p_v_swap = acc_swap(k, i);
        d_array(k, i) = log(p_v/(1-p_v)) - log(p_v_swap/(1-p_v_swap));
    end

end

waitbar(1, f, 'Finished!');
pause(1);
close(f);
%%
save("SVM_parameter_association_test_result", "d_array");
