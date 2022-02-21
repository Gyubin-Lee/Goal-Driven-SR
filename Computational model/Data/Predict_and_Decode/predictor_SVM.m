clear;
load('../data.mat');

CES_D = data.CES_D; % data.CES_D_GRADE;
params = data.result_arb.param; % data.result_SR_bias.param;
N_PERM = 1000;

result_cell = cell(1, N_PERM);
f = waitbar(0, 'Starting...');
pause(1);
tmp = 0;
for perm = 1:N_PERM
    task_string = ['Permutation in progress... (', num2str(tmp+1), '/', num2str(N_PERM), ')'];
    waitbar(tmp/N_PERM, f, task_string);
    
    test_ind = randsample(63, 6)';
    train_ind = setdiff(1:63, test_ind);

    CES_D_test = CES_D(test_ind, :);
    CES_D_train = CES_D(train_ind, :);
    params_test = params(test_ind, :);
    params_train = params(train_ind, :);

    % If hyperparameters are already optimized, skip this
    fitresult = optimize_hyperparams(params_train, CES_D_train);

    % Train
    kernel_scale = fitresult.XAtMinObjective.kernel_size;
    box_constraint = fitresult.XAtMinObjective.box_constraint;

    N_TRAIN_SUB = size(params_train, 1);
    N_PARAM = size(params_train, 2);
    N_PAIR = N_TRAIN_SUB * (N_TRAIN_SUB - 1);

    X_train = zeros(N_PAIR, N_PARAM * 2);
    y_train = zeros(N_PAIR, 1);
    c = 0;
    for i = 1:N_TRAIN_SUB
        for j = 1:N_TRAIN_SUB        
            if i ~= j
                c = c+1;
                X_train(c, :) = [params_train(i, :) params_train(j, :)];
                y_train(c, 1) = CES_D_train(i, 1) > CES_D_train(j, 1);
            end
        end
    end

    mdl = fitcsvm(gpuArray(X_train), gpuArray(y_train), ...        
         'kernelfunction','rbf',...
         'Standardize', true,...
         'KernelScale', kernel_scale, 'BoxConstraint', box_constraint);
     
    score_mdl = fitSVMPosterior(mdl);

    % Predicting
    result_perm = struct;
    N_TEST_SUB = size(params_test, 1);
    [y_test_L, y_test_R]  = deal(cell(2, N_TEST_SUB));

    [~, in] = sort(CES_D_train, 'ascend');
    params_train_aligned = params_train(in, :);

    % test params go left side
    for j = 1:N_TEST_SUB
        X_test_L = zeros(N_TRAIN_SUB, N_PARAM * 2);
        for i = 1:N_TRAIN_SUB
            X_test_L(i, :) = [params_test(j, :) params_train_aligned(i, :)];
        end
        [result, score] = predict(score_mdl, X_test_L);
        y_test_L{1, j} = result;
        y_test_L{2, j} = score;
    end

    % test params go right side
    for j = 1:N_TEST_SUB
        X_test_R = zeros(N_TRAIN_SUB, N_PARAM * 2);
        for i = 1:N_TRAIN_SUB
            X_test_R(i, :) = [params_train_aligned(i, :) params_test(j, :)];
        end
        [result, score] = predict(score_mdl, X_test_R);
        y_test_R{1, j} = result;
        y_test_R{2, j} = score;
    end
    
    result_perm.test_ind = test_ind;
    result_perm.val_acc = (-1) * fitresult.MinObjective;
    result_perm.y_test_L = y_test_L;
    result_perm.y_test_R = y_test_R;
    
    result_cell{1, perm} = result_perm;
    tmp = tmp + 1;
    
    save("predictor_SVM_result(arb)", "result_cell");
end

waitbar(1, f, 'Finished!');
pause(1);
close(f);


