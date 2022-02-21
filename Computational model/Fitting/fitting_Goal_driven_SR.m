%%
clear;
close all;

%% load preprocessed behavior data
addpath('../Model', '../Memory', '../Data');
load('SBJ_preprocessed', 'SBJ_preprocessed');

%% main routine
num_subject = length(SBJ_preprocessed);
subject_list = 1:63; %num_subject;
result_cell = cell(1, length(subject_list));
% load(['./Data/result_fitting_' date], 'result_cell_main');

f = waitbar(0, 'Starting...');
pause(1);
tmp = 0;
for sub = subject_list

    task_string = ['Fitting in progress... (', num2str(tmp+1), '/', num2str(length(subject_list)), ')'];
    waitbar(tmp/length(subject_list), f, task_string);
    disp('-----------------------------------');
    
    subject_data = SBJ_preprocessed{sub};
    fitfunction = @(theta) (-1)*sum(custom_softmax(theta, subject_data, @Qfun_GoalDriven_SR));
    
    options = optimoptions('patternsearch'...
        ,'StepTolerance', 2e-6...
        ,'FunctionTolerance', 1e-5...
        ,'MeshTolerance', 1e-7...
        ,'MaxIterations', 500);

    % [alpha_TD  alpha_SR  df_SR  beta  bias]
    lb = [0.01 0.01 0.01 1.00 -0.80];
    ub = [0.60 0.60 1.00 80.0  0.80];
    
    num_fitting = 128;
    result = zeros(num_fitting, length(lb) + 2);
    num_trial = subject_data.num_main_trial;
    
    parfor ind = 1:num_fitting
        seed = make_seed_pre(lb, ub);   
        [x, fval] = patternsearch(fitfunction, seed, [], [], [], [], lb, ub, options);
        BIC = length(seed)*log(num_trial*2) + 2*fval;
        result(ind, :) = [x return_avg_predictive_accuracy(fval, num_trial) BIC];
    end
    
    % Save result
    result_cell{1, sub} = result;
    save(['../Data/SR_GoalDriven_bias_No_Pre-' date], 'result_cell');
    
    tmp = tmp + 1;
end

waitbar(1, f, 'Finished!');
pause(1);
close(f);
