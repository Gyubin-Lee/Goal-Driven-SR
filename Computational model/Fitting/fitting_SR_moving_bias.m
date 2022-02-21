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
    fitfunction = @(theta) (-1)*sum(custom_softmax(theta, subject_data, @Qfun_SR_moving_bias_pre, @Qfun_SR_moving_bias));
    
    options = optimoptions('patternsearch'...
        ,'StepTolerance', 1e-5...
        ,'FunctionTolerance', 5e-4...
        ,'MeshTolerance', 1e-7...
        ,'MaxIterations', 500);

    % [alpha_TD  alpha_SR  df_SR  alpha_Bias  beta]
    lb = [5e-4 5e-4 5e-4 5e-4 1.00];
    ub = [0.45 0.45 1.00 0.6 80.0];
    
    num_fitting = 96;
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
    % save(['../Data/SR_GoalDriven_moving_bias_pre(1)-' date], 'result_cell');
    
    tmp = tmp + 1;
end

waitbar(1, f, 'Finished!');
pause(1);
close(f);
