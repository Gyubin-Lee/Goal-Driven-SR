%%
clear;
close all;

%% load preprocessed behavior data
addpath('../Model', '../Memory', '../Data');
load('SBJ_preprocessed', 'SBJ_preprocessed');

%% main routine
num_subject = length(SBJ_preprocessed);
subject_list = 27:27; %num_subject;
result_cell = cell(1, length(subject_list));

f = waitbar(0, 'Starting...');
pause(1);
tmp = 0;
for sub = subject_list
    task_string = ['Fitting in progress... (', num2str(tmp+1), '/', num2str(length(subject_list)), ')'];
    waitbar(tmp/length(subject_list), f, task_string);
    disp('-----------------------------------');
    
    subject_data = SBJ_preprocessed{sub};

    fitfunction = @(theta) (-1)*sum(custom_softmax(theta, subject_data, @Qfun_GoalDriven_SR_pre, @Qfun_SR_MF_arb));
    
    options = optimoptions('patternsearch'...
        ,'StepTolerance', 2e-6...
        ,'FunctionTolerance', 2e-3...
        ,'MeshTolerance', 5e-7...
        ,'MaxIterations', 500);

    num_fitting = 4;
    % [alpha_TD  alpha_SR  gamma_SR  alpha_MF  weight  beta]
    lb = [1e-4 1e-4 1e-4 1e-4 0.10 1];
    ub = [0.60 0.60 1.00 0.60 1.00 100];
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
    % save(['../Data/SR_MF_arb(1)-' date], 'result_cell');
    tmp = tmp + 1;
end

waitbar(1, f, 'Finished!');
pause(1);
close(f);