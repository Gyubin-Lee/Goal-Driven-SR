%%
clear;
close all;

%%
addpath("../Data", "../Env", "../Memory", "../Model");

%% useful variables and hyperparameters
uncertainty_low = 0.9;
uncertainty_high = 0.5;

num_trial = 1000;
num_state = 9;
num_action = 2;
alpha_TD = 0.01;
alpha_SR = 0.1;
gamma = 0.9;
beta = 10;
risk_factor = 0;

%% initialize task environment and model
prob_uncertainty = uncertainty_high;

uncertainty_period = 1000;
goal_cond_period = 1000;
specific_cond_num = 0;

task_env = ENV_Lee(prob_uncertainty, uncertainty_period, goal_cond_period, specific_cond_num);
model = model_SR_Distributional_Lee(alpha_TD, alpha_SR, gamma, risk_factor, beta);

%% initiallize memory for memory replaying
% memory = memory_replay_GoalDriven();

%% variables for analyzing result
acc_reward = 0;
acc_reward_list = zeros(1,num_trial);

acc_reward_specific = zeros(1, 2); %col1 for low uncertainty, col2 for high uncertainty
specific_cnt = zeros(1, 2);
hit_specific_cnt = zeros(1, 2);
optimal_action_specific_cnt = zeros(1, 2);

acc_reward_flexible = zeros(1, 2); %col1 for low uncertainty, col2 for high uncertainty
flexible_cnt = zeros(1, 2);
hit_flexible_cnt = zeros(1, 2);
optimal_action_flexible_cnt = zeros(1, 2);

oa_structure = load('optimal_action.mat');
optimal_action_matrix = oa_structure.optimal_action;

%% do task
f = waitbar(0, 'Starting...');
pause(1);
episode_list = nan(num_trial*3, 6);

for i = 1:num_trial
    task_string = ['Doing task... (', num2str(i), '/', num2str(num_trial), ')'];
    waitbar(i/num_trial, f, task_string);
    
    task_env = task_env.start_new_trial();
    [task_env, goal_cond_changed] = task_env.update_env();
 
    % memory replay for changed_goal condition
    % if goal_cond_changed == 1
        % model = model.change_w(memory.return_w(task_env.current_goal_cond));
        % change_string = ['trial ', num2str(i), ': goal condition is changed to ', num2str(task_env.current_goal_cond)];
        % disp(change_string);
    % end
    
    is_terminate = 0;
    episode = [];
    pval_list = [];
    while is_terminate == 0
        cur_state_ind = task_env.current_state_ind;
        pval = model.return_action_prob(cur_state_ind, task_env.current_goal_cond);
        action = model.return_action(cur_state_ind, task_env.current_goal_cond);
        
        % update result
        uncertainty_ind = return_uncertainty_index(task_env.prob_uncertainty);
        if cur_state_ind < 6
            optimal_action = optimal_action_matrix(task_env.current_goal_cond, uncertainty_ind, cur_state_ind);
            if action == optimal_action
                if task_env.current_goal_cond < 4
                    optimal_action_specific_cnt(uncertainty_ind) = optimal_action_specific_cnt(uncertainty_ind) + 1;
                else
                    optimal_action_flexible_cnt(uncertainty_ind) = optimal_action_flexible_cnt(uncertainty_ind) + 1;
                end
            end        
        end

        [task_env, reward, is_terminate] = task_env.transition(action);
        episode = [episode;cur_state_ind action reward task_env.current_state_ind is_terminate];
        pval_list = [pval_list;pval(action)];
    end
    episode_list(3*(i-1)+1:3*i, 1:5) = episode;
    episode_list(3*(i-1)+1:3*i, 6) = pval_list;
    
    % update H and w matrix and append episode to the memory
    model = model.update(episode);
    % memory = memory.save_w_vector(task_env.current_goal_cond, model.w);
    
    % update result
    acc_reward = acc_reward + reward;
    if task_env.current_goal_cond < 4
        specific_cnt(uncertainty_ind) = specific_cnt(uncertainty_ind) + 1;
        acc_reward_specific(uncertainty_ind) = acc_reward_specific(uncertainty_ind) + reward;
        if reward > 0
            hit_specific_cnt(uncertainty_ind) = hit_specific_cnt(uncertainty_ind) + 1;
        end
    else
        flexible_cnt(uncertainty_ind) = flexible_cnt(uncertainty_ind) + 1;
        acc_reward_flexible(uncertainty_ind) = acc_reward_flexible(uncertainty_ind) + reward;
        if reward > 0
            hit_flexible_cnt(uncertainty_ind) = hit_flexible_cnt(uncertainty_ind) + 1;
        end
    end
    acc_reward_list(i) = acc_reward;
end

waitbar(1, f, 'Finished!');
pause(1);
close(f);

%% Show result
mean_reward_value = acc_reward/num_trial;
mean_reward_value_specific = acc_reward_specific./specific_cnt;
mean_reward_value_flexible = acc_reward_flexible./flexible_cnt;
hit_rate_specific = hit_specific_cnt./specific_cnt;
hit_rate_flexible = hit_flexible_cnt./flexible_cnt;
optimal_action_specific = 0.5*optimal_action_specific_cnt./specific_cnt;
optimal_action_flexible = 0.5*optimal_action_flexible_cnt./flexible_cnt;

disp(['Mean reward value(total): ', num2str(mean_reward_value)]);
disp(['Mean reward value(specific): ', num2str(mean_reward_value_specific(1)), ' ', num2str(mean_reward_value_specific(2))]);
disp(['Mean reward value(flexible): ', num2str(mean_reward_value_flexible(1)), ' ', num2str(mean_reward_value_flexible(2))]);
disp(['Hit rate(specific): ', num2str(hit_rate_specific(1)), ' ', num2str(hit_rate_specific(2))]);
disp(['Hit rate(flexible): ', num2str(hit_rate_flexible(1)), ' ', num2str(hit_rate_flexible(2))]);
disp(['Optimal action(specific): ', num2str(optimal_action_specific(1)), ' ', num2str(optimal_action_specific(2))]);
disp(['Optimal action(flexible): ', num2str(optimal_action_flexible(1)), ' ', num2str(optimal_action_flexible(2))]);

figure;
plot(acc_reward_list);

X = categorical({'specific', 'flexible'});
X = reordercats(X,{'specific','flexible'});

figure;
h = bar(X, [mean_reward_value_specific;mean_reward_value_flexible]);
ylabel('Mean reward value');
legend(h, {'Low uncertainty', 'High uncertainty'}, 'Location', 'northwest');

figure;
h = bar(X, [hit_rate_specific;hit_rate_flexible]);
ylabel('Hit rate');
legend(h, {'Low uncertainty', 'High uncertainty'}, 'Location', 'northwest');

figure;
h = bar(X, [optimal_action_specific;optimal_action_flexible]);
ylabel('Proportion of optimal choices');
legend(h, {'Low uncertainty', 'High uncertainty'}, 'Location', 'northwest');

%% save model variables
Q = model.Q;
w = model.w;
H = model.H;
save('model', 'Q', 'w', 'H');

%% helper function
function ind = return_uncertainty_index(uncertainty)
    if uncertainty == 0.9
        ind = 1;
    else
        ind = 2;
    end
end