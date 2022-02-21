load('SBJ_structure_1027_for_revision.mat', 'SBJ3');

num_subject = 63;
SBJ_preprocessed = cell(1, num_subject);

for s = 1:num_subject
    sub_struct = struct;
    subject_data = SBJ3{s};
    
    % Preprocessing pre-train session behavior data
    
    num_pre_train_trial = length(subject_data.HIST_behavior_info_pre{1});
    
    episode_list_train = cell(num_pre_train_trial, 5); % for each row, col 1: is_changed, col 2: goal condition, col 3: episode(3X4)
    prev_goal_cond = 0;

    for ind = 1:num_pre_train_trial
        behavior_info = subject_data.HIST_behavior_info_pre{1};
        
        info = behavior_info(ind, :);
        info_crop = info([4 5 6 7 8 16 18]); %S1, S2, S3, A1, A2, r, goal_state
        [is_changed, new_goal_cond, episode] = make_episode(info_crop, prev_goal_cond);
        episode_list_train{ind, 1} = is_changed;
        episode_list_train{ind, 2} = new_goal_cond;
        episode_list_train{ind, 3} = episode;
        episode_list_train{ind, 4} = info_crop(3);
        episode_list_train{ind, 5} = info_crop(6);

        prev_goal_cond = new_goal_cond;      
    end
    
    sub_struct.pre_train_session = episode_list_train;
    sub_struct.num_pre_train_trial = num_pre_train_trial;
    
    % Calculate total number of trials
    num_sessions = 4; %fix to first 4 sessions (original) length(subject_data.HIST_behavior_info);
    
    episode_list_main = cell(1, 5); % for each row, col 1: is_changed, col 2: goal condition, col 3: episode(3X4)
    ind = 1;
    for i = 1:num_sessions
        behavior_info = subject_data.HIST_behavior_info{i};
        num_trial = length(behavior_info);
        for j = 1:num_trial
            info = behavior_info(j, :);
            info_crop = info([4 5 6 7 8 16 18]); %S1, S2, S3, A1, A2, r, goal_state
            [is_changed, new_goal_cond, episode] = make_episode(info_crop, prev_goal_cond);
            episode_list_main{ind, 1} = is_changed;
            episode_list_main{ind, 2} = new_goal_cond;
            episode_list_main{ind, 3} = episode;
            episode_list_main{ind, 4} = info_crop(3);
            episode_list_main{ind, 5} = info_crop(6);
            
            prev_goal_cond = new_goal_cond;          
            ind = ind + 1;
        end
    end
    
    sub_struct.main_session = episode_list_main;
    sub_struct.num_main_trial = ind - 1;
    
    SBJ_preprocessed{1, s} = sub_struct;
end

save('SBJ_preprocessed', 'SBJ_preprocessed');

function [S1, S2, S3, A1, A2, r, goal_state] = parse_info_crop(info_crop)
    S1 = info_crop(1);
    S2 = info_crop(2);
    S3 = info_crop(3);
    A1 = info_crop(4);
    A2 = info_crop(5);
    r = info_crop(6);
    goal_state = info_crop(7);
end

function goal_condition = state_to_condition(goal_state)
    if goal_state == 6
        goal_condition = 1;
    elseif goal_state == 7
        goal_condition = 2;
    elseif goal_state == 8
        goal_condition = 3;
    else
        goal_condition = 4;
    end
end

function [is_changed, new_goal_cond, episode] = make_episode(info_crop, prev_goal_cond)
    [S1, S2, S3, A1, A2, r, goal_state] = parse_info_crop(info_crop);
    new_goal_cond = state_to_condition(goal_state);

    is_changed = 0;
    if prev_goal_cond ~= new_goal_cond && prev_goal_cond ~= 0
        is_changed = 1;
    end
    
    step_1 = [S1 A1 0 S2];
    step_2 = [S2 A2 0 S3];
    fake_step = [S3 1 r S3+4];
    episode = [step_1;step_2;fake_step];
end