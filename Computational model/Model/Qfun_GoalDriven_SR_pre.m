function [model, memory] = Qfun_GoalDriven_SR_pre(theta, pre_trian_data)
    N = length(pre_trian_data);
    model = model_SR_Standard_Lee(9, 2, theta(1), theta(2), 1, theta(3));
    memory = memory_replay_GoalDriven();
    
    for ind = 1:N
        is_changed = pre_trian_data{ind, 1};
        goal_cond = pre_trian_data{ind, 2};
        episode = pre_trian_data{ind, 3};
        
        if is_changed == 1
            newW = memory.return_w(goal_cond);
            model = model.change_w(newW);
        end
        
        % Update H, w and adjust w matrix in memory
        model = model.update(episode);
        memory = memory.save_w_vector(goal_cond, model.w);     
    end
end

