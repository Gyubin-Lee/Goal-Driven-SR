function Q = Qfun_SR_MF_arb(model_SR, memory, subject_data, theta)
    N = length(subject_data);
    Q = zeros(N, 9, 2);
    model_MF = model_MF_Lee(theta(4), 1, 40);

    for ind = 1:N
        is_changed = subject_data{ind, 1};
        goal_cond = subject_data{ind, 2};
        episode = subject_data{ind, 3};
        
        if is_changed == 1
            newW = memory.return_w(goal_cond);
            model_SR = model_SR.change_w(newW);
        end
        
        Q(ind,:,:) = theta(5) * model_SR.Q + (1 - theta(5)) * model_MF.Q;
        
        % Update w and adjust w matrix in memory
        model_SR = model_SR.update(episode);
        memory = memory.save_w_vector(goal_cond, model_SR.w);
        % Update MF model
        model_MF = model_MF.update(episode);
    end
    
    % save('../Result/model', 'model_SR', 'model_MF', 'memory', 'Q');
end

