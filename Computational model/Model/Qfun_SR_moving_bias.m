function [Q, bias] = Qfun_SR_moving_bias(model, memory, subject_data)
    N = length(subject_data);
    Q = zeros(N, 9, 2);
    bias = zeros(N, 5);

    for ind = 1:N
        is_changed = subject_data{ind, 1};
        goal_cond = subject_data{ind, 2};
        episode = subject_data{ind, 3};
        
        if is_changed == 1
            newW = memory.return_w(goal_cond);
            model = model.change_w(newW);
        end
        
        Q(ind,:,:) = model.Q;
        bias(ind,:,:) = model.bias;
        
        % Update w and adjust w matrix in memory
        model = model.update(episode);
        memory = memory.save_w_vector(goal_cond, model.w);
    end
    
    % save('../Result/model', 'model', 'memory', 'Q', 'bias');
end

