function [state, choice] = make_step_list(subject_data)
    num_trial = length(subject_data);
    num_step = num_trial * 3;
    state = zeros(num_step, 1);
    choice = zeros(num_step, 1);
    
    ind = 1;
    for i = 1:num_trial
        episode = subject_data{i, 3};
        for j = 1:3
            state(ind, 1) = episode(j, 1);
            choice(ind, 1) = episode(j, 2);
            ind = ind + 1;
        end
    end
end

