classdef memory_replay_Dyna
    
    properties
        step_list;
        length = 0;
        replay_length;
    end
    
    methods
        function obj = memory_replay_Dyna(num_trial, replay_length_)
            obj.step_list = nan(num_trial*3, 6); %[goal_cond s a r s_prime RPE]
            % obj.df_memory = df_memory_;
            obj.replay_length = round(20*replay_length_);
        end
        
        function weight = activation(obj, weight_)
            weight = atan(10*weight_);
        end
        
        function obj = add_episode(obj, goal_cond, episode, abs_RPE_list)
            % obj.step_list(1:obj.length, 6) = obj.df_memory * obj.step_list(1:obj.length, 6);
            len = obj.length + 3;
            obj.step_list(len-2:len, 1) = goal_cond * ones(3, 1);
            obj.step_list(len-2:len, 2:5) = episode;
            obj.step_list(len-2:len, 6) = abs_RPE_list;
            obj.length = len;
        end

        function pop_index = select_step(obj, ind)
            switch ind
                case 1
                    % use RPE
                    RPE_list = obj.step_list(1:obj.length, 6);
                    time_factor = (0.9).^fliplr([1:length(RPE_list)]);
                    weight_vector = obj.activation(RPE_list'.*time_factor);
                    pop_length = min(obj.length, obj.replay_length);
                    pop_index = randsample(1:obj.length, pop_length, true, weight_vector);
                otherwise
                    % recent #replay_length episodes will be popped
                    pop_length = min(obj.length, obj.replay_length);                    
                    pop_index = 1:1:pop_length;
                    pop_index = pop_index + (obj.length - pop_length);
            end

        end
        
        function model = replay(obj, model, goal_cond)
            pop_index = obj.select_step(1);
            % disp('------------------------');
            % disp(['goal cond: ' num2str(goal_cond)]);
            % disp(model.w);
               
            for ind = pop_index
                step = obj.step_list(ind, 2:5);

                new_state_ind = step(1, 4);
                if new_state_ind > 9
                    % adjust reward
                    if goal_cond == 4 % flexible
                        [~, reward] = state_to_reward(new_state_ind);
                        step(3) = reward;
                    else % specific
                        [goal_state, reward] = state_to_reward(new_state_ind);
                        if goal_cond == goal_state
                            step(3) = reward;
                        else
                            step(3) = 0;
                        end
                    end
                end

                model = model.update_w_by_step(step, 2);
            end
            % disp(model.w);
        end
    end
end

