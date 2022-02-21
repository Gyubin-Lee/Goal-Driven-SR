classdef memory_replay
    
    properties
        episode_list;
        length = 0;
        replay_length;
    end
    
    methods
        function obj = memory_replay(num_trial, replay_length_)
            obj.episode_list = cell(num_trial, 2);
            obj.replay_length = replay_length_;
        end
        
        function obj = append(obj, goal_cond, episode)
            len = obj.length + 1;
            obj.episode_list(len, 1) = {goal_cond};
            obj.episode_list(len, 2) = {episode};
            obj.length = len;
        end
        
        function weight_vector = make_weight_vector(obj, ind)
            len = obj.length;           
            switch ind
                case 1 % recent episode has more probability to be popped
                    weight_vector = ones(1, len);
                    for i = flip(1:len)
                        weight_vector(i) = weight_vector(i)*(obj.df_memory)^(len-i+1);
                    end
                    weight_vector = weight_vector/sum(weight_vector);
                otherwise % every episode has equal probability to be popped
                    weight_vector = ones(1, len);
                    weight_vector = weight_vector/len;           
            end
        end

        function pop_index = select_episode(obj, ind)
            switch ind
                case 0 % recent (replay_length) episodes will be popped
                    pop_length = min(obj.length, obj.replay_length);                    
                    pop_index = 1:1:pop_length;
                    pop_index = pop_index + (obj.length - pop_length);
                otherwise % making weight vector and use it for picking episodes
                    weight_vector = obj.make_weight_vector(ind);
                    pop_index = randsample([1:obj.length], obj.replay_length, true, weight_vector);
            end
        end
        
        function model = replay(obj, model, task_env)
            goal_condition = task_env.current_goal_cond;
            pop_index = obj.select_episode(0);
            model = model.initialize_w();
            
            for i = pop_index
                for j = 1:obj.length
                    episode = obj.episode_list{j, 2};
                    for k = 1:3
                        if episode(k, 4) > 9
                            % adjust reward
                            new_state = task_env.return_state(episode(k, 4));
                            if goal_condition == 4 % flexible
                                episode(k, 3) = new_state.reward;
                            else % specific
                                if goal_condition == new_state.color
                                    episode(k, 3) = new_state.reward;
                                else
                                    episode(k, 3) = 0;
                                end
                            end
                        end
                    end

                    model = model.update_w(episode);
                end
            end          
        end
        
        function model = replay_fitting(obj, model, goal_cond)
            pop_index = obj.select_episode(0);
            
            for i = pop_index
                for j = 1:obj.length
                    episode = obj.episode_list{j, 2};
                    for k = 1:3
                        new_state_ind = episode(k, 4);
                                                
                        if new_state_ind > 9
                            % adjust reward
                            if goal_cond == 4 % flexible
                                [~, reward] = state_to_reward(new_state_ind);
                                episode(k, 3) = reward;
                            else % specific
                                [goal_state, reward] = state_to_reward(new_state_ind);
                                if goal_cond == goal_state
                                    episode(k, 3) = reward;
                                else
                                    episode(k, 3) = 0;
                                end
                            end
                        end
                    end

                    model = model.update_w(episode);
                end
            end          
        end
    end
end

