classdef ENV_Lee  
    % goal condition
    %   1: specific - R
    %   2: specific - B
    %   3: specific - Y
    %   4: flexible
    % prob_uncertainty
    %   0.9: low uncertainty
    %   0.5: high uncertainty
    
    properties
        current_state_ind = 1;
        current_goal_cond = 4;
        current_trial_num = 0;
        state_array = [State(0, 0, 1)... % state 10-13: S_t
                       State(0, 0, 2)...
                       State(0, 0, 3)...
                       State(0, 0, 4)...
                       State(0, 0, 5)...
                       State(0, 0, 6)...
                       State(0, 0, 7)...
                       State(0, 0, 8)...
                       State(0, 0, 9)...
                       State(40, 1, 10)...
                       State(20, 1, 11)...
                       State(10, 1, 12)...
                       State(0, 1, 13)];
        terminal_state_list = [10 11 12 13];
        prob_uncertainty, uncertainty_period, goal_cond_period, specific_cond_num;
    end
    
    methods
        function obj = ENV_Lee(prob_uncertainty_, uncertainty_period_, goal_cond_period_, specific_cond_num_)
            obj.prob_uncertainty = prob_uncertainty_;
            obj.uncertainty_period = uncertainty_period_;
            obj.goal_cond_period = goal_cond_period_;
            obj.specific_cond_num = specific_cond_num_;
        end
        
        function obj = start_new_trial(obj)
        % called when agent start new trial
            obj.current_trial_num = obj.current_trial_num + 1;
            obj.current_state_ind = 1;
        end
        
        function [obj, goal_cond_changed] = update_env(obj)
        % update task environment    
            
            % change uncertainty
            if mod(obj.current_trial_num, obj.uncertainty_period) == 0
                if obj.prob_uncertainty == 0.9
                    obj.prob_uncertainty = 0.5;
                else
                    obj.prob_uncertainty = 0.9;
                end
            end
            
            % change goal condition
            goal_cond_changed = 0;
            tmp = mod(obj.current_trial_num, obj.goal_cond_period);
            
            if tmp == 1 && obj.current_trial_num > 1
                obj.current_goal_cond = 1;
                goal_cond_changed = 1;
            elseif tmp == obj.specific_cond_num + 1
                obj.current_goal_cond = 2;
                goal_cond_changed = 1;
            elseif tmp == obj.specific_cond_num*2 + 1
                obj.current_goal_cond = 3;
                goal_cond_changed = 1;
            elseif tmp == obj.specific_cond_num*3 + 1
                obj.current_goal_cond = 4;
                goal_cond_changed = 1;
            end
            
            obj.current_goal_cond = 4;
            goal_cond_changed = 0;
            obj.prob_uncertainty = 0.9;
            
        end
        
        function [new_state, reward, is_terminal] = return_new_state(obj, left_state, right_state)
            if rand < obj.prob_uncertainty
                new_state = left_state;
            else
                new_state = right_state;
            end
            reward = obj.state_array(new_state).reward;
            is_terminal = obj.state_array(new_state).is_terminal;
        end
        
        function [obj, reward, is_terminal] = transition(obj, action)
            state = obj.current_state_ind;
            switch state
                case 1
                    if action == 1
                        [new_state, reward, is_terminal] = obj.return_new_state(2, 3);
                        obj.current_state_ind = new_state;
                    else
                        [new_state, reward, is_terminal] = obj.return_new_state(4, 5);
                        obj.current_state_ind = new_state;
                    end
                case 2
                    if action == 1
                        [new_state, reward, is_terminal] = obj.return_new_state(7, 8);
                        obj.current_state_ind = new_state;
                    else
                        [new_state, reward, is_terminal] = obj.return_new_state(8, 9);
                        obj.current_state_ind = new_state;
                    end
                case 3
                    if action == 1
                        [new_state, reward, is_terminal] = obj.return_new_state(8, 9);
                        obj.current_state_ind = new_state;
                    else
                        [new_state, reward, is_terminal] = obj.return_new_state(7, 9);
                        obj.current_state_ind = new_state;
                    end
                case 4
                    if action == 1
                        [new_state, reward, is_terminal] = obj.return_new_state(7, 6);
                        obj.current_state_ind = new_state;
                    else
                        [new_state, reward, is_terminal] = obj.return_new_state(6, 9);
                        obj.current_state_ind = new_state;
                    end
                case 5
                    if action == 1
                        [new_state, reward, is_terminal] = obj.return_new_state(7, 9);
                        obj.current_state_ind = new_state;
                    else
                        [new_state, reward, is_terminal] = obj.return_new_state(9, 6);
                        obj.current_state_ind = new_state;
                    end
                otherwise
                    new_state = obj.current_state_ind + 4;
                    is_terminal = obj.state_array(new_state).is_terminal;
                    obj.current_state_ind = new_state;
                    
                    reward = 0;
                    switch obj.current_goal_cond
                        case 1
                            if new_state == 10
                                reward = obj.state_array(new_state).reward;
                            end
                        case 2
                            if new_state == 11
                                reward = obj.state_array(new_state).reward;
                            end
                        case 3
                            if new_state == 12
                                reward = obj.state_array(new_state).reward;
                            end
                        otherwise
                            reward = obj.state_array(new_state).reward;
                    end
                        
            end
        end
    end
end

