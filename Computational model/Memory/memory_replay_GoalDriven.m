classdef memory_replay_GoalDriven
    
    properties
        w_save = zeros(4, 9); % (goal_cond, num_state)
    end
    
    methods
        function obj = memory_replay_GoalDriven()
        end
        
        function obj = save_w_vector(obj, goal_cond, newW)                       
            for ind = 1:4
                if ind == goal_cond
                    obj.w_save(goal_cond, :) = newW;
                end
            end
        end
        
        function w = return_w(obj, new_goal_cond)
            w = obj.w_save(new_goal_cond, :);
        end

    end
end

