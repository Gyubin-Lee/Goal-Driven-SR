function [goal_cond_state, r_state] = state_to_reward(s_prime)
    switch s_prime
        case 10
            goal_cond_state = 1;
            r_state = 40;
        case 11
            goal_cond_state = 2;
            r_state = 20;
        case 12
            goal_cond_state = 3;
            r_state = 10;
        otherwise
            goal_cond_state = -1;
            r_state = 0;
    end
end

