function r = goal_cond_to_reward(goal_cond)
    switch goal_cond
        case 1
            r = 40;
        case 2
            r = 20;
        case 3
            r = 10;
        otherwise
            r = 40;
    end
end

