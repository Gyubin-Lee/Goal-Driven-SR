function [q_dist, a_star] = find_optimal_action_dist(Q, state)
   Q_left = reshape(Q(state, 1, :), [1, 4]);
   Q_right = reshape(Q(state, 2, :), [1, 4]);
   
   if sum(Q_left) > sum(Q_right)
       q_dist = Q_left;
       a_star = 1;
   else
       q_dist = Q_right;
       a_star = 2;
   end
end

