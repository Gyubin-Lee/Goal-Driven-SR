classdef model_MF_Standard
    % Q-learning model
    properties
        num_state, num_action;
        Q;
        alpha, gamma, beta;
        reward_normalize_factor = 1;
    end
    
    methods
        function obj = model_MF_Standard(num_state_, num_action_, alpha_, gamma_, beta_, reward_normalize_factor_)
            % Initialize model_MF_Standard
            obj.num_state = num_state_;
            obj.num_action = num_action_;
            obj.Q = zeros(num_state_, num_action_);
            obj.alpha = alpha_;
            obj.gamma = gamma_;
            obj.beta = beta_;
            if nargin == 6
                obj.reward_normalize_factor = reward_normalize_factor_;
            end
        end
        
        function [argv, argmax] = find_optimal_action(obj, state)
        % find optimal action of input state
            state_action_value = obj.Q(state, :);
            [argv, argmax] = max(state_action_value);   
        end
        
        function obj = update(obj, episode)
        % update model by episode  
            [row, ~] = size(episode);
            
            for i = row:-1:1
                step = episode(i, :);
                s = step(1);
                a = step(2);
                r = step(3)/obj.reward_normalize_factor;
                s_prime = step(4);
                
                if s_prime > 9
                    is_terminate = 1;
                else
                    is_terminate = 0;
                end
                
                old_qval = obj.Q(s, a);
                
                if is_terminate == 1 % terminal state
                    obj.Q(s, a) = old_qval + obj.alpha * (r - old_qval);
                    
                else % non-terminal
                    [qval, ~] = obj.find_optimal_action(s_prime);
                    obj.Q(s, a) = old_qval + obj.alpha * (r + obj.gamma * qval - old_qval);                   
                end

            end
        end
        
        function pval = return_action_prob(obj, s)
        % return action probability
            Q_val_L = obj.Q(s, 1);
            Q_val_R = obj.Q(s, 2);
            
            pval = [custom_sigmoid(obj.beta, Q_val_L, Q_val_R) custom_sigmoid(obj.beta, Q_val_R, Q_val_L)];           
        end
       
        function a = return_action(obj, s)
        % return action
            pval = obj.return_action_prob(s);
            a = randsample(2, 1, true, pval); 
        end
    end
end

