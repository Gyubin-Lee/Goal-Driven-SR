classdef model_SR_Standard_Lee
    properties
        normalize_factor = 40;
        Q, w, H;
        num_state;
        num_action;
        alpha_TD, alpha_SR, gamma_TD, gamma_SR;
    end
    
    methods
        function obj = model_SR_Standard_Lee(num_state_, num_action_, alpha_TD_, alpha_SR_, gamma_TD_, gamma_SR_)
        % Initialize model_SR_Standard
            obj.num_state = num_state_;
            obj.num_action = num_action_;
            
            obj.Q = zeros(num_state_, num_action_);
            obj.w = zeros(1, num_state_);
            obj.H = zeros(num_state_, num_state_, num_action_);
            obj.num_state = num_state_;
            
            obj.alpha_TD = alpha_TD_;
            obj.alpha_SR = alpha_SR_;
            obj.gamma_TD = gamma_TD_;
            obj.gamma_SR = gamma_SR_;
        end
        
        function [argv, argmax] = find_optimal_action(~, Q, state)
        % find optimal action of input state
            state_action_value = Q(state, :);
            [argv, argmax] = max(state_action_value);       
        end
        
        function obj = update(obj, episode)
        % update model by episode  
            [row, ~] = size(episode);
            
            for i = row:-1:1
                step = episode(i, :);
                s = step(1);
                a = step(2);
                r = step(3)/obj.normalize_factor;
                s_prime = step(4);
                
                % load previous model martices
                oldH = obj.H;
                oldW = obj.w;
                oldQ = obj.Q;
                
                if s_prime > 9
                    is_terminate = 1;
                else
                    is_terminate = 0;
                end
                
                if is_terminate == 1 % terminal state
                    % Calculate new H
                    newH = oldH;
                    one_state = zeros(1, obj.num_state);
                    one_state(s) = 1;
                    newH(:, s, a) = oldH(:, s, a) + obj.alpha_SR * (one_state.' - oldH(:, s, a));
                    
                    % Calculate new w
                    delta = r - oldQ(s, a);
                    newW = oldW + obj.alpha_TD * delta * oldH(:, s, a).';
                    
                else % non-terminal
                    % Calculate new H
                    newH = oldH;
                    one_state = zeros(1, obj.num_state);
                    one_state(s) = 1;
                    [qval, a_star] = find_optimal_action(obj, oldQ, s_prime);
                    newH(:, s, a) = oldH(:, s, a) + obj.alpha_SR * (one_state.' + obj.gamma_SR * oldH(:, s_prime, a_star) - oldH(:,s, a));

                    % Calculate new w
                    delta = r + obj.gamma_TD*qval - oldQ(s, a);
                    newW = oldW + obj.alpha_TD * delta * oldH(:, s, a)';
                end
                
                % Update model
                for s = 1:9
                    for a = 1:2
                        obj.Q(s, a) = newW * newH(:, s, a);
                    end
                end         
                obj.Q(6:9, 2) = [-999 -999 -999 -999];
                obj.H = newH;
                obj.w = newW;
            end
        end
        
        function obj = change_w(obj, newW) % Only for Goal_driven_SR
            obj.w = newW;
            for s = 1:obj.num_state
                for a = 1:obj.num_action
                    currH = obj.H(:, s, a); % column vector
                    obj.Q(s, a) = newW * currH;
                end
            end
            obj.Q(6:9, 2) = [-999 -999 -999 -999];
        end
    end
end