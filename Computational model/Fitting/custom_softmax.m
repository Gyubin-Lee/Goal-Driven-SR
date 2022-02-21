function result = custom_softmax(theta, subject_data, Qfun, Qfun_pre)
    
    main_session_data = subject_data.main_session;    
    N = subject_data.num_main_trial;
    y = nan(N*3, 2);
    
    if nargin == 4 % Use pre-session data
        pre_train_data = subject_data.pre_train_session;
        [model_SR, memory] = Qfun_pre(theta, pre_train_data);
        Q = Qfun(model_SR, memory, main_session_data);    
    else % Does not use pre-session data
        % disp("Does not use pre-seesion data");
        model_SR = model_SR_Standard_Lee(9, 2, theta(1), theta(2), 1, theta(3));
        memory = memory_replay_GoalDriven();        
        Q = Qfun(model_SR, memory, main_session_data);
    end
    
    [state, choice] = make_step_list(main_session_data);
       
    for trial = 1:N
        for ind = 3*(trial-1)+1:1:3*trial
            Q_L = Q(trial, state(ind), 1, :);
            Q_R = Q(trial, state(ind), 2, :);
            
            pval = custom_sigmoid(Q_L, Q_R, theta(4), theta(5));
            y(ind, 1) = pval(choice(ind));
        end
    end
    
    y(:, 2) = log(y(:, 1));
    % save('y', 'y');
    result = y(:, 2);
end