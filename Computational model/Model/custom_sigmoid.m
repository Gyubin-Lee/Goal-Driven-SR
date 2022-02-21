function pval = custom_sigmoid(Q_L, Q_R, beta, bias)
% return action probability with sigmoid function    
    p_left = 1/(1 + exp(-beta*(Q_L - Q_R + bias)));
    pval = [p_left 1-p_left];
end