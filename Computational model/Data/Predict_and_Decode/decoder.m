%% load CES-D and predicted result
clear;

load('../data.mat');
load('predictor_SVM_result(SR).mat');

%%
CES_D = data.CES_D; 
params = data.result_SR_bias.param;
N_PREDICTED = length(result_cell);
decoded_result = nan(63, N_PREDICTED, 2);

for rs = 1:N_PREDICTED
    predicted_result = result_cell{1, rs};
    test_ind = predicted_result.test_ind;
    train_ind = setdiff(1:63, test_ind);
    CES_D_train = CES_D(train_ind, :);
    params_test = params(test_ind, :);
    params_train = params(train_ind, :);
    
    N_TRAIN_SUB = size(params_train, 1);
    N_PARAM = size(params_train, 2);
    N_TEST_SUB = size(params_test, 1);
    
    [y_test_L, y_test_R]  = deal(zeros(N_TEST_SUB, N_TRAIN_SUB));
    [ces_d_sorted, in] = sort(CES_D_train, 'ascend');
    params_train_aligned = params_train(in, :);
     
    for sub = 1:N_TEST_SUB 

        predicted_L  = predicted_result.y_test_L{1, sub};
        p_val_L = predicted_result.y_test_L{2, sub};
        predicted_R  = predicted_result.y_test_R{1, sub};
        p_val_R = predicted_result.y_test_R{2, sub};
        accumulated_p = zeros(2, N_TRAIN_SUB);

        for train_sub = 1:N_TRAIN_SUB
            % CES-D decoding by y_test_L
            accumulated_p(1, 1:train_sub) = accumulated_p(1, 1:train_sub) + p_val_L(train_sub, 1);
            accumulated_p(1, (train_sub+1):end) = accumulated_p(1,(train_sub+1):end) + p_val_L(train_sub, 2);
            
            % CES-D decoding by y_test_R
            accumulated_p(2, train_sub:end) = accumulated_p(2, train_sub:end) + p_val_R(train_sub, 1);
            accumulated_p(2, 1:(train_sub - 1)) = accumulated_p(2, 1:(train_sub - 1)) + p_val_R(train_sub, 2);
        end
        
        
        [~, argmax] = max(accumulated_p(1, :));
        decoded_result(test_ind(1, sub), rs, 1) = ces_d_sorted(argmax, 1);
        
        [~, argmax] = max(accumulated_p(2, :));
        decoded_result(test_ind(1, sub), rs, 2) = ces_d_sorted(argmax, 1);
        
    end
end

%%
bank = zeros(63, 4);
decoded = cell(1, 63);

for sub = 1:63
    decoded_sub = struct;
    
    decoded_L = decoded_result(sub, :, 1);
    decoded_L = decoded_L(~isnan(decoded_L));
    decoded_sub.decoded_L = decoded_L;
    SE_L = standard_error(decoded_L);
    
    decoded_R = decoded_result(sub, :, 2);
    decoded_R = decoded_R(~isnan(decoded_R));
    decoded_sub.decoded_R = decoded_R;
    SE_R = standard_error(decoded_R);
    
    
    bank(sub, 1) = mean(decoded_L);
    bank(sub, 2) = SE_L;
    bank(sub, 3) = mean(decoded_R);
    bank(sub, 4) = SE_R;
    
    decoded{sub} = decoded_sub;
    
    disp("+++++++++++++++++++++");
    disp(sub);
    disp(decoded_L);
    
end

save("decoded_result(SR)", "decoded", "bank")

%%
corr_mat = nan(2, 2);
for ind = 1:2
    [rho, pval] = corr(bank(:, ind), CES_D, 'Type', 'Pearson');
    corr_mat(ind, 1) = rho;
    corr_mat(ind, 2) = pval;
end
