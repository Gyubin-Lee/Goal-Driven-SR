function [acc, acc_list] = acc_cross_validation(kernel_scale, box_constraint, train_data, test_data, k)
    
    N_DATA = size(train_data ,1);
    N_PARAM = size(train_data, 2);
    N_PAIR = N_DATA*(N_DATA-1);

    X = zeros(N_PAIR, 2*N_PARAM);
    y = zeros(N_PAIR,1);
    Z = zeros(N_PAIR,2);
    
    c = 0;
    for i = 1:N_DATA
        for j = 1:N_DATA
            if (i~=j)
                c = c+1;
                X(c,:) = [train_data(i, :) train_data(j, :)];
                y(c,:) = test_data(i) > test_data(j);
                Z(c,:) = [i,j];
            end
        end
    end
    
    acc_list = zeros(1, k);
    for iter = 1:k
        validation_idx = nonzeros(round((iter-1)*N_DATA/k):round(iter*N_DATA/k));
        train_idx = setdiff(1:N_DATA, validation_idx);
        train_mask = (ismember(Z(:,1),train_idx))&(ismember(Z(:,2),train_idx));
        validation_mask = ~train_mask;

        X_train = X(train_mask,:);
        X_val = X(validation_mask,:);
               
        mdl = fitcsvm(X_train, y(train_mask, 1), ...        
         'kernelfunction','rbf',...
         'Standardize', true,...
         'KernelScale', kernel_scale, 'BoxConstraint', box_constraint);
        
        y_val = predict(mdl,X_val);
        is_correct = (y_val == y(validation_mask,1));
        acc_list(iter) = sum(is_correct)/length(is_correct);
           
    end
    
    acc = mean(acc_list);
    
end

