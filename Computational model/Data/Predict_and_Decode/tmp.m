%%
clear;

load('../data.mat');
% load('Naive_SVM_result.mat');

%%
bank = zeros(63, 2);

for sub = 1:63
    predicted = result(sub, :);
    predicted = predicted(~isnan(predicted));
    SE = standard_error(predicted);
    
    bank(sub, 1) = mean(predicted);
    bank(sub, 2) = SE;
    
    disp("+++++++++++++++++++++");
    disp(sub);
    
end

save("decoded_result(Naive_SVM)", "bank")

%%
[rho, pval] = corr(bank(:, 1), data.CES_D, 'Type', 'Pearson');
disp(rho)
disp(pval)
