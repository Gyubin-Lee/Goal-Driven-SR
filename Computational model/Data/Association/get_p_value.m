index = 1;

if index == 1
    load('SVM_parameter_association_test_result.mat');
else
    load('SVM_parameter_association_test(arb)_result.mat');
end

N_TEST = size(d_array, 2);
N_PARAM = size(d_array, 1);

result = zeros(2, N_PARAM);

for i = 1:N_PARAM
    for j = 1:N_TEST
        if d_array(i, j) > 0
            result(1, i) = result(1, i) + 1;
        end
    end
    result(2, i) = 1 - result(1,i)/N_TEST;
end
    




