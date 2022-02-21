%%
load("decoded_result(SR).mat");
load("../data");
ces_d = data.CES_D;

%%
figure;
hold on;
scatter(predicted_result, ces_d, [], 'black', 'x');
xlabel("Predicted CES-D");
ylabel("Actual CES-D");
range = 0.04*(1:1000);
p1 = plot(21*ones(1, 1000), range, 'black');
p2 = plot(range, 21*ones(1, 1000), 'black');
% plot(range, 0.29134*range + 11.21, 'green');