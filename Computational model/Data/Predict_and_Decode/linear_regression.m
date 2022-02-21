%%
clear;
close all;

%%
warning('off','all')

%%
load("../data");
load("decoded_result(SR).mat");

params = data.result_SR_bias.param;
ces_d = data.CES_D;

%%
mdl = fitlm(bank(:, 1),ces_d);
disp(mdl.Coefficients);