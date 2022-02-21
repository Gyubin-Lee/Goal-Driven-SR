function fitresult = optimize_hyperparams(params_train, CES_D_train)

    ks = optimizableVariable('kernel_size',[1e-2, 10]);
    bc = optimizableVariable('box_constraint',[1e-2, 5]);
    fitfunction = @(theta) (-1)*acc_cross_validation(theta.kernel_size, theta.box_constraint, params_train, CES_D_train, 10);
    fitresult = bayesopt(fitfunction, [ks, bc],...
        'Useparallel', true, 'MaxObjectiveEvaluations', 200, 'Verbose', 0);
    close all;
    
end