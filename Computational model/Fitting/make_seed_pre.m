function seed = make_seed_pre(lb, ub)
    % make seed with lb and ub
    seed = lb+rand(size(lb)).*(ub-lb);
end

