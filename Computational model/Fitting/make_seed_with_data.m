function seed = make_seed_with_data(subject_pre_data, num_param)
    
    max_param = max(subject_pre_data);
    max_param = max_param(1:num_param);
    min_param = min(subject_pre_data);
    min_param = min_param(1:num+param);
    seed = min_param + rand(num_param).*(max_param - min_param);
    
end

