function mse_metric = poison_dataset(dlnet,fgsm_power,test_name,test_name_orig)
    H = readmatrix(test_name);
    x_test = H(:,1:2048);
    y_test = H(:,2049:end);
    mini_batch_size = 50;
    
    y_test_orig = readmatrix(test_name_orig);
    
    x_test_ds = arrayDatastore(x_test);
    y_test_orig_ds = arrayDatastore(y_test_orig);
    ds_test_orig = combine(x_test_ds, y_test_orig_ds);
    
    mbq_test_fgsm = minibatchqueue(ds_test_orig, ...
    'MiniBatchSize',mini_batch_size,...
    'MiniBatchFormat',{'BC','BC'});

    epsilon = fgsm_power;
    alpha = fgsm_power;
    num_adv_iter = 1;
    
    [x_adv, y_pred_adv] = adversarial_examples(dlnet,mbq_test_fgsm,epsilon,alpha,num_adv_iter);

    mse_metric = immse(double(extractdata(y_pred_adv)), y_test_orig');
end