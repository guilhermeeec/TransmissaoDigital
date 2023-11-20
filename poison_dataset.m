function [mse_metric,ls] = poison_dataset(dlnet,fgsm_power,test_name,test_name_orig,SNR,r)
    H = readmatrix(test_name);
    x_test = H(:,1:2048);
    y_test = H(:,2049:end);
    
    y_test_orig = readmatrix(test_name_orig);
    
    x_test_ds = arrayDatastore(x_test);
    y_test_orig_ds = arrayDatastore(y_test_orig);
    ds_test_orig = combine(x_test_ds, y_test_orig_ds);
    
    mini_batch_size = 50;
    
    mbq_test_fgsm = minibatchqueue(ds_test_orig, ...
    'MiniBatchSize',mini_batch_size,...
    'MiniBatchFormat',{'BC','BC'});

    %epsilon = fgsm_power;
    %alpha = fgsm_power;
    num_adv_iter = 1;
    
    epsilon = fgsm_power * (1-r)^(SNR/5-1);
    alpha = 1.25 * epsilon; 
    
    [x_adv, y_pred_adv] = adversarial_examples(dlnet,mbq_test_fgsm,epsilon,alpha,num_adv_iter);
    ls = immse(double(extractdata(x_adv)),y_test_orig');

    mse_metric = immse(double(extractdata(y_pred_adv)), y_test_orig');
end