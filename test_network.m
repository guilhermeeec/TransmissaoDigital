function mse_metric = test_network(dlnet,test_dataset,test_dataset_original)
    H = readmatrix(test_dataset);
    x_test = H(:,1:2048);
    y_test = H(:,2049:end);
    x_test_ds = arrayDatastore(x_test);
    y_test_ds = arrayDatastore(y_test);
    ds_test = combine(x_test_ds, y_test_ds);
    mini_batch_size = 50;
    
    mbq_test = minibatchqueue(ds_test, ...
    'MiniBatchSize',mini_batch_size,...
    'MiniBatchFormat',{'BC','BC'});
    
    y_pred_test = model_predictions(dlnet,mbq_test);
    
    y_test_orig = readmatrix(test_dataset_original);
    mse_metric = immse(double(extractdata(y_pred_test)), y_test_orig');
end