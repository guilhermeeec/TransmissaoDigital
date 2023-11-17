layers = [
    featureInputLayer(2048,"Normalization","none","Name","Input")
    fullyConnectedLayer(2048,"Name","Middle")
    fullyConnectedLayer(2048,"Name","End")
    ]; 
train_name = '..\databases\train.csv';
train_name_orig = '..\databases\train_orig.csv';
test_name = '..\databases\test.csv';
test_name_orig = '..\databases\test_orig.csv';
poisoned_name = '..\databases\poisoned.csv';
errors_ls = [];
errors_ls_fgsm = [];
errors_nn = [];
errors_nn_fgsm = [];
snr = 30;
for fgsm_power = [0.05 0.1 0.75]
    error_ls = generate_dataset_clean(3000,snr,5,train_name,train_name_orig);
    errors_ls = [errors_ls error_ls]
    
    generate_dataset_clean(1000,snr,5,test_name,test_name_orig);
    
    dlnet = train_network_clean(layers,150,train_name);
    
    error_nn = test_network(dlnet,test_name,test_name_orig);
    errors_nn = [errors_nn error_nn]
    
    error_nn_fgsm = poison_dataset(dlnet,fgsm_power,test_name,test_name_orig,30,0);
    errors_nn_fgsm = [errors_nn_fgsm error_nn_fgsm]
end
