layers = [
    featureInputLayer(2048,"Normalization","none","Name","Input")
    fullyConnectedLayer(2048,"Name","Middle")
    fullyConnectedLayer(2048,"Name","End")
    ]; 
train_name = '..\databases\train.csv';
train_name_orig = '..\databases\train_orig.csv';
test_name = '..\databases\test.csv';
test_name_orig = '..\databases\test_orig.csv';
random_name = '..\databases\random.csv';
random_name_orig = '..\databases\random_orig.csv';
errors_ls_random = [];
errors_nn_random = [];
snr = 30;

generate_dataset_clean(3000,snr,5,train_name,train_name_orig);
dlnet = train_network_clean(layers,150,train_name);

generate_dataset_clean(1000,snr,5,test_name,test_name_orig);
error_nn = test_network(dlnet,test_name,test_name_orig)

for power = [0.001 0.01 0.1]  
    error_ls_random = generate_dataset_attacked(1000,snr,5,power,random_name,random_name_orig);
    errors_ls_random = [errors_ls_random error_ls_random]
    
    error_nn_random = test_network(dlnet,random_name,random_name_orig);
    errors_nn_random = [errors_nn_random error_nn_random]
end