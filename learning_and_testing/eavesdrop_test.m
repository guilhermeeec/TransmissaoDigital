layers = [
    featureInputLayer(2048,"Normalization","none","Name","Input")
    fullyConnectedLayer(2048,"Name","Middle")
    fullyConnectedLayer(2048,"Name","End")
    ]; 

train_name = '..\databases\train.csv';
train_name_orig = '..\databases\train_orig.csv';
test_name = '..\databases\test.csv';
test_name_orig = '..\databases\test_orig.csv';

fgsm_power = 0.1;
num_epochs = 100;
snr = 30;
r=0;

error_ls = generate_dataset_clean(3000,snr,5,train_name,train_name_orig)
generate_dataset_clean(1000,snr,5,test_name,test_name_orig);

dlnet = train_network_clean(layers,num_epochs,train_name);
error_nn = test_network(dlnet,test_name,test_name_orig)

% Aqui pode dar ruim
error_nn_eavesdrop = poison_eavesdrop(dlnet,fgsm_power,test_name,test_name_orig,snr,r)


