layers = [
    featureInputLayer(2048,"Normalization","none","Name","Input")
    fullyConnectedLayer(2048,"Name","Middle")
    fullyConnectedLayer(2048,"Name","End")
    ]; 
train_name = '..\databases\train.csv';
train_name_orig = '..\databases\train_orig.csv';
test_name = '..\databases\test.csv';
test_name_orig = '..\databases\test_orig.csv';

errors_nn = [];
errors_nn_fgsm = [];

errors_nn_robust = [];
errors_nn_robust_fgsm = [];

errors_nn_enhanced = [];
errors_nn_enhanced_fgsm = [];

snr = 30;
%generate_dataset_clean(3000,snr,5,train_name,train_name_orig);
%generate_dataset_clean(1000,snr,5,test_name,test_name_orig);

% Rede tradicional
%dlnet_trad = train_network_clean(layers,num_epochs,train_name);
    
%error_nn = test_network(dlnet_trad,test_name,test_name_orig);
%errors_nn = [errors_nn error_nn]

%for fgsm_power = [0.05 0.1 0.75]  
for fgsm_power = [0.75]    
    
    error_nn_fgsm = poison_dataset(dlnet_trad,fgsm_power,test_name,test_name_orig,snr,0);
    errors_nn_fgsm = [errors_nn_fgsm error_nn_fgsm]
    
    % Rede robusta
    dlnet_robust = train_network_adversarial(layers,num_epochs,train_name,1.2*fgsm_power,0,0,snr);
    
    error_nn_robust = test_network(dlnet_robust,test_name,test_name_orig);
    errors_nn_robust = [errors_nn_robust error_nn_robust]
    
    error_nn_robust_fgsm = poison_dataset(dlnet_robust,fgsm_power,test_name,test_name_orig,snr,0);
    errors_nn_robust_fgsm = [errors_nn_robust_fgsm error_nn_robust_fgsm]
    
    % Rede melhorada
    dlnet_enhanced = train_network_adversarial(layers,num_epochs,train_name,1.2*fgsm_power,0.2,0.2,snr);
    
    error_nn_enhanced = test_network(dlnet_enhanced,test_name,test_name_orig);
    errors_nn_enhanced = [errors_nn_enhanced error_nn_enhanced]
    
    error_nn_enhanced_fgsm = poison_dataset(dlnet_enhanced,fgsm_power,test_name,test_name_orig,snr,0.2);
    errors_nn_enhanced_fgsm = [errors_nn_enhanced_fgsm error_nn_enhanced_fgsm]
   
end
