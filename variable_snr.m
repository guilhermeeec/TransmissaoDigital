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

errors_ls = [];
errors_ls_random = [];
errors_ls_fgsm = [];
errors_ls_fgsm_robust = [];
errors_ls_fgsm_enhanced = [];

errors_nn = [];
errors_nn_random = [];
errors_nn_fgsm = [];

errors_nn_robust = [];
errors_nn_robust_random = [];
errors_nn_robust_fgsm = [];

errors_nn_enhanced = [];
errors_nn_enhanced_random = [];
errors_nn_enhanced_fgsm = [];

errors_nn_enhanced_eavesdrop = [];
errors_ls_enhanced_eavesdrop = [];

fgsm_power = 0.1;
num_epochs = 200;

for snr = [20 30 40]
%for snr = [30]
    error_ls = generate_dataset_clean(3000,snr,5,train_name,train_name_orig);
    errors_ls = [errors_ls error_ls]
    
    generate_dataset_clean(1000,snr,5,test_name,test_name_orig);
    
    random_power = 0.01;
    error_ls_random = generate_dataset_attacked(1000,snr,5,random_power,random_name,random_name_orig);
    errors_ls_random = [errors_ls_random error_ls_random]
    
    % Rede tradicional
    dlnet_trad = train_network_clean(layers,num_epochs,train_name);
    
    error_nn = test_network(dlnet_trad,test_name,test_name_orig);
    errors_nn = [errors_nn error_nn]
    
    error_nn_random = test_network(dlnet_trad,random_name,random_name_orig);
    errors_nn_random = [errors_nn_random error_nn_random]
    
    [error_nn_fgsm,error_ls_fgsm] = poison_dataset(dlnet_trad,fgsm_power,test_name,test_name_orig,snr,0);
    errors_ls_fgsm = [errors_ls_fgsm error_ls_fgsm]
    errors_nn_fgsm = [errors_nn_fgsm error_nn_fgsm]
    filename = sprintf('../workspaces/backup_%g.mat', snr);
    save(filename);
    
    % Rede robusta
    dlnet_robust = train_network_adversarial(layers,num_epochs,train_name,1.2*fgsm_power,0,0,snr);
    
    error_nn_robust = test_network(dlnet_robust,test_name,test_name_orig);
    errors_nn_robust = [errors_nn_robust error_nn_robust]
    
    error_nn_robust_random = test_network(dlnet_robust,random_name,random_name_orig);
    errors_nn_robust_random = [errors_nn_robust_random error_nn_robust_random]
    
    [error_nn_robust_fgsm,error_ls_fgsm_robust] = poison_dataset(dlnet_robust,fgsm_power,test_name,test_name_orig,snr,0);
    errors_ls_fgsm_robust = [errors_ls_fgsm_robust error_ls_fgsm_robust]
    errors_nn_robust_fgsm = [errors_nn_robust_fgsm error_nn_robust_fgsm]
    filename = sprintf('../workspaces/backup_%g.mat', snr);
    save(filename);
    
    % Rede melhorada lambda=0.2 e r=0.2 
    dlnet_enhanced = train_network_adversarial(layers,num_epochs,train_name,1.2*fgsm_power,0.2,0.2,snr);
    
    error_nn_enhanced = test_network(dlnet_enhanced,test_name,test_name_orig);
    errors_nn_enhanced = [errors_nn_enhanced error_nn_enhanced]
    
    random_power = 0.01*(1-0.2)^(snr/5);
    generate_dataset_attacked(1000,snr,5,random_power,random_name,random_name_orig);
    
    error_nn_enhanced_random = test_network(dlnet_enhanced,random_name,random_name_orig);
    errors_nn_enhanced_random = [errors_nn_enhanced_random error_nn_enhanced_random]
    
    [error_nn_enhanced_fgsm,error_ls_fgsm_enhanced] = poison_dataset(dlnet_enhanced,fgsm_power,test_name,test_name_orig,snr,0.2);
    errors_ls_fgsm_enhanced = [errors_ls_fgsm_enhanced error_ls_fgsm_enhanced]
    errors_nn_enhanced_fgsm = [errors_nn_enhanced_fgsm error_nn_enhanced_fgsm]
    
    [error_nn_eavesdrop_enhanced,error_ls_eavesdrop_enhanced] = poison_eavesdrop(dlnet_enhanced,0.1,test_name,test_name_orig,snr,0.2);
    errors_nn_enhanced_eavesdrop = [errors_nn_enhanced_eavesdrop error_nn_eavesdrop_enhanced]
    errors_ls_enhanced_eavesdrop = [errors_ls_enhanced_eavesdrop error_ls_eavesdrop_enhanced]
    
    filename = sprintf('../workspaces/backup_%g.mat', snr);
    save(filename);
    
end
