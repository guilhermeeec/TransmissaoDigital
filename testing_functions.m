layers = [
    featureInputLayer(2048,"Normalization","none","Name","Input")
    fullyConnectedLayer(2048,"Name","Middle")
    fullyConnectedLayer(2048,"Name","End")
    ]; 
%dlnet = train_network_clean(layers,200,'..\databases\train.csv');
mse_metric = test_network(dlnet,'..\databases\test.csv','..\databases\test_original.csv');
