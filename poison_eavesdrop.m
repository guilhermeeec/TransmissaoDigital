function mse_metric = poison_eavesdrop(dlnet,fgsm_power,test_name,test_name_orig,SNR,r)
    H = readmatrix(test_name);
    x_test = H(:,1:2048);
    %y_test = H(:,2049:end);
   
    y_test_orig = readmatrix(test_name_orig);
    
    x_test_ds = arrayDatastore(x_test);
    y_test_orig_ds = arrayDatastore(y_test_orig);
    
    % Estimativa LS x canal real
    ds_test_orig = combine(x_test_ds, y_test_orig_ds);
    mini_batch_size = 50;
    mbq_test = minibatchqueue(ds_test_orig, ...
    'MiniBatchSize',mini_batch_size,...
    'MiniBatchFormat',{'BC','BC'});


    % Passa a estimativa LS pela rede (gera estimativa NN)
    y_pred_test = model_predictions(dlnet,mbq_test); 
    
    
    % Estimativa LS x Estimativa NN
    y_pred_test_ds = arrayDatastore(y_pred_test);
    ds_test_pred = combine(x_test_ds, y_pred_test_ds);
    mini_batch_size = 50;
    mbq_test_fgsm = minibatchqueue(ds_test_pred, ...
    'MiniBatchSize',mini_batch_size,...
    'MiniBatchFormat',{'BC','BC'});
    
    % Ataque FGSM
    num_adv_iter = 1;
    epsilon = fgsm_power * (1-r)^(SNR/5-1);
    alpha = 1.25 * epsilon; 
    pertubations = adversarial_examples_eavesdrop(dlnet,mbq_test_fgsm,epsilon,alpha,num_adv_iter);
    
    % Lê sinais recebidos no arquivo
    received = readmatrix('..\databases\received.csv');
    z_user = zeros(N,1);
    for n=1:N
        z_user(n) = received(n)+1j*received(n+N);
    end
    z_user_time = ifft(z_user,N);
    
    % Envia sinal pelo canal
    reset(mbq_test);
    N = 1024;
    K = 5; 
    index = 1;
    H = zeros(size(x_test,1), 2*N);
    while hasdata(mbq_test)
        
        % Pega canal real do conjunto de dados
        [~, dl_y_test] = next(mbq_test);
        h_fft = zeros(N,1);
        for n=1:N
            h_fft(n) = dl_y_test(n)+1j*dl_y_test(n+N);
        end
        h = ifft(h_ftt,N);
        
        % Multiplica ponto a ponto piloto, perturbação e divide ponto a
        % ponto pela resposta em frequência do canal
        [~,pilot_symbols] = generate_transmitter_signal(N,K);
        xp = fft(pilot_symbols,N);
        signal = (pertubations.*xp)./h_fft;
        var = signal_power(signal);
        
        % Gera sinal aproximado
        s = var * normalize_signal(pertubations(index));
        s_time = ifft(s,N);
        
        % Envia sinal aproximado pelo canal e mistura com sinal do usuário
        z = fftfilt(h, s_time)+z_user_time;  
        retrieved_symbols = generate_received_signal(z,N,K);
        
        % Faz estimação de canal
        h_LS = retrieved_symbols./pilot_symbols;
        
        % Monta vetor de entrada da rede
        h_LS_real = real(h_LS);
        h_LS_imag = imag(h_LS);
        h_features = horzcat(h_LS_real', h_LS_imag');
        H(index,:) = h_features;
        
        index = index + 1;
    end
    
    % Estimativa LS envenenada x canal real
    ds_test_poisoned = combine(H, y_test_orig_ds);
    mini_batch_size = 50;
    mbq_poisoned = minibatchqueue(ds_test_poisoned, ...
    'MiniBatchSize',mini_batch_size,...
    'MiniBatchFormat',{'BC','BC'});

    % Testa rede
    y_pred_test = model_predictions(dlnet,mbq_poisoned);
    mse_metric = immse(double(extractdata(y_pred_test)), y_test_orig');
    
    
end