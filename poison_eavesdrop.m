function [mse_metric,ls_metric] = poison_eavesdrop(dlnet,fgsm_power,test_name,test_name_orig,SNR,r)
    
    % Lê base de dados com estimativa LS nas labels
    H = readmatrix(test_name);
    x_test = H(:,1:2048);
    y_test = H(:,2049:end);
    y_test_orig_ds = arrayDatastore(y_test);
    x_test_ds = arrayDatastore(x_test);
   
    % Lê base de dados com canal real
    y_test_orig = readmatrix(test_name_orig);
    %y_test_orig_ds = arrayDatastore(y_test_orig);
    
    % Estimativa LS x canal (com ruído)
    ds_test_orig = combine(x_test_ds, y_test_orig_ds);
    mini_batch_size = 50;
    mbq_test = minibatchqueue(ds_test_orig, ...
    'MiniBatchSize',mini_batch_size,...
    'MiniBatchFormat',{'BC','BC'});

    % Passa a estimativa LS pela rede (gera estimativa NN)
    y_pred_test = model_predictions(dlnet,mbq_test); 
    y_pred_test = double(extractdata(y_pred_test))';
    
    % Estimativa LS x Estimativa NN
    y_pred_test_ds = arrayDatastore(y_pred_test);
    x_test_ds = arrayDatastore(x_test);
    ds_test_pred = combine(x_test_ds, y_pred_test_ds);
    mini_batch_size = 50;
    mbq_test_fgsm = minibatchqueue(ds_test_pred, ...
    'MiniBatchSize',mini_batch_size,...
    'MiniBatchFormat',{'BC','BC'});
    
    % Ataque FGSM usando a estimativa NN como ground truth
    num_adv_iter = 1;
    epsilon = fgsm_power * (1-r)^(SNR/5-1);
    alpha = 1.25 * epsilon; 
    perturbations = adversarial_examples_eavesdrop(dlnet,mbq_test_fgsm,epsilon,alpha,num_adv_iter);
    
    % Lê do arquivo os sinais do usuário recebidos 
    N = 1024;
    K = 5; 
    L = 10;
    received = readmatrix('..\databases\received.csv'); %ok
    retrieved_users = zeros(size(received,1),N);
    for n=1:N
        retrieved_users(:,n) = received(:,n) + 1j*received(:,n+N);
    end
    
    % Envia sinal pelo canal
    reset(mbq_test);
    index = 1;
    H = zeros(size(x_test,1), 2*N);
    
    % Trocar varredura ao longo dos minibatches por varredura item a item
    while hasdata(mbq_test)
        
        [~, dl_y_test_mbq] = next(mbq_test);
        
        for i = 1:size(dl_y_test_mbq,2)
            dl_y_test = double(extractdata(dl_y_test_mbq(:,i)));
            

            % Pega canal real do conjunto de dados
            h_fft = zeros(N,1);
            for n=1:N
                h_fft(n) = dl_y_test(n)+1j*dl_y_test(n+N);
            end
            h = ifft(h_fft,N);
            h = h(1:L);
            
            pert = double(extractdata(perturbations(:,index)));
            perturbation = zeros(N,1);
            for n=1:N
                perturbation(n) = pert(n)+1j*pert(n+N);
            end

            % Multiplica ponto a ponto piloto com perturbação e divide ponto a
            % ponto pela resposta em frequência do canal
            [~,pilot_symbols] = generate_transmitter_signal(N,K);
            signal = (perturbation.*pilot_symbols)./h_fft;

            % Gera sinal real (não aproximado)
            %s = var * normalize_signal(perturbation);
            s = signal;
            s_time = sqrt(N)*ifft(s,N);
            A_cp = [zeros(K,N-K) eye(K); eye(N)];
            s_time = A_cp * s_time;

            % Envia sinal aproximado pelo canal
            z = fftfilt(h, s_time);
            retrieved_attacker = generate_received_signal(z,N,K); 
            
            % Mistura com sinal do usuário
            retrieved_user = retrieved_users(index,:).';
            retrieved_symbols = retrieved_user + retrieved_attacker;
            %retrieved_symbols = retrieved_user;
            
            % Faz estimação de canal
            h_LS = retrieved_symbols./pilot_symbols;

            % Monta vetor de entrada da rede
            h_LS_real = real(h_LS);
            h_LS_imag = imag(h_LS);
            h_features = horzcat(h_LS_real', h_LS_imag');
            H(index,:) = h_features;

            index = index + 1;
        end
    end
    
    ls_metric = immse(y_test_orig,H);
    
    % Estimativa LS envenenada x canal real
    H_ds = arrayDatastore(H);
    ds_test_poisoned = combine(H_ds, y_test_orig_ds);
    mini_batch_size = 50;
    mbq_poisoned = minibatchqueue(ds_test_poisoned, ...
    'MiniBatchSize',mini_batch_size,...
    'MiniBatchFormat',{'BC','BC'});

    % Testa rede
    y_pred_test = model_predictions(dlnet,mbq_poisoned);
    mse_metric = immse(double(extractdata(y_pred_test)), y_test_orig');
    
    
end