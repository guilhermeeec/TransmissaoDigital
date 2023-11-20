function error = generate_dataset_clean(N_samples,SNR,K,filename,filename_original)

    N = 1024;           % Número de subportadoras (símbolos por bloco)
    ts = 200e-9;        % Tempo de um símbolo
    fs = 1/ts;          % Taxa de amostragem
    fc = 2e9;           % Frequência da portadora
    v = 4/3.6;          % Velocidade em m/s
    L = 9;              % Ordem do canal
    N_FFT = 1024;

    H = zeros(N_samples, 4*N);
    H_original = zeros(N_samples, 2*N);
    errors = zeros(1, N_samples);
    pilot_symbols = get_pilot();  % Símbolos combinados para estimação de canal
    
    received_signals = zeros(N_samples, 2*N);
    received_signals_filename = '..\databases\received.csv';

    for sample_index = 1:N_samples
        [h_coef,h_coef_noisy,h_FFT,h_FFT_noisy]= generate_random_channel(SNR,v,fc,L,fs,N_FFT);

        [up,pilot_symbols] = generate_transmitter_signal(N,K);                    

        z = fftfilt(h_coef_noisy, up);  
        z = add_noise(z,inf);

        retrieved_symbols = generate_received_signal(z,N,K);

        h_LS = retrieved_symbols./pilot_symbols;

        error = immse(h_LS,h_FFT);
        errors(sample_index) = error;

        h_FFT_noisy_real = real(h_FFT_noisy);
        h_FFT_noisy_imag = imag(h_FFT_noisy);
        h_LS_real = real(h_LS);
        h_LS_imag = imag(h_LS);

        h_features = horzcat(h_LS_real', h_LS_imag');
        h_labels = horzcat(h_FFT_noisy_real', h_FFT_noisy_imag');
        line = horzcat(h_features, h_labels);

        H(sample_index,:) = line;

        h_FFT_real = real(h_FFT);
        h_FFT_imag = imag(h_FFT);

        H_original(sample_index,:) = horzcat(h_FFT_real',h_FFT_imag');
        
        retrieved_symbols_real = real(retrieved_symbols);
        retrieved_symbols_imag = imag(retrieved_symbols);
        received_signals(sample_index,:) = horzcat(retrieved_symbols_real', retrieved_symbols_imag');
    end
    
    error = mean(errors);
    writematrix(H,filename);
    writematrix(H_original,filename_original);
    writematrix(received_signals,received_signals_filename);
end
