function [h_coef,h_coef_noisy,h_fft] = generate_random_channel(SNR)
    ts = 200e-9;        % Tempo de um símbolo
    fs = 1/ts;          % Taxa de amostragem
    fc = 2e9;           % Frequência da portadora
    v = 4/3.6;          % Velocidade em m/s
    L = 9;              % Ordem do canal
    N_FFT = 1024;
    
    c = physconst('LightSpeed'); 
    max_doppler_shift = v*fc/c;
    chan = stdchan('cdmaTUx',fs,max_doppler_shift); 
    
    data = zeros(100,1); % Esse 100 poderia ser qualquer coisa, vamos truncar depois
    data(1) = 1; % Gera impulso
    h_coef = chan(data);
    h_coef = h_coef(7:L+7);
    h_coef = normalize_signal(h_coef);
    
    h_fft = fft(h_coef,N_FFT);
    
    h_coef_noisy = add_noise(h_coef,SNR);
end