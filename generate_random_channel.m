function [h_coef,h_coef_noisy,h_fft,h_FFT_noisy] = generate_random_channel(SNR,v,fc,L,fs,N_FFT)
    c = physconst('LightSpeed'); 
    max_doppler_shift = v*fc/c;
    chan = stdchan('cdmaTUx',fs,max_doppler_shift); 
    
    data = zeros(100,1); % Esse 100 poderia ser qualquer coisa, vamos truncar depois
    data(1) = 1; % Gera impulso
    h_coef = chan(data);
    h_coef = h_coef(7:L+7);
    h_coef = normalize_signal(h_coef);
    
    h_coef_noisy = add_noise(h_coef,SNR);
    
    h_fft = fft(h_coef,N_FFT);
    h_FFT_noisy = fft(h_coef_noisy,N_FFT);
end